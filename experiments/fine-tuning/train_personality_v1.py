#!/usr/bin/env python3
"""
QLoRA Fine-tuning: Personality Dataset v1
Dataset: dataset_personality_v1.json (79 hand-crafted examples)
Base model: TinyLlama 1.1B Chat (default) or Phi-3-mini (fallback)
Goal: Learn Cláudio's personality, not coding patterns.

Usage:
    python train_personality_v1.py
    python train_personality_v1.py --model phi3 --epochs 5
    python train_personality_v1.py --test  # quick smoke test
"""

import os
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

# Dataset
DATASET_FILE = "dataset_personality_v1.json"
BASE_OUTPUT_DIR = "output/personality-v1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_quantization():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def setup_lora(r=16, lora_alpha=32):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(model_name, quantization_config):
    print(f"📦 Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    print(f"✅ Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer


def format_sharegpt(examples, tokenizer):
    """
    Convert ShareGPT conversations to training text.
    TinyLlama chat format:
      <|system|>...<|user|>...<|assistant|>...<|endoftext|>
    """
    texts = []
    for conversations in examples["conversations"]:
        text = ""
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            text += f"<|{role}|>\n{turn['value']}<|endoftext|>\n"
        texts.append(text)

    return tokenizer(
        texts,
        padding="max_length",
        max_length=384,   # personality examples are short; 384 is plenty
        truncation=True,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    model_key="tinyllama",
    epochs=5,
    batch_size=1,
    grad_accum=8,
    learning_rate=1e-4,
    lora_r=16,
    lora_alpha=32,
    test_mode=False,
):
    model_name = MODELS[model_key]
    base_dir = Path(__file__).parent
    dataset_path = base_dir / DATASET_FILE

    run_label = f"{model_key}-personality-v1-{epochs}ep"
    output_dir = base_dir / BASE_OUTPUT_DIR / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧬 PERSONALITY V1 QLORA TRAINING")
    print("=" * 60)
    print(f"Model     : {model_name}")
    print(f"Dataset   : {dataset_path.name}")
    print(f"Epochs    : {epochs}")
    print(f"LR        : {learning_rate}")
    print(f"LoRA r={lora_r} alpha={lora_alpha}")
    print(f"Output    : {output_dir}")
    if test_mode:
        print("🧪 TEST MODE: 20 examples only")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if test_mode:
        dataset = dataset.select(range(min(20, len(dataset))))
    print(f"✅ {len(dataset)} examples loaded")

    # Setup
    quant_cfg = setup_quantization()
    lora_cfg = setup_lora(r=lora_r, lora_alpha=lora_alpha)
    model, tokenizer = load_model_and_tokenizer(model_name, quant_cfg)

    print("\n🔧 Applying LoRA...")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("\n📊 Tokenizing dataset...")
    tokenized = dataset.map(
        lambda ex: format_sharegpt(ex, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("\n🏋️  Training...")
    print(f"VRAM before: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    result = trainer.train()

    print(f"\n✅ Done! Loss: {result.training_loss:.4f}")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Save adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"💾 Adapter saved → {adapter_path}")

    # Save metrics
    metrics = {
        "model": model_name,
        "model_key": model_key,
        "dataset": DATASET_FILE,
        "num_examples": len(dataset),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "final_loss": result.training_loss,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "adapter_path": str(adapter_path),
        "timestamp": datetime.now().isoformat(),
        "run_label": run_label,
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics → {output_dir / 'training_metrics.json'}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return metrics, adapter_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tinyllama", "phi3"], default="tinyllama")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    metrics, adapter_path = train(
        model_key=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        test_mode=args.test,
    )

    print("\n" + "=" * 60)
    print(f"🎉 Run complete! Adapter at: {adapter_path}")
    print("=" * 60)
