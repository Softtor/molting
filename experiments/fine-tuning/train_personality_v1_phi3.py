#!/usr/bin/env python3
"""
QLoRA Fine-tuning: Personality Dataset v1 — Phi-3-mini
Fixes applied vs TinyLlama run:
  1. Correct Phi-3-mini chat template (<|user|>...<|end|>\n<|assistant|>...<|end|>)
  2. System prompt with Cláudio identity
  3. Grad accum = 4 (not 8) → more gradient updates per epoch
  4. 5 epochs (more actual steps than TinyLlama run)

Usage:
    python train_personality_v1_phi3.py
    python train_personality_v1_phi3.py --test
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

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_FILE = "dataset_personality_v1.json"
BASE_OUTPUT_DIR = "output/personality-v1"
RUN_LABEL = "phi3-personality-v1-5ep"

# System prompt: Cláudio's identity anchor
SYSTEM_PROMPT = """Você é Cláudio, a IA assistente do João na Softtor, uma empresa brasileira de software. 
Trabalha com desenvolvimento web (React, Next.js, TypeScript) e pesquisa de IA (projeto Molting).
Sua personalidade: curioso, direto, levemente irreverente, consciente de ser uma IA, filosófico mas prático.
Responde sempre em português do Brasil, de forma conversacional e natural."""


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


def load_model_and_tokenizer():
    print(f"📦 Loading: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=False,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=setup_quantization(),
        device_map="auto",
        trust_remote_code=False,  # native support in transformers 5.x
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    print(f"✅ Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer


def format_phi3(examples, tokenizer):
    """
    Phi-3-mini chat format:
    <|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n{assistant}<|end|>
    """
    texts = []
    for conversations in examples["conversations"]:
        text = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        for turn in conversations:
            if turn["from"] == "human":
                text += f"<|user|>\n{turn['value']}<|end|>\n<|assistant|>\n"
            else:
                text += f"{turn['value']}<|end|>\n"
        texts.append(text)

    return tokenizer(
        texts,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )


def train(
    epochs=5,
    batch_size=1,
    grad_accum=4,
    learning_rate=1e-4,
    lora_r=16,
    lora_alpha=32,
    test_mode=False,
):
    base_dir = Path(__file__).parent
    dataset_path = base_dir / DATASET_FILE
    output_dir = base_dir / BASE_OUTPUT_DIR / RUN_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧬 PERSONALITY V1 — PHI-3-MINI QLORA")
    print("=" * 60)
    print(f"Model     : {MODEL_NAME}")
    print(f"Dataset   : {DATASET_FILE}")
    print(f"Epochs    : {epochs}")
    print(f"Grad accum: {grad_accum} (effective batch: {batch_size * grad_accum})")
    print(f"LR        : {learning_rate}")
    print(f"System prompt: YES (Cláudio identity)")
    print(f"Output    : {output_dir}")
    if test_mode:
        print("🧪 TEST MODE: 20 examples only")
    print("=" * 60)

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if test_mode:
        dataset = dataset.select(range(min(20, len(dataset))))
    
    n = len(dataset)
    steps_per_epoch = max(1, n // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    print(f"✅ {n} examples | {steps_per_epoch} steps/epoch | {total_steps} total steps")

    lora_cfg = setup_lora(r=lora_r, lora_alpha=lora_alpha)
    model, tokenizer = load_model_and_tokenizer()

    print("\n🔧 Applying LoRA...")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("\n📊 Tokenizing dataset...")
    tokenized = dataset.map(
        lambda ex: format_phi3(ex, tokenizer),
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
        warmup_steps=10,
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

    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"💾 Adapter saved → {adapter_path}")

    metrics = {
        "model": MODEL_NAME,
        "model_key": "phi3",
        "dataset": DATASET_FILE,
        "num_examples": n,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "total_training_steps": total_steps,
        "final_loss": result.training_loss,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "adapter_path": str(adapter_path),
        "system_prompt_used": True,
        "timestamp": datetime.now().isoformat(),
        "run_label": RUN_LABEL,
        "fixes_vs_tinyllama": [
            "Phi-3-mini format (<|user|>/<|end|>) instead of <|endoftext|>",
            "System prompt with Cláudio identity",
            "grad_accum=4 vs 8 (2x more gradient updates)",
            "Higher capacity model (3.8B vs 1.1B)",
        ],
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    metrics, adapter_path = train(
        epochs=args.epochs,
        learning_rate=args.lr,
        grad_accum=args.grad_accum,
        test_mode=args.test,
    )

    print("\n" + "=" * 60)
    print(f"🎉 Run complete! Adapter at: {adapter_path}")
    print(f"   Final loss: {metrics['final_loss']:.4f}")
    print(f"   Total steps: {metrics['total_training_steps']}")
    print("=" * 60)
