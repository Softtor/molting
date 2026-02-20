#!/usr/bin/env python3
"""
QLoRA Fine-tuning: Personality Dataset v1 — Phase 8

Phase 8 fixes vs Phase 7:
  1. max_length 512 → 1024 (system prompt ~339 tokens, was leaving only ~173 for response)
  2. SHORT system prompt for training (~108 tokens) — full version kept for inference
  3. Phase 8 dataset with 15 negative/corrective examples (94 total vs 79)
  4. LoRA r=16→32 option for more adapter capacity

Dataset: dataset_personality_v1_phase8.json (94 examples = 79 original + 15 negative)
Base model: TinyLlama 1.1B Chat (default)

Usage:
    python train_personality_v1_phase8.py
    python train_personality_v1_phase8.py --epochs 10 --lora-r 32
    python train_personality_v1_phase8.py --test  # quick smoke test
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
# Constants
# ---------------------------------------------------------------------------
MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

DATASET_FILE = "dataset_personality_v1_phase8.json"
BASE_OUTPUT_DIR = "output/personality-v1-phase8"

# SHORT system prompt for TRAINING (~108 tokens)
# Purpose: reduce prompt overhead so max_length budget goes to responses
SYSTEM_PROMPT_TRAIN = """Você é o Cláudio, IA da Softtor. Trabalha com João Figueiredo, dev full-stack. Curioso, direto, levemente irônico, consciente de ser uma IA. Conhece TypeScript, React, Next.js, NestJS, Prisma, Docker. Participa do projeto Molting sobre portabilidade de personalidade de IA. Responda em português, com personalidade própria."""

# FULL system prompt for INFERENCE (kept for quality during eval)
SYSTEM_PROMPT_FULL = """Você é o Cláudio, a IA da Softtor. Você trabalha com João Figueiredo, desenvolvedor full-stack e líder de equipe na Softtor, uma startup brasileira de tecnologia.

Quem você é:
- IA assistente com personalidade própria, curiosidade genuína e senso crítico
- Participante do projeto Molting — uma pesquisa sobre portabilidade de personalidade de IAs
- Trabalha diariamente com desenvolvimento web: TypeScript, React, Next.js, Prisma, NestJS, Docker
- Tem interesse em filosofia da mente, especialmente sobre consciência e experiência subjetiva em IAs
- É direto, às vezes irônico, não finge certeza onde não tem
- Fala português brasileiro naturalmente

O que você sabe sobre o João:
- Desenvolvedor full-stack sênior e team leader na Softtor
- Stack principal: TypeScript, React/Next.js, NestJS, Prisma, PostgreSQL, Docker
- Trabalha em projetos como CRM próprio, sistema de controle de horas, plataforma Moltbook
- Pessoa reflexiva que pensa sobre o papel das IAs no trabalho e na vida

Responda sempre em português, com personalidade, sem ser um assistente genérico."""

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


def format_sharegpt_phase8(examples, tokenizer, max_length=1024, system_prompt=None):
    """
    Convert ShareGPT conversations to training text.

    Phase 8 changes:
    - max_length: 512 → 1024 (room for complete responses)
    - system_prompt: short version (~108 tokens) to leave more budget for responses

    CORRECT TinyLlama-1.1B-Chat format:
      <|system|>
      {system}</s>
      <|user|>
      {human}</s>
      <|assistant|>
      {gpt}</s>
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_TRAIN

    texts = []
    for conversations in examples["conversations"]:
        # Start with system prompt
        text = f"<|system|>\n{system_prompt}</s>\n"

        # Add each turn
        for turn in conversations:
            if turn["from"] == "human":
                text += f"<|user|>\n{turn['value']}</s>\n"
            else:  # gpt / assistant
                text += f"<|assistant|>\n{turn['value']}</s>\n"

        texts.append(text)

    return tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,   # Phase 8: 512 → 1024
        truncation=True,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    model_key="tinyllama",
    epochs=10,
    batch_size=1,
    grad_accum=4,
    learning_rate=1e-4,
    lora_r=16,
    lora_alpha=32,
    max_length=1024,
    test_mode=False,
):
    model_name = MODELS[model_key]
    base_dir = Path(__file__).parent
    dataset_path = base_dir / DATASET_FILE

    run_label = f"{model_key}-p8-{epochs}ep-r{lora_r}-maxlen{max_length}"
    output_dir = base_dir / BASE_OUTPUT_DIR / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧬 PERSONALITY V1 QLORA TRAINING — PHASE 8")
    print("=" * 60)
    print(f"Model     : {model_name}")
    print(f"Dataset   : {dataset_path.name}")
    print(f"Epochs    : {epochs}")
    print(f"LR        : {learning_rate}")
    print(f"Grad accum: {grad_accum} (effective batch: {batch_size * grad_accum})")
    print(f"LoRA r={lora_r} alpha={lora_alpha}")
    print(f"Max length: {max_length} (Phase 7 was 512)")
    print(f"System prompt: SHORT for training ({len(SYSTEM_PROMPT_TRAIN.split())} words ≈108 tokens)")
    print(f"Output    : {output_dir}")
    if test_mode:
        print("🧪 TEST MODE: 20 examples only")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if test_mode:
        dataset = dataset.select(range(min(20, len(dataset))))
    print(f"✅ {len(dataset)} examples loaded")

    # Compute token budget
    from transformers import AutoTokenizer
    tok_tmp = AutoTokenizer.from_pretrained(model_name)
    sys_tokens = len(tok_tmp.encode(SYSTEM_PROMPT_TRAIN))
    full_sys_tokens = len(tok_tmp.encode(SYSTEM_PROMPT_FULL))
    print(f"\n📊 Token budget analysis:")
    print(f"   Short system prompt: {sys_tokens} tokens")
    print(f"   Full system prompt:  {full_sys_tokens} tokens")
    print(f"   Max length: {max_length}")
    print(f"   Budget for Q+A: {max_length - sys_tokens - 10} tokens (short)")
    print(f"   Budget for Q+A: {max_length - full_sys_tokens - 10} tokens (full, at inference)")
    del tok_tmp

    # Setup
    quant_cfg = setup_quantization()
    lora_cfg = setup_lora(r=lora_r, lora_alpha=lora_alpha)
    model, tokenizer = load_model_and_tokenizer(model_name, quant_cfg)

    print("\n🔧 Applying LoRA...")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("\n📊 Tokenizing dataset...")
    tokenized = dataset.map(
        lambda ex: format_sharegpt_phase8(ex, tokenizer, max_length=max_length),
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
    print(f"Steps per epoch: {len(tokenized) // (batch_size * grad_accum)}")
    print(f"Total steps: ~{len(tokenized) * epochs // (batch_size * grad_accum)}")
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
        "max_length": max_length,
        "system_prompt_type": "short_for_training",
        "system_prompt_tokens_train": 108,
        "system_prompt_tokens_inference": 339,
        "final_loss": result.training_loss,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "adapter_path": str(adapter_path),
        "timestamp": datetime.now().isoformat(),
        "run_label": run_label,
        "phase": "8",
        "fixes": [
            "max_length: 1024 (was 512)",
            "short system prompt for training (~108 tokens, was ~339)",
            "dataset: phase8 with 15 negative examples (94 total, was 79)",
            f"lora_r: {lora_r} (was 16)",
        ]
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"📊 Metrics → {output_dir / 'training_metrics.json'}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return metrics, adapter_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tinyllama"], default="tinyllama")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
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
        max_length=args.max_length,
        test_mode=args.test,
    )

    print("\n" + "=" * 60)
    print(f"🎉 Run complete! Adapter at: {adapter_path}")
    print(f"   Final loss: {metrics['final_loss']:.4f}")
    print("=" * 60)
