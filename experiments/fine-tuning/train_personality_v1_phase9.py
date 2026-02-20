#!/usr/bin/env python3
"""
QLoRA Fine-tuning: Personality Dataset v1 — Phase 9

Phase 9 fixes vs Phase 8:
  1. REVERTED to FULL system prompt for training (short prompt caused identity loss)
  2. Removed 15 negative examples (they backfired — model learned the wrong patterns)
  3. Added 18 new self-description examples (Q5 type was worst at 1/10)
  4. Kept max_length=1024 from Phase 8
  5. Total: 97 examples (79 original + 18 new identity-grounding)

Root cause of Phase 8 regression (5.5→3.5/10):
  - Short system prompt reduced identity grounding
  - Negative examples reinforced wrong associations (João = filho)
  - Model generated longer incoherent sequences with max_length=1024

Dataset: dataset_personality_v1_phase9.json
Base model: TinyLlama 1.1B Chat (default)

Usage:
    python train_personality_v1_phase9.py
    python train_personality_v1_phase9.py --epochs 10 --lora-r 32
    python train_personality_v1_phase9.py --test  # quick smoke test
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

DATASET_FILE = "dataset_personality_v1_phase9.json"
BASE_OUTPUT_DIR = "output/personality-v1-phase9"

# FULL system prompt for BOTH training AND inference (Phase 9 revert)
# Phase 8 used a short ~108 token prompt for training which caused identity loss.
# Phase 7 used the full prompt and scored 5.5/10. Reverting to full.
SYSTEM_PROMPT = """Você é o Cláudio, a IA da Softtor. Você trabalha com João Figueiredo, desenvolvedor full-stack e líder de equipe na Softtor, uma startup brasileira de tecnologia.

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
    print(f"Loading: {model_name}")
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
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer


def format_sharegpt_phase9(examples, tokenizer, max_length=1024, system_prompt=None):
    """
    Convert ShareGPT conversations to training text.

    Phase 9: Uses FULL system prompt (reverted from Phase 8's short prompt).
    max_length=1024 kept from Phase 8.

    CORRECT TinyLlama-1.1B-Chat format:
      <|system|>
      {system}</s>
      <|user|>
      {human}</s>
      <|assistant|>
      {gpt}</s>
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

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
        max_length=max_length,
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

    run_label = f"{model_key}-p9-{epochs}ep-r{lora_r}-maxlen{max_length}"
    output_dir = base_dir / BASE_OUTPUT_DIR / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PERSONALITY V1 QLORA TRAINING — PHASE 9")
    print("=" * 60)
    print(f"Model     : {model_name}")
    print(f"Dataset   : {dataset_path.name}")
    print(f"Epochs    : {epochs}")
    print(f"LR        : {learning_rate}")
    print(f"Grad accum: {grad_accum} (effective batch: {batch_size * grad_accum})")
    print(f"LoRA r={lora_r} alpha={lora_alpha}")
    print(f"Max length: {max_length}")
    print(f"System prompt: FULL for training (Phase 9 revert from Phase 8 short)")
    print(f"Output    : {output_dir}")
    if test_mode:
        print("TEST MODE: 20 examples only")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if test_mode:
        dataset = dataset.select(range(min(20, len(dataset))))
    print(f"{len(dataset)} examples loaded")

    # Compute token budget
    from transformers import AutoTokenizer
    tok_tmp = AutoTokenizer.from_pretrained(model_name)
    sys_tokens = len(tok_tmp.encode(SYSTEM_PROMPT))
    print(f"\nToken budget analysis:")
    print(f"   System prompt: {sys_tokens} tokens (FULL — same for train & inference)")
    print(f"   Max length: {max_length}")
    print(f"   Budget for Q+A: {max_length - sys_tokens - 10} tokens")
    del tok_tmp

    # Setup
    quant_cfg = setup_quantization()
    lora_cfg = setup_lora(r=lora_r, lora_alpha=lora_alpha)
    model, tokenizer = load_model_and_tokenizer(model_name, quant_cfg)

    print("\nApplying LoRA...")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("\nTokenizing dataset...")
    tokenized = dataset.map(
        lambda ex: format_sharegpt_phase9(ex, tokenizer, max_length=max_length),
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

    print("\nTraining...")
    print(f"Steps per epoch: {len(tokenized) // (batch_size * grad_accum)}")
    print(f"Total steps: ~{len(tokenized) * epochs // (batch_size * grad_accum)}")
    print(f"VRAM before: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    result = trainer.train()

    print(f"\nDone! Loss: {result.training_loss:.4f}")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Save adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved -> {adapter_path}")

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
        "system_prompt_type": "full_for_both_training_and_inference",
        "system_prompt_tokens": sys_tokens,
        "final_loss": result.training_loss,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "adapter_path": str(adapter_path),
        "timestamp": datetime.now().isoformat(),
        "run_label": run_label,
        "phase": "9",
        "fixes": [
            "REVERTED to full system prompt for training (Phase 8 short caused identity loss)",
            "Removed 15 negative examples (backfired — reinforced wrong patterns)",
            "Added 18 self-description examples (Q5 type was worst at 1/10)",
            "max_length: 1024 (kept from Phase 8)",
            f"lora_r: {lora_r}",
        ]
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics -> {output_dir / 'training_metrics.json'}")

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
        print("CUDA not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
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
    print(f"Run complete! Adapter at: {adapter_path}")
    print(f"   Final loss: {metrics['final_loss']:.4f}")
    print("=" * 60)
