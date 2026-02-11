#!/usr/bin/env python3
"""
QLoRA Fine-tuning on Phi-3-mini for Personality Transfer
Hardware: RTX 3050 (6GB VRAM), 31GB RAM
"""

import os
import argparse
import json
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
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gc


def setup_quantization_config():
    """Configure 4-bit quantization for QLoRA"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # normalized float 4-bit
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # nested quantization for extra memory savings
    )


def setup_lora_config(r=16, lora_alpha=32):
    """Configure LoRA adapter"""
    return LoraConfig(
        r=r,  # LoRA rank
        lora_alpha=lora_alpha,  # scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


def load_model_and_tokenizer(model_name="microsoft/Phi-3-mini-4k-instruct", quantization_config=None):
    """Load Phi-3-mini with 4-bit quantization"""
    print(f"📦 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'  # Important for training
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # automatically distribute across available GPUs
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation='eager'  # Fix for flash attention compatibility
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print(f"✅ Model loaded. Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return model, tokenizer


def format_conversation_sharegpt(examples, tokenizer):
    """
    Format ShareGPT conversations into training format
    Converts: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
    Into: "<|user|>\n...<|end|>\n<|assistant|>\n...<|end|>"
    """
    texts = []
    for conversations in examples['conversations']:
        text = ""
        for turn in conversations:
            role = "user" if turn['from'] == 'human' else 'assistant'
            text += f"<|{role}|>\n{turn['value']}<|end|>\n"
        texts.append(text)
    
    # Tokenize
    model_inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=512,  # Phi-3-mini context window is 4k, but we use 512 for memory efficiency
        return_tensors="pt"
    )
    
    # Labels are the same as input_ids for causal LM
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs


def train_qlora(
    dataset_path="dataset_sharegpt_filtered.json",
    output_dir="output",
    model_name="microsoft/Phi-3-mini-4k-instruct",
    epochs=3,
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=32,
    test_mode=False,
    max_samples=None
):
    """Main training function"""
    
    print("=" * 60)
    print("🚀 QLoRA FINE-TUNING EXPERIMENT")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA r={lora_r}, alpha={lora_alpha}")
    if test_mode:
        print(f"🧪 TEST MODE: Training on {max_samples or 50} samples only")
    print("=" * 60)
    
    # Setup paths
    base_dir = Path(__file__).parent
    dataset_path = base_dir / dataset_path
    output_dir = base_dir / output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = load_dataset('json', data_files=str(dataset_path), split='train')
    
    if test_mode:
        dataset = dataset.select(range(min(max_samples or 50, len(dataset))))
    
    print(f"✅ Loaded {len(dataset)} examples")
    
    # Setup quantization and LoRA
    quantization_config = setup_quantization_config()
    lora_config = setup_lora_config(r=lora_r, lora_alpha=lora_alpha)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, quantization_config)
    
    # Apply LoRA
    print("\n🔧 Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocess dataset
    print("\n📊 Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: format_conversation_sharegpt(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,  # mixed precision training
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",  # memory-efficient optimizer
        gradient_checkpointing=True,  # save VRAM at cost of speed
        report_to="none",  # disable wandb/tensorboard
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Train
    print("\n🏋️  Starting training...")
    print(f"VRAM before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    train_result = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"VRAM peak usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Final loss: {train_result.training_loss:.4f}")
    
    # Save adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"💾 LoRA adapter saved to {adapter_path}")
    
    # Save training metrics
    metrics = {
        "model": model_name,
        "dataset": str(dataset_path.name),
        "num_examples": len(dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "final_loss": train_result.training_loss,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📊 Metrics saved to {output_dir / 'training_metrics.json'}")
    
    # Cleanup
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3-mini with QLoRA")
    parser.add_argument('--dataset', default='dataset_sharegpt_filtered.json', help='Dataset file')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--model', default='microsoft/Phi-3-mini-4k-instruct', help='Base model')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Per-device batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--test', action='store_true', help='Test mode (50 samples only)')
    parser.add_argument('--test-samples', type=int, default=50, help='Number of samples for test mode')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires a GPU.")
        print(f"   Torch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        exit(1)
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Run training
    metrics = train_qlora(
        dataset_path=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        test_mode=args.test,
        max_samples=args.test_samples
    )
    
    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT COMPLETE!")
    print("=" * 60)
