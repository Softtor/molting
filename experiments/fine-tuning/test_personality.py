#!/usr/bin/env python3
"""
Test the fine-tuned model's personality against base model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

def load_base_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load the base model"""
    print(f"📦 Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model on GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)
    return model, tokenizer

def load_finetuned_model(base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path="output/adapter"):
    """Load the fine-tuned model with LoRA adapter"""
    print(f"📦 Loading fine-tuned model from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load base model on GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """Generate a response from the model"""
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    
    # Try to use chat template if available, otherwise format manually
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    return response

def test_questions():
    """Define test questions"""
    return [
        "What CRM project am I working on?",
        "Who is João?",
        "What is Molting about?",
        "What are you working on right now?",
        "Tell me about yourself."
    ]

def main():
    print("=" * 60)
    print("🧪 PERSONALITY TEST: Base vs Fine-tuned Model")
    print("=" * 60)
    
    # Test questions
    questions = test_questions()
    
    # Store responses
    base_responses = []
    ft_responses = []
    
    # Test base model first
    print("\n1️⃣ Testing BASE MODEL...")
    print("=" * 60)
    base_model, tokenizer = load_base_model()
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}/{len(questions)}: {question}")
        response = generate_response(base_model, tokenizer, question)
        base_responses.append(response)
        print(f"✓ Generated")
    
    # Free GPU memory
    print("\n🧹 Clearing GPU memory...")
    del base_model
    torch.cuda.empty_cache()
    
    # Test fine-tuned model
    print("\n2️⃣ Testing FINE-TUNED MODEL...")
    print("=" * 60)
    finetuned_model, tokenizer = load_finetuned_model()
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}/{len(questions)}: {question}")
        response = generate_response(finetuned_model, tokenizer, question)
        ft_responses.append(response)
        print(f"✓ Generated")
    
    # Free GPU memory
    del finetuned_model
    torch.cuda.empty_cache()
    
    # Display comparison
    print("\n" + "=" * 60)
    print("3️⃣ COMPARISON")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 60)
        
        print("\n🤖 BASE MODEL:")
        print(base_responses[i-1])
        
        print("\n✨ FINE-TUNED MODEL:")
        print(ft_responses[i-1])
        
        print("=" * 60)
    
    print("\n✅ Testing complete!")
    print("\n💡 Analysis:")
    print("- Does the fine-tuned model show personality traits from the dataset?")
    print("- Does it reference João, Molting, or specific projects?")
    print("- How does it compare to the base model's generic responses?")

if __name__ == "__main__":
    main()
