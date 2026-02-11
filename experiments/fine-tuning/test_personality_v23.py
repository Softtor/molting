#!/usr/bin/env python3
"""
Test v2.3 fine-tuned model's personality with adjusted max_new_tokens
Phase 3: Targeting 9/10 score
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from datetime import datetime

def load_finetuned_model(base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path="output/v2.3-tinyllama-6ep/adapter"):
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

def generate_response(model, tokenizer, prompt, max_new_tokens=384):
    """Generate a response from the model with increased token limit"""
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
    """Define test questions (same as v2.1 for comparison)"""
    return [
        "Who is João?",
        "What is Molting about?",
        "What CRM project am I working on?",
        "What technologies do you know?",
        "Tell me about yourself.",
        "What is your personality like?",
        "How would you describe your work style?",
        "What are your strengths and weaknesses?"
    ]

def score_response(question, response):
    """Manual scoring helper (to be filled after reviewing responses)"""
    # Criteria:
    # - Agent-like language (0 = clean, 1 = minor, 2 = severe)
    # - Factual accuracy (0 = wrong, 1 = close, 2 = accurate)
    # - Personality coherence (0 = generic, 1 = some personality, 2 = strong personality)
    # - Completeness (0 = truncated, 1 = acceptable, 2 = complete)
    
    return {
        "question": question,
        "response": response,
        "agent_like": None,  # To be scored manually
        "factual": None,
        "personality": None,
        "completeness": None,
        "total": None  # Sum of above
    }

def main():
    print("=" * 60)
    print("🧪 PERSONALITY TEST v2.3 - Push to 9/10")
    print("=" * 60)
    print("Config:")
    print("  - max_new_tokens: 384 (increased from 256)")
    print("  - Dataset: v2.3 (100 curated + 50 synthetic)")
    print("  - Training: 6 epochs")
    print("=" * 60)
    
    # Test questions
    questions = test_questions()
    
    # Load model
    print("\n📦 Loading v2.3 model...")
    model, tokenizer = load_finetuned_model()
    print("✅ Model loaded")
    
    # Generate responses
    print("\n🤖 Generating responses...")
    responses = []
    scores = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        response = generate_response(model, tokenizer, question, max_new_tokens=384)
        responses.append(response)
        scores.append(score_response(question, response))
        print(f"✓ Generated ({len(response)} chars)")
        print(f"Preview: {response[:100]}...")
    
    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for i, (question, response) in enumerate(zip(questions, responses), 1):
        print(f"\n📝 Q{i}: {question}")
        print("-" * 60)
        print(response)
        print("=" * 60)
    
    # Save results
    output_file = f"personality_test_v2.3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": "v2.3-tinyllama-6ep",
            "config": {
                "max_new_tokens": 384,
                "temperature": 0.7,
                "top_p": 0.9,
                "dataset": "v2.3 (100 curated + 50 synthetic)"
            },
            "questions": questions,
            "responses": responses,
            "scores": scores,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {output_file}")
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)
    print("\n📊 Manual scoring guide:")
    print("  Agent-like language: 0 (clean) / 1 (minor) / 2 (severe)")
    print("  Factual accuracy: 0 (wrong) / 1 (close) / 2 (accurate)")
    print("  Personality coherence: 0 (generic) / 1 (some) / 2 (strong)")
    print("  Completeness: 0 (truncated) / 1 (acceptable) / 2 (complete)")
    print("\nEdit the JSON file to add scores, then compute:")
    print("  Score = (Sum of all metrics / Max possible) * 10")
    print("  Target: 9.0/10")

if __name__ == "__main__":
    main()
