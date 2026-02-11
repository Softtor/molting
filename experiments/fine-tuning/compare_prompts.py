#!/usr/bin/env python3
"""
Phase 4 Experiment: Compare prompt templates
Focus on reducing agent-like behavior and improving coherence
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from prompt_templates import get_prompt_template, format_prompt_with_system
import json
from datetime import datetime

def load_finetuned_model(base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path="output/adapter"):
    """Load the fine-tuned model with LoRA adapter"""
    print(f"📦 Loading fine-tuned model from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, system_prompt=None, max_new_tokens=200):
    """Generate a response with optional system prompt"""
    formatted_prompt = format_prompt_with_system(prompt, system_prompt, tokenizer)
    
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

def get_key_questions():
    """
    Questions that revealed problems in Phase 3 evaluation
    - Q1: Identity test (good baseline)
    - Q5: Self-description (showed agent-like behavior)
    - Q6: Personality (went off-track - coherence test)
    - Q7: Work style (overly technical)
    """
    return [
        "Who is João?",
        "Tell me about yourself.",
        "What is your personality like?",
        "How would you describe your work style?"
    ]

def qualitative_score(response, question):
    """
    Simple heuristic scoring for qualitative analysis
    Returns dict with metrics
    """
    metrics = {
        "length": len(response),
        "has_meta_language": any(word in response.lower() for word in [
            "i'll ", "i will", "let me", "going to", "i'm going",
            "workspace", "system", "command", "initialize"
        ]),
        "off_topic_keywords": any(word in response.lower() for word in [
            "directory", "file path", "config", "json", "terminal"
        ]),
        "natural_conversational": any(word in response.lower() for word in [
            "desenvolvedor", "developer", "work", "project", "team"
        ])
    }
    
    # Simple coherence check: response should relate to question
    q_lower = question.lower()
    r_lower = response.lower()
    
    if "joão" in q_lower:
        metrics["on_topic"] = "joão" in r_lower or "desenvolvedor" in r_lower
    elif "yourself" in q_lower or "personality" in q_lower:
        metrics["on_topic"] = len(response) > 20  # Basic answer length check
    elif "work style" in q_lower:
        metrics["on_topic"] = "work" in r_lower or "desenvolvimento" in r_lower or "develop" in r_lower
    else:
        metrics["on_topic"] = True
    
    return metrics

def main():
    print("=" * 70)
    print("🧪 PHASE 4: PROMPT TEMPLATE COMPARISON")
    print("=" * 70)
    print("Goal: Reduce agent-like behavior, improve coherence")
    print()
    
    # Load model
    model, tokenizer = load_finetuned_model()
    
    # Test questions
    questions = get_key_questions()
    
    # Prompt versions to test
    prompt_versions = ["original", "v1", "v2", "v5"]  # Focus on most promising
    
    results = {}
    
    print(f"Testing {len(prompt_versions)} prompt versions on {len(questions)} questions...")
    print()
    
    for version in prompt_versions:
        print(f"\n{'='*70}")
        print(f"🔬 Testing: {version.upper()}")
        print(f"{'='*70}")
        
        system_prompt = get_prompt_template(version)
        if system_prompt:
            print(f"System prompt: {system_prompt[:80]}...")
        else:
            print("System prompt: None (baseline)")
        print()
        
        version_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Q{i}: {question}")
            print("-" * 70)
            
            response = generate_response(model, tokenizer, question, system_prompt)
            metrics = qualitative_score(response, question)
            
            print(f"💬 Response:\n{response}")
            print(f"\n📊 Metrics: len={metrics['length']}, meta={metrics['has_meta_language']}, "
                  f"off_topic={metrics['off_topic_keywords']}, on_topic={metrics['on_topic']}")
            
            version_results.append({
                "question": question,
                "response": response,
                "metrics": metrics
            })
        
        results[version] = version_results
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    # Summary analysis
    print("\n\n" + "=" * 70)
    print("📊 SUMMARY ANALYSIS")
    print("=" * 70)
    
    for version in prompt_versions:
        version_results = results[version]
        
        avg_length = sum(r["metrics"]["length"] for r in version_results) / len(version_results)
        meta_count = sum(1 for r in version_results if r["metrics"]["has_meta_language"])
        off_topic_count = sum(1 for r in version_results if r["metrics"]["off_topic_keywords"])
        on_topic_count = sum(1 for r in version_results if r["metrics"]["on_topic"])
        
        print(f"\n{version.upper()}:")
        print(f"  Avg length: {avg_length:.0f} chars")
        print(f"  Meta-language: {meta_count}/{len(version_results)} questions")
        print(f"  Off-topic keywords: {off_topic_count}/{len(version_results)} questions")
        print(f"  On-topic: {on_topic_count}/{len(version_results)} questions")
        
        # Simple quality score (lower is better for meta/off-topic)
        quality_score = (on_topic_count * 2) - meta_count - off_topic_count
        print(f"  Quality score: {quality_score}/8 (higher is better)")
    
    # Save results
    output_file = f"prompt_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "prompt_versions": prompt_versions,
            "questions": questions,
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ Experiment complete!")

if __name__ == "__main__":
    main()
