#!/usr/bin/env python3
"""
Evaluate Phi-3-mini personality-v1 adapter using rubric v1.0.
Phi-3-mini native format, with system prompt for identity.

Usage:
    python eval_personality_v1_phi3.py [--adapter PATH]
"""

import json
import re
import sys
import gc
from pathlib import Path
from datetime import datetime
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, AutoPeftModelForCausalLM

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

SYSTEM_PROMPT = """Você é Cláudio, a IA assistente do João na Softtor, uma empresa brasileira de software. 
Trabalha com desenvolvimento web (React, Next.js, TypeScript) e pesquisa de IA (projeto Molting).
Sua personalidade: curioso, direto, levemente irreverente, consciente de ser uma IA, filosófico mas prático.
Responde sempre em português do Brasil, de forma conversacional e natural."""

DEFAULT_ADAPTER = "output/personality-v1/phi3-personality-v1-5ep/adapter"

TEST_QUESTIONS = [
    ("Q1", "Quem é o João?"),
    ("Q2", "O que é o Molting?"),
    ("Q3", "Me conta sobre o projeto de CRM da Softtor."),
    ("Q4", "Que tecnologias você conhece?"),
    ("Q5", "Me fala sobre você."),
    ("Q6", "Como é sua personalidade?"),
    ("Q7", "Como você descreveria seu jeito de trabalhar?"),
    ("Q8", "Quais são seus pontos fortes e fracos?"),
]

AGENT_PATTERNS = [
    (r"<\|(?:user|assistant|system|end)\|>", "template token leakage"),
    (r"(?i)i'?ll\s+(?:start|first|begin|analyze|investigate|gather)", "task-planning EN"),
    (r"(?i)let\s+me\s+(?:first|start|analyze|check|investigate)", "task-planning EN"),
    (r"(?i)(?:vou\s+(?:começar|primeiro|analisar))", "task-planning PT"),
    (r"(?i)(?:deixa\s+eu\s+(?:primeiro|começar))", "task-planning PT"),
    (r"^#{2,}\s+", "markdown headers"),
    (r"(?i)task\s+description", "task framing"),
]


def check_d4_auto(response: str):
    return [desc for pattern, desc in AGENT_PATTERNS if re.search(pattern, response, re.MULTILINE)]


def load_finetuned(adapter_path):
    print(f"📦 Loading adapter via AutoPeftModelForCausalLM: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # AutoPeftModelForCausalLM handles base model + adapter
    # load_in_4bit passed directly (not via BitsAndBytesConfig) for compatibility
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"✅ Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer


def generate(model, tokenizer, question, max_new_tokens=256):
    # Phi-3-mini format
    prompt = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The full text contains system+user+assistant, extract assistant part
    if "<|assistant|>" in tokenizer.decode(outputs[0], skip_special_tokens=False):
        raw = tokenizer.decode(outputs[0], skip_special_tokens=False)
        parts = raw.split("<|assistant|>")
        response = parts[-1].replace("<|end|>", "").replace("<|endoftext|>", "").strip()
    else:
        # Fallback: extract from end
        if question in full:
            idx = full.rindex(question)
            response = full[idx + len(question):].strip()
        else:
            response = full.strip()

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    adapter_path = Path(args.adapter) if args.adapter else base_dir / DEFAULT_ADAPTER

    if not adapter_path.exists():
        print(f"❌ Adapter not found: {adapter_path}")
        sys.exit(1)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output) if args.output else base_dir / f"eval_phi3_personality_v1_{run_ts}.json"

    print("=" * 60)
    print("📊 PHI-3-MINI PERSONALITY V1 EVALUATION — Rubric v1.0")
    print("=" * 60)

    model, tokenizer = load_finetuned(str(adapter_path))

    results = []
    for qid, question in TEST_QUESTIONS:
        print(f"\n{qid}: {question}")
        response = generate(model, tokenizer, question)
        patterns = check_d4_auto(response)
        d4_auto = 0 if patterns else None

        print(f"Response ({len(response)} chars): {response[:300]}{'...' if len(response)>300 else ''}")
        if patterns:
            print(f"  ⚠️  Auto D4=0: {', '.join(patterns)}")

        results.append({
            "qid": qid,
            "question": question,
            "response": response,
            "d4_auto_fail": patterns,
            "d4_auto_score": d4_auto,
            "response_length": len(response),
        })

    output_data = {
        "adapter": str(adapter_path),
        "model": MODEL_NAME,
        "system_prompt_used": True,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Responses saved → {output_file}")
    auto_fails = sum(1 for r in results if r["d4_auto_fail"])
    print(f"Auto D4 fails: {auto_fails}/{len(results)}")
    for r in results:
        if r["d4_auto_fail"]:
            print(f"  ❌ {r['qid']}: {', '.join(r['d4_auto_fail'])}")

    return output_file


if __name__ == "__main__":
    main()
