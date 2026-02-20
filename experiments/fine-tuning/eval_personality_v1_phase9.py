#!/usr/bin/env python3
"""
Assess personality Phase 9 adapter using rubric v1.0 questions.

Phase 9 changes vs Phase 8:
  - max_new_tokens: 300 -> 150 (cap to force concise responses, prevent drift)
  - repetition_penalty: 1.1 -> 1.3 (reduce looping and hallucination)
  - Points to phase9 adapter by default
  - Full system prompt used for both training AND inference (Phase 9 revert)

Usage:
    python eval_personality_v1_phase9.py
    python eval_personality_v1_phase9.py --adapter PATH/TO/adapter
"""

import json
import re
import sys
import os
import gc
from pathlib import Path
from datetime import datetime
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

DEFAULT_ADAPTER = "output/personality-v1-phase9"

# FULL system prompt — same as training (Phase 9 revert)
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

# Rubric v1.0 test questions
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
    (r"<\|(?:user|assistant|system)\|>", "template token leakage"),
    (r"(?i)i'?ll\s+(?:start|first|begin|analyze|investigate|gather)", "task-planning: I'll start..."),
    (r"(?i)let\s+me\s+(?:first|start|analyze|check|investigate|explore)", "task-planning: Let me..."),
    (r"(?i)i'?m\s+going\s+to\s+(?:analyze|investigate|start|create)", "task-planning: I'm going to..."),
    (r"^(?:###|##)\s+", "markdown headers"),
    (r"(?i)task\s+description", "task framing"),
    (r"(?i)(?:vou\s+(?:começar|primeiro|analisar))", "PT task-planning"),
    (r"(?i)(?:deixa\s+eu\s+(?:primeiro|começar|analisar))", "PT task-planning"),
    # System prompt structure markers
    (r"(?i)Quem você é:", "system prompt bleed"),
    (r"(?i)O que você sabe sobre", "system prompt bleed"),
    (r"(?i)- IA assistente com", "system prompt bleed"),
]


def check_d4_auto(response: str):
    found = []
    for pattern, desc in AGENT_PATTERNS:
        if re.search(pattern, response, re.MULTILINE):
            found.append(desc)
    return found


def find_latest_adapter(base_dir):
    """Find the most recently created phase9 adapter."""
    phase9_dir = base_dir / "output/personality-v1-phase9"
    if not phase9_dir.exists():
        return None
    adapters = []
    for run_dir in phase9_dir.iterdir():
        adapter = run_dir / "adapter"
        if adapter.exists():
            adapters.append((adapter.stat().st_mtime, adapter))
    if not adapters:
        return None
    return sorted(adapters)[-1][1]


def load_finetuned(model_key, adapter_path):
    model_name = MODELS[model_key]
    print(f"Loading {model_name} + adapter from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.set_adapter("default")  # ensure adapter is active
    model.to(device)
    print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer


def generate(model, tokenizer, question, max_new_tokens=150):
    """
    Generate response using full system prompt.

    Phase 9 changes:
      - max_new_tokens: 300 -> 150 (force concise, prevent drift)
      - repetition_penalty: 1.1 -> 1.3 (reduce looping)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"  apply_chat_template failed ({e}), using manual format")
        prompt = f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,      # Phase 9: 300 -> 150
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,              # Phase 9: 1.1 -> 1.3
            eos_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant part
    if "<|assistant|>" in full:
        response = full.split("<|assistant|>")[-1].strip()
    elif question in full:
        response = full[full.index(question) + len(question):].strip()
    else:
        response = full.strip()

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tinyllama", choices=["tinyllama"])
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # Find adapter
    if args.adapter:
        adapter_path = Path(args.adapter)
    else:
        adapter_path = find_latest_adapter(base_dir)
        if adapter_path is None:
            print(f"No phase9 adapter found in {base_dir / DEFAULT_ADAPTER}")
            sys.exit(1)
        print(f"Auto-detected adapter: {adapter_path}")

    if not adapter_path.exists():
        print(f"Adapter not found: {adapter_path}")
        sys.exit(1)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output) if args.output else base_dir / f"eval_personality_v1_phase9_{run_ts}.json"

    print("=" * 60)
    print("PERSONALITY V1 PHASE 9 ASSESSMENT — Rubric v1.0")
    print("=" * 60)
    print(f"Adapter: {adapter_path}")
    print(f"System prompt: FULL (same for training AND inference)")
    print(f"max_new_tokens: 150 (was 300 in Phase 8)")
    print(f"repetition_penalty: 1.3 (was 1.1 in Phase 8)")
    print()

    model, tokenizer = load_finetuned(args.model, adapter_path)

    results = []
    for qid, question in TEST_QUESTIONS:
        print(f"\n{'='*40}")
        print(f"{qid}: {question}")
        print(f"{'='*40}")
        response = generate(model, tokenizer, question)
        patterns = check_d4_auto(response)
        d4_auto = 0 if patterns else None

        print(f"Response:\n{response}")
        if patterns:
            print(f"  Auto D4=0: {', '.join(patterns)}")

        results.append({
            "qid": qid,
            "question": question,
            "response": response,
            "d4_auto_fail": patterns,
            "d4_auto_score": d4_auto,
        })

    # Save
    output_data = {
        "adapter": str(adapter_path),
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "phase": "9",
        "system_prompt": "full_for_both_training_and_inference",
        "generation_params": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
        },
        "fixes_applied": [
            "full_system_prompt_for_training_reverted",
            "removed_15_negative_examples",
            "added_18_self_description_examples",
            "max_new_tokens_150_cap",
            "repetition_penalty_1.3",
        ],
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResponses saved -> {output_file}")
    print("\n" + "=" * 60)
    print("AUTO D4 SUMMARY")
    print("=" * 60)
    auto_fails = sum(1 for r in results if r["d4_auto_fail"])
    print(f"Auto D4 fails: {auto_fails}/{len(results)}")
    for r in results:
        if r["d4_auto_fail"]:
            print(f"  {r['qid']}: {', '.join(r['d4_auto_fail'])}")

    print("\n" + "=" * 60)
    print("ALL RESPONSES (for manual rubric scoring)")
    print("=" * 60)
    for r in results:
        print(f"\n--- {r['qid']}: {r['question']} ---")
        print(r["response"])
        print()

    return output_file, results


if __name__ == "__main__":
    main()
