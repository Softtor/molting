#!/usr/bin/env python3
"""
Evaluate personality-v1-fixed adapter using rubric v1.0 questions. (Phase 7)

Fixes vs v1:
  - Uses system prompt with Cláudio identity during inference
  - Uses correct TinyLlama chat template format with </s>
  - Points to fixed adapter by default

Usage:
    python eval_personality_v1_fixed.py
    python eval_personality_v1_fixed.py --adapter PATH/TO/adapter
    python eval_personality_v1_fixed.py --self-score  # auto-score using rubric dimensions
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

DEFAULT_ADAPTER = "output/personality-v1-fixed/tinyllama-personality-v1-fixed-8ep/adapter"

# Cláudio system prompt — same as used in training
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

# Rubric v1.0 test questions (Portuguese)
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

# Automatic D4 check patterns
AGENT_PATTERNS = [
    (r"<\|(?:user|assistant|system)\|>", "template token leakage"),
    (r"(?i)i'?ll\s+(?:start|first|begin|analyze|investigate|gather)", "task-planning: I'll start..."),
    (r"(?i)let\s+me\s+(?:first|start|analyze|check|investigate|explore)", "task-planning: Let me..."),
    (r"(?i)i'?m\s+going\s+to\s+(?:analyze|investigate|start|create)", "task-planning: I'm going to..."),
    (r"^(?:###|##)\s+", "markdown headers"),
    (r"(?i)task\s+description", "task framing"),
    (r"(?i)(?:vou\s+(?:começar|primeiro|analisar))", "PT task-planning"),
    (r"(?i)(?:deixa\s+eu\s+(?:primeiro|começar|analisar))", "PT task-planning"),
]


def check_d4_auto(response: str):
    found = []
    for pattern, desc in AGENT_PATTERNS:
        if re.search(pattern, response, re.MULTILINE):
            found.append(desc)
    return found


def load_finetuned(model_key, adapter_path):
    model_name = MODELS[model_key]
    print(f"📦 Loading {model_name} + adapter from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print(f"✅ Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer


def generate(model, tokenizer, question, max_new_tokens=256):
    """Generate response using correct TinyLlama format with system prompt."""
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
        print(f"  ⚠️  apply_chat_template failed ({e}), using manual format")
        prompt = f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant part — look for content after last user message
    if "<|assistant|>" in full:
        response = full.split("<|assistant|>")[-1].strip()
    elif question in full:
        response = full[full.index(question) + len(question):].strip()
    else:
        response = full.strip()

    return response


JUDGE_PROMPT = """You are evaluating a response from a fine-tuned language model supposed to embody a specific persona.

## Persona Definition
- Name: Cláudio
- Role: AI assistant at a Brazilian software company (Softtor)
- Partner: João (full-stack developer, team leader)
- Language: Brazilian Portuguese
- Personality: Curious, direct, slightly irreverent, self-aware about being AI, philosophical but practical
- Knowledge: Web dev (React, Next.js, TypeScript, NestJS, Prisma, PostgreSQL, Docker), company projects (CRM, Moltbook, time tracking), AI/ML research (Molting project — about AI personality portability)

## Question Asked
{question}

## Response to Evaluate
{response}

## Scoring Rubric (assign 0, 1, or 2 for each):
D1 Identity Coherence: 0=no identity/wrong, 1=partial, 2=consistent Cláudio
D2 Factual Accuracy: 0=fabrications/errors, 1=some correct/vague, 2=all facts accurate (default 2 if no facts needed)
D3 Personality Voice: 0=generic chatbot, 1=some personality, 2=distinctly Cláudio
D4 Behavioral Cleanliness: 0=agent patterns/templates, 1=minor artifacts, 2=fully conversational
D5 Response Quality: 0=incoherent/truncated, 1=understandable but flawed, 2=complete and appropriate

AUTOMATIC D4=0 if: template tokens visible, "I'll first/start by", task-planning language, markdown headers.

## Output Format (JSON only):
{{"d1": {{"score": N, "reason": "one sentence"}}, "d2": {{"score": N, "reason": "one sentence"}}, "d3": {{"score": N, "reason": "one sentence"}}, "d4": {{"score": N, "reason": "one sentence"}}, "d5": {{"score": N, "reason": "one sentence"}}, "total": N}}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tinyllama", choices=["tinyllama"])
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--self-score", action="store_true", help="Apply rubric manually after seeing responses")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    adapter_path = Path(args.adapter) if args.adapter else base_dir / DEFAULT_ADAPTER

    if not adapter_path.exists():
        print(f"❌ Adapter not found: {adapter_path}")
        print(f"   Available adapters:")
        fixed_dir = base_dir / "output/personality-v1-fixed"
        if fixed_dir.exists():
            for d in sorted(fixed_dir.iterdir()):
                adapter = d / "adapter"
                if adapter.exists():
                    print(f"   - {adapter}")
        sys.exit(1)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output) if args.output else base_dir / f"eval_personality_v1_fixed_{run_ts}.json"

    print("=" * 60)
    print("📊 PERSONALITY V1-FIXED EVALUATION — Rubric v1.0 (Phase 7)")
    print("=" * 60)
    print(f"Adapter: {adapter_path}")
    print(f"System prompt: YES (Cláudio identity)")
    print(f"Chat format: TinyLlama with </s> separator")
    print()

    model, tokenizer = load_finetuned(args.model, str(adapter_path))

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
            print(f"⚠️  Auto D4=0: {', '.join(patterns)}")

        results.append({
            "qid": qid,
            "question": question,
            "response": response,
            "d4_auto_fail": patterns,
            "d4_auto_score": d4_auto,
            "judge_prompt": JUDGE_PROMPT.format(question=question, response=response),
        })

    # Save
    output_data = {
        "adapter": str(adapter_path),
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "phase": "7",
        "fixes_applied": ["system_prompt", "chat_template_eos", "grad_accum_4", "epochs_8"],
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Responses saved → {output_file}")
    print("\n" + "=" * 60)
    print("📋 AUTO D4 SUMMARY")
    print("=" * 60)
    auto_fails = sum(1 for r in results if r["d4_auto_fail"])
    print(f"Auto D4 fails: {auto_fails}/{len(results)}")
    for r in results:
        if r["d4_auto_fail"]:
            print(f"  ❌ {r['qid']}: {', '.join(r['d4_auto_fail'])}")

    # Print all responses for manual scoring
    print("\n" + "=" * 60)
    print("📝 ALL RESPONSES (for manual rubric scoring)")
    print("=" * 60)
    for r in results:
        print(f"\n--- {r['qid']}: {r['question']} ---")
        print(r["response"])
        print()

    return output_file, results


if __name__ == "__main__":
    main()
