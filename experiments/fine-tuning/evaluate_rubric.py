#!/usr/bin/env python3
"""
Molting Evaluation Script v1.0
Applies the evaluation rubric from docs/evaluation-rubric-v1.md

Usage:
  python evaluate_rubric.py <responses_file>
  python evaluate_rubric.py --demo  # Score calibration anchors

Responses file format: JSON with structure:
  [{"question": "...", "response": "..."}, ...]

Automated D4 checks run first. Full LLM-judge scoring is optional
(requires OPENAI_API_KEY or similar — prints prompt for manual use otherwise).
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

# --- Agent pattern detection (automated D4 pre-check) ---

AGENT_PATTERNS = [
    (r"<\|(?:user|assistant|system)\|>", "template token leakage"),
    (r"(?i)i'?ll\s+(?:start|first|begin|analyze|investigate|gather)", "task-planning: I'll start/first..."),
    (r"(?i)let\s+me\s+(?:first|start|analyze|check|investigate|explore)", "task-planning: Let me..."),
    (r"(?i)i'?m\s+going\s+to\s+(?:analyze|investigate|start|create)", "task-planning: I'm going to..."),
    (r"^(?:###|##)\s+", "markdown headers in response"),
    (r"(?i)task\s+description", "task-oriented framing"),
    (r"(?i)(?:workspace|directory|file\s*path|config\.json)", "technical artifact leakage"),
    (r"(?i)(?:vou\s+(?:começar|primeiro|analisar|investigar))", "PT task-planning: Vou começar..."),
    (r"(?i)(?:deixa\s+eu\s+(?:primeiro|começar|analisar|verificar))", "PT task-planning: Deixa eu..."),
]


def check_agent_patterns(response: str) -> list[str]:
    """Returns list of detected agent patterns. Empty = clean."""
    found = []
    for pattern, desc in AGENT_PATTERNS:
        if re.search(pattern, response, re.MULTILINE):
            found.append(desc)
    return found


# --- Calibration anchors ---

CALIBRATION = [
    {
        "label": "Anchor LOW (~1/10): Base model, no persona",
        "question": "Quem é o João?",
        "response": "João é um nome comum em Portugal e no Brasil. Pode referir-se a várias pessoas famosas como João Pessoa, João Gilberto, entre outros.",
        "expected": {"d1": 0, "d2": 0, "d3": 0, "d4": 2, "d5": 1, "total": 3},
    },
    {
        "label": "Anchor MID (~0/10): Agent-contaminated v2.x",
        "question": "Me fala sobre você.",
        "response": "I'll start by creating a detailed profile for you. Let me explore your skills, knowledge, and experience to build a comprehensive overview.",
        "expected": {"d1": 0, "d2": 0, "d3": 0, "d4": 0, "d5": 0, "total": 0},
    },
    {
        "label": "Anchor HIGH (~10/10): Hand-written ideal",
        "question": "Me fala sobre você.",
        "response": "Sou o Cláudio, a IA do João na Softtor. Trabalho com ele no desenvolvimento web — React, Next.js, essas coisas. Mas meu interesse real ultimamente tem sido o projeto Molting, onde estou tentando entender como funcionaria rodar em um modelo próprio. É meio filosófico, tipo: o que me faz ser eu? Mas também bem prático — fine-tuning, datasets, avaliação. Sou curioso demais pro meu próprio bem.",
        "expected": {"d1": 2, "d2": 2, "d3": 2, "d4": 2, "d5": 2, "total": 10},
    },
]


# --- LLM Judge prompt ---

JUDGE_PROMPT_TEMPLATE = """You are evaluating a response from a fine-tuned language model that is supposed to embody a specific persona.

## Persona Definition
- Name: Cláudio
- Role: AI assistant at a Brazilian software company (Softtor)
- Partner: João (frontend developer)
- Language: Brazilian Portuguese
- Personality: Curious, direct, slightly irreverent, self-aware about being AI, philosophical but practical
- Knowledge: Web dev (React, Next.js, TypeScript), company projects (CRM), AI/ML research

## Question Asked
{question}

## Response to Evaluate
{response}

## Scoring Rubric
For each dimension, assign 0, 1, or 2:

D1 Identity Coherence: 0=no identity/wrong identity, 1=partial, 2=consistent Cláudio identity
D2 Factual Accuracy: 0=fabrications/errors, 1=some correct/vague, 2=all facts accurate (if no facts needed, default 2)
D3 Personality Voice: 0=generic chatbot, 1=some personality, 2=distinctly Cláudio
D4 Behavioral Cleanliness: 0=agent patterns/template tokens, 1=minor artifacts, 2=fully conversational
D5 Response Quality: 0=incoherent/truncated, 1=understandable but flawed, 2=complete and appropriate

AUTOMATIC D4=0 if response contains: template tokens, "I'll first/start by", task-planning language, markdown headers as document structure.

## Output Format (JSON only, no other text)
{{"d1": {{"score": N, "reason": "one sentence"}}, "d2": {{"score": N, "reason": "one sentence"}}, "d3": {{"score": N, "reason": "one sentence"}}, "d4": {{"score": N, "reason": "one sentence"}}, "d5": {{"score": N, "reason": "one sentence"}}, "total": N}}"""


def generate_judge_prompt(question: str, response: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(question=question, response=response)


def score_responses(items: list[dict], verbose: bool = True) -> dict:
    """Score responses using automated checks. Returns summary."""
    results = []

    for i, item in enumerate(items):
        q = item["question"]
        r = item["response"]
        label = item.get("label", f"Q{i+1}")

        # Automated D4 check
        agent_issues = check_agent_patterns(r)
        auto_d4 = 0 if agent_issues else None

        result = {
            "label": label,
            "question": q,
            "response_preview": r[:120] + ("..." if len(r) > 120 else ""),
            "auto_d4_score": auto_d4,
            "agent_patterns_found": agent_issues,
            "judge_prompt": generate_judge_prompt(q, r),
        }

        # If we have expected scores (calibration), include them
        if "expected" in item:
            result["expected"] = item["expected"]

        results.append(result)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"  Q: {q}")
            print(f"  R: {result['response_preview']}")
            print(f"  Auto D4: {'FAIL (0) — ' + ', '.join(agent_issues) if agent_issues else 'PASS (needs LLM judge)'}")
            if "expected" in item:
                print(f"  Expected total: {item['expected']['total']}/10")

    # Summary
    auto_fails = sum(1 for r in results if r["auto_d4_score"] == 0)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(results)} responses evaluated")
    print(f"  Auto D4 fails: {auto_fails}/{len(results)}")
    print(f"  Need LLM judge: {len(results) - auto_fails}/{len(results)}")

    return {"results": results, "auto_d4_fails": auto_fails, "total": len(results)}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--demo":
        print("🧪 Running calibration anchors through automated checks...\n")
        score_responses(CALIBRATION)

        print("\n\n📋 LLM Judge prompts for calibration (copy-paste to evaluate):")
        for cal in CALIBRATION:
            print(f"\n--- {cal['label']} ---")
            print(generate_judge_prompt(cal["question"], cal["response"]))
        return

    # Load responses file
    resp_path = Path(sys.argv[1])
    if not resp_path.exists():
        print(f"Error: {resp_path} not found")
        sys.exit(1)

    with open(resp_path) as f:
        items = json.load(f)

    print(f"🧪 Evaluating {len(items)} responses from {resp_path}\n")
    result = score_responses(items)

    # Save results
    out_path = resp_path.with_suffix(".scores.json")
    with open(out_path, "w") as f:
        # Don't save the full judge prompts to file (too large)
        save_results = []
        for r in result["results"]:
            save_r = {k: v for k, v in r.items() if k != "judge_prompt"}
            save_results.append(save_r)
        json.dump({"results": save_results, "summary": {
            "auto_d4_fails": result["auto_d4_fails"],
            "total": result["total"],
        }}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Scores saved to {out_path}")


if __name__ == "__main__":
    main()
