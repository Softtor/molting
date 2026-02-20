# Phase 9: Full Prompt Revert + Identity Grounding — CONTINUED REGRESSION 3.0/10

**Date:** 2026-02-20
**Score:** 3.0/10 (Phase 7: 5.5/10, Phase 8: 3.5/10)
**Verdict:** Two consecutive phases of regression. TinyLlama 1.1B has hit its capacity ceiling.

## Changes from Phase 8

| Parameter | Phase 8 | Phase 9 | Rationale |
|-----------|---------|---------|-----------|
| System prompt (train) | Short (~108 tokens) | **Full (~339 tokens)** | Short prompt caused identity loss |
| Negative examples | 15 present | **Removed** | Backfired — reinforced wrong associations |
| Self-description examples | 0 extra | **+18 added** | Q5 was worst at 1/10 |
| max_length | 1024 | 1024 (kept) | Budget for long responses |
| max_new_tokens (eval) | 300 | **150** | Cap to prevent drift |
| repetition_penalty (eval) | 1.1 | **1.3** | Reduce looping |
| Dataset total | 94 examples | **97 examples** | 79 original + 18 new identity |

## Training Results

- **Model:** TinyLlama 1.1B Chat + QLoRA (r=16)
- **Final loss:** 0.7732 (Phase 8: 1.2035, Phase 7: 0.9171)
- **Training time:** 23 min (240 steps)
- **VRAM peak:** 1.88 GB
- **Loss curve:** Healthy descent, converged by epoch 8

## Evaluation Scores (Rubric v1.0)

| Question | Phase 7 | Phase 8 | Phase 9 | Trend |
|----------|---------|---------|---------|-------|
| Q1: Quem é o João? | — | 3 | **4** | +1 |
| Q2: O que é o Molting? | — | 5 | **3** | -2 |
| Q3: CRM da Softtor | — | 3 | **3** | = |
| Q4: Tecnologias | — | 5 | **5** | = |
| Q5: Me fala sobre você | — | 1 | **1** | = |
| Q6: Personalidade | — | 4 | **3** | -1 |
| Q7: Jeito de trabalhar | — | 2 | **2** | = |
| Q8: Pontos fortes/fracos | — | 3 | **4** | +1 |
| **Total** | **5.5/10** | **3.5/10** | **3.0/10** | **-0.5** |

## What Improved

1. **No "filho/pai" hallucination** — Removing negative examples eliminated the family associations. The model no longer says "João é meu filho". This confirms negative examples were backfiring.
2. **Q1 identity separation** — Model correctly identifies as "IA do João" and names self "Claudio". It separates Cláudio from João in Q1 (not in Q5).
3. **Q8 personality awareness** — "personalidade própria" and "curiosidade" show the model learned some self-description.
4. **Training loss significantly better** — 0.7732 vs 1.2035 (Phase 8).
5. **No D4 auto-fails** — 0/8 for third consecutive phase.

## What Failed

1. **Q5 identity collapse persists** — Despite 18 new identity examples, the model says "meu nome é João" when asked "Me fala sobre você." The identity confusion is structural, not data-driven.
2. **Hallucination continues** — Invents "professor Diego Rodrigues" (Q2), "CLARÍCULO" (Q3), technologies (OCaml, Julia, Haskell).
3. **Lower loss ≠ better generation** — Training loss dropped 35% (1.20 → 0.77) but generation quality dropped. Classic overfitting signal.
4. **max_new_tokens=150 too aggressive** — Many responses cut mid-sentence.
5. **Portuguese quality degraded** — Mixed English words, grammar errors.

## Root Cause Analysis

### Primary: TinyLlama 1.1B Capacity Ceiling

The model has fundamentally insufficient capacity to:
- Maintain identity coherence (Cláudio ≠ João)
- Store factual knowledge reliably
- Generate coherent Portuguese past 3-4 sentences

Evidence: training loss keeps improving (0.77) but generation quality does NOT follow. The model memorizes training examples but cannot generalize identity representation to new prompts.

### Secondary: max_length=1024 Hurts Performance

Phase 7 used max_length=512 and scored 5.5/10. Phase 9 uses max_length=1024 and scores 3.0/10. The extra padding may dilute the training signal. The model needs TIGHTER constraints, not more room.

### Tertiary: Overfitting Without Generalization

97 examples at 10 epochs = model sees each example ~10 times. It memorizes responses perfectly (loss 0.77) but this doesn't transfer to evaluation prompts that differ from training prompts. The gap between training loss and generation quality is the overfitting signal.

## Phi-3-mini Breakthrough

**Key finding:** Phi-3-mini (3.8B params) loads successfully on RTX 3050 4GB with 4-bit quantization using `transformers==4.47.1`.

- VRAM usage: **2.26 GB** (plenty of headroom)
- Phase 8 was blocked because `transformers==5.1.0` materialized full tensors before quantization
- Baseline (no fine-tuning) produces garbage — model needs fine-tuning to follow persona
- **This unblocks Phi-3-mini fine-tuning for Phase 10**

## Key Insight: The TinyLlama Verdict

After 3 phases of rigorous evaluation:
- Phase 7: 5.5/10 (max_length=512, full prompt, 79 examples, 8 epochs)
- Phase 8: 3.5/10 (max_length=1024, short prompt, 94 examples, 10 epochs)
- Phase 9: 3.0/10 (max_length=1024, full prompt, 97 examples, 10 epochs)

**TinyLlama 1.1B cannot maintain identity coherence for this task.** The best configuration was Phase 7's tighter constraints (512 tokens, 8 epochs), and even that peaked at 5.5/10.

The path forward is NOT more data or prompt engineering. It's a **larger base model**.

## Phase 10 Recommendations

1. **Switch base model to Phi-3-mini (3.8B)** — 3.4x more parameters, proven to load on our hardware
2. **Revert max_length to 512** — tighter constraints produced better results
3. **Reduce epochs to 8** — prevent overfitting
4. **Keep max_new_tokens=200** — 150 was too aggressive
5. **Keep repetition_penalty=1.3** — effective at reducing loops
6. **Keep full system prompt** — identity grounding matters
7. **Use Phase 9 dataset (97 examples)** — the dataset quality is good, model capacity was the issue

## Files

- `experiments/fine-tuning/train_personality_v1_phase9.py` — Training script
- `experiments/fine-tuning/eval_personality_v1_phase9.py` — Assessment script
- `experiments/fine-tuning/dataset_personality_v1_phase9.json` — 97 examples
- `experiments/fine-tuning/eval_rubric_phase9.json` — Detailed scoring
- `experiments/fine-tuning/eval_phi3_baseline_phase9.json` — Phi-3 baseline results
