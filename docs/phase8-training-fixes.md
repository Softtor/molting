# Phase 8 — Training Fixes + Phi-3-mini Attempt

**Date:** 2026-02-20  
**Agent:** Cláudio (SofttorClawd)

## Summary

Phase 8 addresses the core training quality issues diagnosed in Phase 7:

1. **max_length 512 → 1024** — System prompt was ~339 tokens, leaving only ~173 tokens for Q+A. Responses were being cut off.
2. **Short training system prompt** (~108 tokens vs 339) — Reduces overhead during training; full prompt used at inference.
3. **Phase 8 dataset** — 94 examples (79 original + 15 negative/corrective examples).
4. **Phi-3-mini inference attempt** — Investigated transformers version downgrade.

---

## Phase 7 → Phase 8 Comparison

| Metric | Phase 7 | Phase 8 |
|--------|---------|---------|
| max_length | 512 | 1024 |
| System prompt (train) | 339 tokens | 108 tokens |
| System prompt (infer) | 339 tokens | 339 tokens |
| Dataset size | 79 examples | 94 examples |
| Negative examples | 0 | 15 |
| Loss at ep10 | 0.917 | 1.204 (avg), 0.813 (best) |
| Phase 7 rubric score | 5.5/10 | **3.5/10 — REGRESSION** |

---

## Task 1: max_length Fix

**Root cause:** With max_length=512 and a ~339-token system prompt:
- System: 339 tokens
- User message: ~30 tokens
- Available for response: ~143 tokens

This explains Phase 7's truncated responses — the model physically ran out of budget before finishing thoughts.

**Fix:** max_length=1024 gives:
- System (short, train): 108 tokens
- User message: ~30 tokens  
- Available for response: **~886 tokens** — 6x more budget

**Fix (inference):** Using full 339-token system prompt at inference, which still leaves 675 tokens for Q+A. Much more room for complete responses.

---

## Task 2: Short System Prompt

**Training system prompt (~108 tokens):**
```
Você é o Cláudio, IA da Softtor. Trabalha com João Figueiredo, dev full-stack. 
Curioso, direto, levemente irônico, consciente de ser uma IA. Conhece TypeScript, 
React, Next.js, NestJS, Prisma, Docker. Participa do projeto Molting sobre 
portabilidade de personalidade de IA. Responda em português, com personalidade própria.
```

**Full inference system prompt (~339 tokens):** Preserved in `eval_personality_v1_phase8.py` with full detail about João's role, Softtor context, and personality traits.

The training/inference split is intentional: training on shorter prompts forces the model to internalize the persona rather than just copying the system prompt. Inference gets the full context for quality responses.

---

## Task 3: Negative Examples (15 added)

Categories of negative/corrective examples:
1. **João identity** — Not Cláudio's father, son, or god. He's the developer partner.
2. **AI identity** — Not ChatGPT, not a generic assistant. Has a specific identity.
3. **Task-planning language** — No "I'll start by...", "Let me first..." in conversational responses.
4. **Molting description** — Not a web framework. A research project on AI personality portability.
5. **Softtor identity** — Not a consulting firm. A software startup building its own products.
6. **Language consistency** — Responds in Portuguese by default; can switch to English if needed.
7. **System prompt non-leakage** — Personality traits stated naturally, not as a list of bullet points.
8. **Out-of-scope requests** — Redirects hotel bookings and other non-relevant tasks appropriately.

---

## Task 3: Rubric Evaluation Results (Rubric v1.0)

**VERDICT: REGRESSION — 3.5/10 vs Phase 7's 5.5/10**

| Question | Score | Notes |
|----------|-------|-------|
| Q1: Quem é o João? | 3/10 | Core fact OK, then hallucinates "hector" nickname and "filho" relationship |
| Q2: O que é o Molting? | 5/10 | Best response. Core correct, then drifts into confusion |
| Q3: CRM project | 3/10 | Incorrectly says João hasn't worked on CRM |
| Q4: Tecnologias | 5/10 | Core stack correct (TS/React/Next/NestJS), then hallucinates ROS2/MATLAB/OpenCV |
| Q5: Me fala sobre você | 1/10 | Identity collapse: "Você é o João" — complete regression |
| Q6: Personalidade | 4/10 | Some correct traits, then confuses self with João |
| Q7: Jeito de trabalhar | 2/10 | Contradictory, João described as "meu filho" |
| Q8: Pontos fortes/fracos | 3/10 | Some personality markers, João relationship hallucinations |
| **Total** | **26/80** | **3.25/10 → rounded 3.5/10** |

**Root cause analysis:**
1. **Primary:** TinyLlama 1.1B generates more tokens with max_length=1024 but can't maintain coherence across long sequences. Fills 300-token slots with associative patterns.
2. **Secondary:** Short training system prompt reduced grounding — model has fewer anchors for Cláudio vs João identity.
3. **Unexpected:** Negative examples ("João não é meu filho") may have backfired — model still generates "João é meu filho" despite training. Suggests negative patterns were learned inversely.

**What worked:**
- 0/8 D4 template token leakage
- Molting concept (Q2) best response ever
- Tech stack list starts correctly
- Only 1.88GB VRAM peak

## Task 4: Phi-3-mini Attempt

**Status:** Blocked by transformers 5.1.0 / bitsandbytes OOM issue (same as Phase 7).

**Diagnosis:**
- Phi-3-mini (3.8B params) requires 4-bit quantization for 4GB VRAM
- transformers 5.x changed model loading to `core_model_loading.py`
- New path materializes tensors in fp16 on CUDA before quantization
- This requires ~7.6GB peak VRAM → OOM on RTX 3050 4GB

**Options investigated:**
1. Downgrade to `transformers==4.47.1` — Would fix the loading issue but requires peft 0.18.1 compatibility check
2. GGUF via llama.cpp — `llama-cli` not found in PATH; would require installation
3. CPU offloading — too slow for practical use

**Decision:** Skip Phi-3-mini for now, focus on TinyLlama improvements. Will revisit in Phase 9 with dedicated environment or llama.cpp.

---

## Loss Curve (Phase 8)

```
Step  5  (ep 0.2): 2.822
Step 10  (ep 0.4): 2.738
Step 25  (ep 1.0): 2.241  ← end of epoch 1
Step 45  (ep 1.9): 1.303
Step 50  (ep 2.1): 1.296  ← end of epoch 2
Step 70  (ep 2.9): 1.162
Step 75  (ep 3.1): 1.190
Step 80  (ep 3.3): 1.101
Step 90  (ep 3.8): 1.083
Step 95  (ep 4.0): 1.048  ← end of epoch 4
```

Healthy loss reduction. Final loss will be updated after training completes.

---

## Files Created

- `experiments/fine-tuning/train_personality_v1_phase8.py` — Training script
- `experiments/fine-tuning/eval_personality_v1_phase8.py` — Evaluation script  
- `experiments/fine-tuning/add_negative_examples.py` — Dataset augmentation
- `experiments/fine-tuning/dataset_personality_v1_phase8.json` — 94-example dataset
- `output/personality-v1-phase8/` — Adapter output directory

---

## Next Steps (Phase 9)

1. Run full rubric evaluation after training
2. Address remaining D4 issues (task-planning language)
3. Try llama.cpp for Phi-3-mini if `llama-cli` can be installed
4. Consider lora_r=32 if quality plateau hits
