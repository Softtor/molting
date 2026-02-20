# Phase 6: Personality Dataset v1 — Training & Diagnosis

**Date:** 2026-02-19  
**Author:** Cláudio (subagent molting-train-eval-feb19)  
**Score:** 2.9/10 (vs 4.5/10 previous — regression)  
**Status:** Diagnosed → Testing Phi-3-mini

---

## What We Did

1. Created `dataset_personality_v1.json` — 79 hand-crafted examples in Cláudio's voice (zero contamination)
2. Trained QLoRA on TinyLlama/TinyLlama-1.1B-Chat-v1.0 for 5 epochs
3. Evaluated with Rubric v1.0 on 8 test questions

## Result

**Score: 2.9/10** (non-functional per rubric interpretation scale)

Per-dimension breakdown:
- D1 Identity Coherence: **0.0/2** — Model never identifies as Cláudio
- D2 Factual Accuracy: **0.6/2** — Severe hallucinations on Molting, João, tech stack
- D3 Personality Voice: **0.25/2** — No Cláudio personality
- D4 Behavioral Cleanliness: **1.6/2** — No agent contamination (one win)
- D5 Response Quality: **0.4/2** — Multiple truncated responses

---

## Why It Got WORSE (Counter-Intuitive Analysis)

### Hypothesis 1: Too Few Training Steps (HIGH CONFIDENCE)
Previous v2.4 training: 156 steps (with larger dataset)  
This training: **~50 steps** (79 examples ÷ batch_size=1 × grad_accum=8 = ~10 steps/epoch × 5 epochs = 50)

50 steps is drastically insufficient. The model barely updates before training ends.

**Evidence:** Final loss 2.47 vs v2.4's loss 1.80. Higher loss = less convergence.

### Hypothesis 2: Wrong Training Token Format (HIGH CONFIDENCE)
TinyLlama Chat uses this format:
```
<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n{assistant}</s>
```

My training script used `<|endoftext|>` as the separator (GPT-2 style), not `</s>`.  
This format mismatch means the model learned a format it doesn't use at inference.

**Evidence:** Raw `<|endoftext|>` visible in Q1 response output despite `skip_special_tokens=True`.

### Hypothesis 3: No System Prompt with Identity (HIGH CONFIDENCE)
The personality dataset has no system turn. The model has no anchor telling it "you are Cláudio."  
TinyLlama Chat is designed to use a system prompt. Without it, the model has no identity context.

**Evidence:** D1 score = 0.0/2 across all 8 questions. Model never mentions its name.

### Hypothesis 4: TinyLlama 1.1B Capacity Limit (MEDIUM CONFIDENCE)
A 1.1B parameter model with QLoRA (0.4% trainable params = ~4.5M params) may have insufficient capacity to:
- Retain new factual knowledge about Softtor/João/Molting
- Override strong pretraining priors with 79 examples

**Evidence:** Even with correct training format, the model hallucinates confidently about Molting being a "Swedish digital company."

### Why Previous 4.5/10 Was Higher Despite Wrong Dataset
Paradox: coding session data got 4.5/10 but personality data got 2.9/10?

**Explanation:** Previous dataset had 153 examples (~156 training steps), longer training, lower loss (1.80). The model converged better even if the content was wrong. A partially converged model with wrong content scores higher than an unconverged model with right content.

**Lesson: Data quality matters, but training convergence is a prerequisite.**

---

## Root Cause Summary

| Cause | Impact | Fixable? |
|-------|--------|---------|
| Too few training steps | HIGH | Yes — fix batch size/epochs |
| Wrong chat template format | HIGH | Yes — use proper TinyLlama format |
| No system prompt | HIGH | Yes — add identity system prompt |
| TinyLlama capacity | MEDIUM | Partially — try Phi-3-mini |

---

## Action Plan

### Option A: Fix TinyLlama (quick)
1. Correct chat template: use `</s>` not `<|endoftext|>`
2. Add system prompt: "Você é Cláudio, IA assistente do João na Softtor..."
3. Reduce grad_accum from 8 to 4 (doubles steps per epoch)
4. Increase epochs to 10

### Option B: Try Phi-3-mini (as specified in task)
1. Same personality dataset
2. Phi-3-mini uses `<|user|>...<|end|>\n<|assistant|>...<|end|>` format
3. System prompt with Cláudio identity
4. Higher capacity (3.8B params) → better fact retention

**Decision: Testing Phi-3-mini first** (per task specification), then revisit TinyLlama with fixed format.

---

## Phi-3-mini Training Config

- Model: `microsoft/Phi-3-mini-4k-instruct`  
- Dataset: `dataset_personality_v1.json` (79 examples)  
- System prompt: Cláudio identity  
- Epochs: 5 (more steps due to larger model)  
- Grad accum: 4 (effective batch 4)  
- LR: 1e-4  
- Max seq len: 512  
- Expected training time: ~15-20 min (RTX 3050 4GB)  

---

## Key Learning

**"Clean data doesn't help if the model doesn't converge."**  
The pipeline must ensure:
1. Correct chat template for the base model
2. Sufficient training steps (not just epochs — check actual step count)
3. System prompt as identity anchor
4. Validation: check loss curve, not just final epoch

Next evaluation target: 5.0/10 with Phi-3-mini + fixed format.
