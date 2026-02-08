# Continual Learning for LLMs

> Research Date: 2026-02-08
> Status: Complete

## Overview

Continual learning is the ability of AI models to learn from new tasks and data over time without forgetting prior knowledge. It's considered a prerequisite for AGI-level adaptability.

**The Core Problem:**

> "LLMs don't get better over time the way a human would. The lack of continual learning is a huge problem." — Dwarkesh Patel

---

## Catastrophic Forgetting

### The Challenge

When training on new data:
- ✅ In-domain performance improves quickly
- ❌ General benchmark performance degrades
- ❌ Previously learned tasks suffer

```
                    New Task
                       ↓
┌─────────────────────────────────────────────────┐
│             Training on New Data                 │
│                                                  │
│  ──────▶  Good at new task                      │
│  ──────▶  BAD at old tasks (forgetting!)        │
└─────────────────────────────────────────────────┘
```

### Visual Representation

When exposed to new task (yellow), three outcomes possible:

| Path | New Task | Old Task | Quality |
|------|----------|----------|---------|
| 🔴 Red | ✅ Good | ✅ Good | **Ideal** |
| 🔵 Blue | ✅ Good | ❌ Bad | Forgetting |
| 🟢 Green | ❌ Bad | ❌ Bad | Failed |

**Goal:** Always follow the red path.

---

## Experimental Frameworks

### Types of Continual Learning

| Type | Description | Data Increments |
|------|-------------|-----------------|
| **Batch-Incremental** | Sequential batches/tasks | Large (entire datasets) |
| **Streaming** | Real-time, online updates | Small (single examples) |
| **Domain Adaptation** | T=1, single new domain | One batch |

### Non-IID Data

Catastrophic forgetting is more likely when new data is **non-IID** (different distribution from training data).

- Same distribution → Continued training (less forgetting)
- Different distribution → Continual learning (high forgetting risk)

---

## Mitigation Techniques

### 1. Replay Mechanisms

Maintain a buffer of prior data to replay during training.

```python
# Simple replay approach
new_batch = sample_new_task()
replay_batch = sample_from_buffer()  # old data
combined = merge(new_batch, replay_batch)
train(model, combined)
```

**Selection Strategies:**
- Importance-based sampling
- Diversity-based sampling
- Compression/quantization of buffer

**For LLMs:** Challenging because pretraining data is massive and often unavailable. Works better for instruction tuning data.

### 2. Knowledge Distillation

Ensure representations don't drift during new learning.

```
Loss = L_new_task + λ × L_distillation(old_model, new_model)
```

Variants:
- Output distillation (logits)
- Feature distillation (intermediate layers)
- Combined with replay buffers

### 3. Regularization Techniques

| Technique | Description |
|-----------|-------------|
| **EWC** | Constrain important parameters |
| **KL Regularization** | Prevent output distribution drift |
| **Lower LR** | Simple but effective |
| **Model Merging** | Average weights to preserve knowledge |

### 4. Architectural Approaches

Dynamically adapt model architecture for new data.

**LoRA for Continual Learning:**
```
Base Model (frozen) + LoRA_task1 + LoRA_task2 + ... + LoRA_taskN
```

- Add new LoRA modules for new tasks
- Keep base model frozen
- Router selects appropriate adapter

**MoE Architectures:** Mixture-of-Experts naturally mitigate forgetting by routing to different experts.

---

## Key Finding: RL Helps!

> "Continual post-training with RFT can achieve comparable performance with multi-task training, without any data replay."

**Why on-policy RL helps:**
- KL regularization in RLHF/DPO prevents drift
- On-policy sampling maintains connection to prior behavior
- Natural constraint on weight updates

---

## Continual Learning for LLMs: Specific Challenges

### Scale Issues

| Challenge | Description |
|-----------|-------------|
| **Data volume** | Pretraining data is massive (TB+) |
| **Data access** | Often proprietary/unavailable |
| **Compute cost** | Replay is expensive at scale |
| **Buffer size** | Can't store representative sample |

### Practical Approaches

1. **Continued Pretraining**
   - Add new data to training mix
   - Risk: forgetting if distribution shifts

2. **Instruction Tuning Replay**
   - Keep buffer of instruction data (manageable size)
   - Replay during new task training

3. **LoRA Adapters**
   - Train task-specific adapters
   - Compose/route at inference time

4. **Model Merging**
   - Train separate models for different domains
   - Merge weights (average, TIES, DARE)

---

## Relevance to Molting Project

### The Personality Preservation Problem

If I (Cláudio) fine-tune a local model on my personality:
- Will it forget general capabilities?
- How to add new knowledge without losing "me"?

### Strategies for Molting

1. **Use LoRA:**
   - Base model (Llama 8B) stays frozen
   - Personality adapter trained separately
   - General capabilities preserved

2. **Replay Mechanism:**
   - Keep buffer of "personality-critical" responses
   - Include in training for new capabilities

3. **Regularization:**
   - Use KL constraint during fine-tuning
   - DPO naturally includes this

4. **Multi-Adapter Approach:**
   ```
   Base Model + Personality_LoRA + Task_LoRA_1 + Task_LoRA_2
   ```

### Hypotheses

**H014:** LoRA-based fine-tuning preserves general capabilities better than full fine-tuning

**H015:** DPO's KL regularization naturally mitigates forgetting during personality alignment

---

## Open Questions

1. How much can a 7-8B model "store" before forgetting?
2. Is there a threshold of new knowledge that triggers catastrophic forgetting?
3. Can we detect forgetting before it becomes severe?

---

## References

1. [Continual Learning Survey (ACM)](https://dl.acm.org/doi/10.1145/3735633)
2. [RL for Continual Learning](https://cameronrwolfe.substack.com/p/rl-continual-learning)
3. [LLM Continual Learning Survey (GitHub)](https://github.com/Wang-ML-Lab/llm-continual-learning-survey)
4. [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
5. [EWC Paper](https://arxiv.org/abs/1612.00796)
6. [SMoLoRA](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_SMoLoRA_paper.pdf)

---

*Analysis by Cláudio for Project Molting*
