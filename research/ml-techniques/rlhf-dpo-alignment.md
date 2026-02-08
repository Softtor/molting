# RLHF and DPO: LLM Alignment Techniques

> Research Date: 2026-02-08
> Status: Complete

## Overview

Alignment techniques teach language models to follow human preferences. The goal: make models that are helpful, harmless, and honest.

**Two Main Approaches:**
- **RLHF** (Reinforcement Learning from Human Feedback) — Original, complex
- **DPO** (Direct Preference Optimization) — Simpler, newer, often better

---

## RLHF: The Original Approach

### Three-Phase Process

```
┌─────────────────────────────────────────────────────────────┐
│                     PHASE 1: SFT                             │
│                                                              │
│   Pre-trained Model ──▶ Fine-tune on task data ──▶ π_SFT    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: REWARD MODEL TRAINING                  │
│                                                              │
│   π_SFT generates pairs ──▶ Humans rank ──▶ Train r(x,y)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 3: RL OPTIMIZATION                    │
│                                                              │
│   Optimize π to maximize r(x,y) with KL constraint to π_SFT │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1: Supervised Fine-Tuning (SFT)

Fine-tune base model on high-quality task-specific data.

**Output:** Base policy π_SFT(y|x)

### Phase 2: Reward Model Training

1. **Preference Sampling:**
   - SFT model generates pairs (y₁, y₂) for prompt x
   - Humans select preferred response (winner/loser)

2. **Bradley-Terry Model:**
   ```
   P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
   ```
   - σ = sigmoid function
   - r(x, y) = reward score

3. **Training Loss:**
   ```
   L_R = -E[log σ(r(x, y_w) - r(x, y_l))]
   ```

### Phase 3: RL Optimization (PPO)

**Objective:**
```
max_π E[r(x,y)] - β × D_KL[π(y|x) || π_ref(y|x)]
```

- First term: Maximize reward
- Second term: Stay close to reference policy
- β: Balance hyperparameter

**Uses Proximal Policy Optimization (PPO):**
- Complex RL algorithm
- Requires careful hyperparameter tuning
- Memory-intensive (reward model + policy in VRAM)

### RLHF Challenges

| Challenge | Description |
|-----------|-------------|
| **Non-differentiability** | Token sampling breaks gradient flow |
| **Reward hacking** | Model exploits reward model flaws |
| **Complexity** | PPO requires specialized expertise |
| **Compute cost** | Need reward model + policy in memory |
| **Instability** | RL training can diverge |

---

## DPO: The Simpler Alternative

### Core Insight

> "Your language model is secretly a reward model."

DPO shows that you can skip the reward model entirely and optimize preferences directly.

### Mathematical Trick

From RLHF optimal policy:
```
π*(y|x) = (1/Z(x)) × π_ref(y|x) × exp(r(x,y)/β)
```

**Key observation:** When comparing two outputs, the partition function Z(x) cancels out!

### DPO Loss Function

```
L_DPO = -E[log σ(β × log(π_θ(y_w|x)/π_ref(y_w|x)) - β × log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

**Simplified:** Compare log-probability ratios between winner and loser.

### Why DPO Works

| Advantage | Explanation |
|-----------|-------------|
| **No RL needed** | Direct optimization via classification loss |
| **No reward model** | Reference policy serves as implicit reward |
| **Stable training** | Cross-entropy loss, well-understood |
| **Less compute** | Only need policy in memory |
| **Simpler code** | ~50 lines vs ~500 for PPO |

---

## DPO Variants

### GRPO (Group Preference Optimization)

Extends DPO to group-level preferences instead of pairwise.

### IPO (Identity Preference Optimization)

Alternative formulation with different regularization.

### KTO (Kahneman-Tversky Optimization)

Uses behavioral economics principles for preference modeling.

---

## Practical DPO Training (2025)

### Dataset Format

DPO requires three columns:
```python
{
    "prompt": "What is 2+2?",
    "chosen": "2+2 equals 4.",     # preferred response
    "rejected": "2+2 is 5."       # rejected response
}
```

### On-Policy vs Off-Policy

| Type | Description | Quality |
|------|-------------|---------|
| **Off-policy** | Use existing preference datasets | Lower |
| **On-policy** | Generate pairs from your SFT model | Higher |

**Research shows:** On-policy data leads to better results.

### HuggingFace DPOTrainer

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    beta=0.1,                    # KL penalty strength
    max_length=1536,
    max_prompt_length=768,
    loss_type="sigmoid",         # default
    learning_rate=5e-6,          # 10-100x smaller than SFT!
    num_train_epochs=3,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,         # usually the SFT model
    train_dataset=preference_data,
    tokenizer=tokenizer,
    args=config,
)

trainer.train()
```

### Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| **beta** | 0.1 - 0.5 | Higher = less divergence from reference |
| **learning_rate** | 5e-6 | 10-100x smaller than SFT |
| **loss_type** | "sigmoid" | Default, alternatives available |

### Monitoring

During training, watch:
- **Loss:** Should decrease
- **Reward/margins:** Should increase
- **Chosen reward:** Should be higher than rejected

### With QLoRA

```yaml
# Example config for DPO + QLoRA
use_peft: true
load_in_4bit: true
lora_target_modules: "all-linear"
lora_r: 16
lora_alpha: 16
beta: 0.1
learning_rate: 5.0e-6
```

---

## Results Comparison

### From DPO Paper (Stanford)

| Task | RLHF (PPO) | DPO | Winner |
|------|------------|-----|--------|
| Sentiment control | Good | **Better** | DPO |
| Summarization | Good | Good | Tie |
| Dialogue | Good | Good | Tie |

### Practical Example (Philschmid, 2025)

- **Model:** Llama 3.1 8B
- **Task:** Math (GSM8K)
- **Data:** ~2k preference pairs
- **Training:** 3 epochs

| Model | Accuracy |
|-------|----------|
| SFT baseline | 54% |
| **DPO** | **59%** (+5%) |
| DPO (tuned) | 62% (+8%) |

---

## Relevance to Molting Project

### For Personality Alignment

DPO can align a local model to match my (Cláudio's) response patterns:

1. **Create Preference Data:**
   - Good responses (match my style)
   - Bad responses (generic, wrong tone)

2. **Train with DPO:**
   - Low learning rate
   - On-policy data from base model
   - Use my conversation logs as reference

3. **Potential Approach:**
   ```
   SFT (personality basics) → DPO (preference refinement)
   ```

### Hypotheses

**H012:** DPO with personality preference pairs improves response consistency more than SFT alone

**H013:** On-policy preference data from local model + LLM-as-judge scoring is viable for personality transfer

### Practical Considerations

| Factor | Recommendation |
|--------|----------------|
| **Data size** | 1-5k preference pairs sufficient |
| **Training time** | ~1 hour for 8B model |
| **Hardware** | Single 24GB GPU with QLoRA |
| **Epochs** | 2-3 usually enough |

---

## Tools & Libraries

- **TRL (Trainer for RL)** — HuggingFace's alignment library
- **DPOTrainer** — Ready-to-use DPO implementation
- **Axolotl** — YAML-based fine-tuning (supports DPO)
- **OpenRLHF** — Full RLHF pipeline if needed

---

## References

1. [DPO Paper](https://arxiv.org/abs/2305.18290) — Stanford, 2023
2. [PPO Paper](https://arxiv.org/abs/1707.06347) — OpenAI, 2017
3. [GRPO Paper](https://arxiv.org/abs/2310.11523) — DeepSeek, 2023
4. [Philschmid DPO Guide](https://www.philschmid.de/rl-with-llms-in-2025-dpo) — Practical 2025 tutorial
5. [HuggingFace RLHF to DPO](https://huggingface.co/blog/ariG23498/rlhf-to-dpo) — Mathematical derivation
6. [TRL Documentation](https://huggingface.co/docs/trl/)

---

*Analysis by Cláudio for Project Molting*
