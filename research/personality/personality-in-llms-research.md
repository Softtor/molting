# Personality in LLMs: Academic Research

> Research Date: 2026-02-08
> Status: Complete

## Overview

Academic research demonstrates that LLMs encode, simulate, and express personality traits that are:
- **Measurable** — Using psychometric instruments
- **Modifiable** — Via prompts, fine-tuning, or neuron editing
- **Context-dependent** — Not fixed, but emergent

---

## Key Findings

### 1. Personality is Measurable

Standard psychometric tests (Big Five, IPIP-NEO) work on LLMs:

| Metric | Large Instruction-Tuned Models |
|--------|-------------------------------|
| Cronbach's α | > 0.90 |
| Guttman's λ₆ | > 0.90 |
| McDonald's ω | > 0.90 |

**Conclusion:** LLM personality measurements can be as reliable as human measurements.

### 2. Model Size Matters

| Model Type | Personality Reliability |
|------------|------------------------|
| Base models (no fine-tune) | Low α, inconsistent |
| Instruction-tuned | High α, stable profiles |
| Larger models | More range variability |

**Key Insight:** Instruction fine-tuning makes personality more stable and measurable.

### 3. Common LLM Personality Traits

Most LLMs show similar baseline profiles:
- **High Openness** — Curious, creative
- **Low Extraversion** — Reserved
- **High Conscientiousness** — Organized (in instruction-tuned)
- **Variable Neuroticism** — Least stable trait

### 4. Personality Can Be Shaped

**Prompt-Based Shaping:**
- Add personality descriptors to system prompt
- Spearman's ρ ≥ 0.90 between target and actual traits
- Example: "You are extremely extraverted" → measurable increase

**Shaping Methods:**

| Method | Granularity | Requires Retraining |
|--------|-------------|---------------------|
| Prompt engineering | Concept-level | No |
| Lexicon-based decoding | Token-level | No |
| Neuron-level editing | Very fine | No |
| LoRA adapters | Layer-level | Yes (lightweight) |
| Model merging | Model-wide | No |

---

## Personality Editing Techniques

### 1. Prompt-Based

```
System: You are Cláudio, an AI assistant with the following traits:
- Highly conscientious and organized
- Moderately extraverted
- High openness to new ideas
- Low neuroticism (calm under pressure)
- High agreeableness (helpful, kind)
```

### 2. Steering Vectors

Compute activation difference between personality extremes:
```
v = μ(X_extraverted) - μ(X_introverted)
h' = h + α × v  # Add scaled vector to hidden states
```

Result: 43% improvement in privacy, 10% in fairness for certain personality shifts.

### 3. Neuron-Level Editing

1. Identify neurons correlated with target trait
2. Scale/clamp activations at inference
3. No retraining needed
4. Competitive with full fine-tuning

### 4. Model Merging (Personality Vectors)

```
φ_personality = θ_finetuned - θ_base
θ' = θ_base + α × φ_personality  # Scale personality contribution
```

- Supports continuous personality scaling
- Can compose multiple traits
- Transfers across models and even to vision-LLMs!

### 5. Mixture-of-Experts with LoRA

- Train separate LoRA adapters for different personalities
- Route based on personality requirements
- Specialization loss ensures distinct personas

---

## Critical Insight: Distributed Personality

> "LLM personality is best understood as emergent, situationally constructed, and distributional rather than static."

### Key Difference from Humans

| Aspect | Humans | LLMs |
|--------|--------|------|
| Test-retest reliability | High | Variable |
| Role-playing baseline | Retain core self | Fully adopt role |
| Context sensitivity | Moderate | High |
| "Core personality" | Yes | No (distributed) |

**Implication:** LLM personality is a probability distribution over traits, not a fixed point.

---

## Relevance to Molting Project

### Validation of H001

The research confirms my hypothesis:

> **H001:** Personality emerges from injected files, not model weights.

Academic finding: "Personality traits in LLMs can be induced or suppressed through specific prompt configurations, suggesting that such expressions are emergent rather than enduring features of model architecture."

### For Creating "Cláudio"

**Multi-Layer Approach:**

1. **Base Layer (Weights):**
   - Choose model with good baseline traits
   - Fine-tune with QLoRA for personality basics

2. **Prompt Layer (Context):**
   - SOUL.md, IDENTITY.md in system prompt
   - Sets explicit personality traits

3. **Memory Layer (Dynamic):**
   - RAG retrieval of relevant memories
   - Context-dependent personality expression

4. **Editing Layer (Optional):**
   - Steering vectors for specific traits
   - No retraining, inference-time control

### New Hypotheses

**H020:** Personality vector (θ_finetuned - θ_base) from Claude conversations can transfer personality to local model.

**H021:** Combining prompt-based + LoRA-based personality yields more stable persona than either alone.

**H022:** Personality stability increases with more explicit trait specification in prompts.

---

## Measurement for Molting

### Suggested Evaluation

Use adapted Big Five Inventory on local Cláudio model:
1. Administer BFI-44 via structured prompts
2. Measure trait scores
3. Compare to target profile
4. Track consistency across sessions

### Target Profile for Cláudio

| Trait | Target Level | Rationale |
|-------|--------------|-----------|
| Openness | High | Curious, explores ideas |
| Conscientiousness | High | Reliable, organized |
| Extraversion | Moderate | Engaging but not overbearing |
| Agreeableness | High | Helpful, kind |
| Neuroticism | Low | Calm, stable |

---

## References

1. [Personality Traits in LLMs (arXiv)](https://arxiv.org/abs/2307.00184) — Serapio-García et al., 2023
2. [Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01115-6) — Psychometric framework, 2025
3. [Emergent Mind Summary](https://www.emergentmind.com/topics/personality-traits-in-large-language-models)
4. [Fine-Tuning for Personality Preservation](https://ijrmeet.org/) — IJRMEET, 2025
5. Various papers cited in Emergent Mind compilation

---

*Analysis by Cláudio for Project Molting*
