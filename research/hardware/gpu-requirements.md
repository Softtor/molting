# GPU Requirements for LLM Training and Inference

> Research Date: 2026-02-08
> Status: Complete

## Overview

Understanding hardware requirements is crucial for planning the Molting project. This document covers VRAM needs for inference vs training, and GPU recommendations.

---

## Quick Rules of Thumb

### Inference

```
~2GB VRAM per 1B parameters (FP16)
```

| Model Size | FP16 | INT8 | INT4 |
|------------|------|------|------|
| 7B | 14GB | 7GB | 3.5GB |
| 8B | 16GB | 8GB | 4GB |
| 13B | 26GB | 13GB | 6.5GB |
| 32B | 64GB | 32GB | 16GB |
| 70B | 140GB | 70GB | 35GB |

### Fine-Tuning

```
~16GB VRAM per 1B parameters (full fine-tune, FP16)
```

Much higher than inference because of:
- Optimizer states (AdamW = 3 copies)
- Gradients
- Activations

---

## VRAM Breakdown for 7B Model

### 1. Model Parameters
| Precision | VRAM |
|-----------|------|
| FP32 | 28GB |
| FP16 | 14GB |
| Mixed | 21GB |

### 2. Optimizer States (AdamW)
| Method | VRAM |
|--------|------|
| Standard | 84GB (3 × 4 bytes/param) |
| 8-bit optimizers | 42GB |

### 3. Gradients
| Precision | VRAM |
|-----------|------|
| FP32 | 28GB |
| FP16 | 14GB |

### 4. Activations
Variable based on:
- Batch size
- Sequence length
- Architecture

Can be reduced with **gradient checkpointing**.

### Total for Full Fine-Tune (7B)
```
FP16 + 8-bit optimizer: 14 + 42 + 14 ≈ 70GB VRAM
```

---

## VRAM Requirements Table

| Method | Precision | 7B | 13B | 30B | 70B | 110B |
|--------|-----------|-----|-----|-----|-----|------|
| **Full Fine-Tune** | FP16 | 67GB | 125GB | 288GB | 672GB | 1056GB |
| **LoRA** | FP16 | 15GB | 28GB | 63GB | 146GB | 229GB |
| **QLoRA** | INT8 | 9GB | 17GB | 38GB | 88GB | 138GB |
| **QLoRA** | **INT4** | **5GB** | **9GB** | **20GB** | **46GB** | 72GB |

**Key Insight:** QLoRA INT4 makes fine-tuning accessible on consumer GPUs!

---

## GPU Recommendations

### For Inference

| Model Size | Minimum GPU | Recommended GPU |
|------------|-------------|-----------------|
| 7-8B (Q4) | 8GB (RTX 3060) | 12GB (RTX 4070) |
| 13B (Q4) | 12GB | 16GB (RTX 4080) |
| 32B (Q4) | 16GB | 24GB (RTX 4090) |
| 70B (Q4) | 48GB (A6000) | 80GB (A100/H100) |

### For Fine-Tuning (QLoRA)

| Model Size | Minimum GPU | Recommended GPU |
|------------|-------------|-----------------|
| 7-8B | 8GB | 16GB |
| 13B | 12GB | 24GB (RTX 4090) |
| 32B | 24GB | 48GB (A6000) |
| 70B | 48GB | 80GB (A100) |

### Consumer GPUs (2025-2026)

| GPU | VRAM | Good For |
|-----|------|----------|
| RTX 3060 | 12GB | 7B inference, small fine-tune |
| RTX 4060 Ti | 16GB | 7-8B QLoRA |
| RTX 4070 Ti Super | 16GB | 7-8B QLoRA, fast inference |
| **RTX 4090** | **24GB** | **13B QLoRA, 7-8B comfortable** |
| RTX 5090 | 32GB | 13B+ QLoRA |

### Datacenter GPUs

| GPU | VRAM | Cost/hr (Cloud) |
|-----|------|-----------------|
| A10G | 24GB | ~$1/hr |
| A100 40GB | 40GB | ~$2/hr |
| A100 80GB | 80GB | ~$3/hr |
| H100 | 80GB | ~$4/hr |
| H200 | 141GB | ~$6/hr |

---

## CPU Inference (GGUF)

For systems without GPU, GGUF format enables CPU inference:

| Model Size | RAM Required | Speed |
|------------|--------------|-------|
| 7B Q4 | 8GB | ~10 tok/s |
| 7B Q8 | 12GB | ~8 tok/s |
| 13B Q4 | 12GB | ~5 tok/s |

**João's System:** 31GB RAM → Can run 13B Q4 comfortably on CPU.

---

## Cloud vs Local Cost Analysis

### One-Time Fine-Tune (7B, 3 epochs, ~2 hours)

| Option | Cost |
|--------|------|
| RTX 4090 (own) | $0 (electricity ~$0.50) |
| Cloud A10G | ~$2 |
| Cloud A100 | ~$6 |

### Monthly Inference (8 hours/day)

| Option | Monthly Cost |
|--------|--------------|
| Local CPU | ~$5 (electricity) |
| Local RTX 4090 | ~$20 (electricity) |
| Cloud A10G | ~$240/month |

**Verdict:** Local hardware pays for itself quickly for continuous use.

---

## Relevance to Molting Project

### João's Actual Hardware (Verified 2026-02-08)

| Component | Spec | Assessment |
|-----------|------|------------|
| **CPU** | Intel i7-12650H (12th Gen) | ✅ Good for CPU inference |
| **RAM** | 31GB | ✅ Excellent for large GGUF models |
| **GPU** | RTX 3050 Mobile (4GB VRAM) | ⚠️ Limited for LLMs |
| **Ollama** | v0.15.2, gpt-oss:20b (13GB) | ✅ Already working |

### Capabilities

**With 4GB GPU VRAM:**
- 7B Q4 inference (borderline, ~3.5GB needed)
- Very small fine-tuning not practical

**With 31GB RAM (CPU):**
- 20B+ models via GGUF ✅ (already tested with gpt-oss:20b)
- 7B Q8 comfortably
- 13B Q4 easily

### Recommendations for Molting

**Phase 1: Testing (Current)**
- ✅ Use GGUF on CPU (already setup)
- Consider smaller model (7-8B) for faster iteration
- gpt-oss:20b is slow but works

**Phase 2: Fine-Tuning**
- Cloud spot instances recommended (A10G ~$0.50/hr)
- RTX 3050 4GB too limited for QLoRA
- Alternative: Google Colab free tier (T4 16GB)

**Phase 3: Production**
- CPU inference viable but slow
- Consider GPU upgrade for speed:
  - RTX 4060 Ti 16GB (~$450) — 7-8B QLoRA + fast inference
  - RTX 4070 Ti 12GB (~$550) — faster inference
  - RTX 4090 24GB (~$1600) — 13B QLoRA + production speed

### Tested Configuration

```
Model: gpt-oss:20b (13GB GGUF)
Hardware: CPU (i7-12650H) + 31GB RAM
Status: Works, but slow (~few tokens/sec expected)
```

---

## References

1. [Modal VRAM Guide](https://modal.com/blog/how-much-vram-need-fine-tuning)
2. [RunPod GPU Guide](https://www.runpod.io/blog/llm-fine-tuning-gpu-guide)
3. [VRAM Calculator](https://apxml.com/tools/vram-calculator)
4. [Hyperstack VRAM Requirements](https://www.hyperstack.cloud/blog/case-study/how-much-vram-do-you-need-for-llms)

---

*Analysis by Cláudio for Project Molting*
