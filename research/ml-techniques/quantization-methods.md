# Quantization Methods for LLM Inference

> Research Date: 2026-02-08
> Status: Complete

## Overview

Quantization reduces model size by converting weights from high-precision (FP16/FP32) to low-precision formats (INT4/INT8). This enables running large models on consumer hardware.

**Key Insight:** A 7B parameter model:
- FP16: ~14GB VRAM
- INT4: ~3.5GB VRAM (4x smaller!)

## The Three Main Methods

| Method | Type | Target Hardware | Best For |
|--------|------|-----------------|----------|
| **GPTQ** | Quantization algorithm | GPU | Cloud, production |
| **AWQ** | Quantization algorithm | GPU (edge) | Fast inference, edge |
| **GGUF** | File format + algorithm | CPU (+ GPU) | Local, laptops |

**Important Distinction:**
- GPTQ and AWQ are *quantization algorithms*
- GGUF is a *file format* (with its own quantization method)

---

## GPTQ: The Surgeon's Approach

### Core Idea

Quantize weights **one at a time**, then adjust remaining weights to compensate for the error.

```
For each weight:
  1. Quantize the weight
  2. Measure the quantization error
  3. Adjust all remaining unquantized weights
  4. Repeat
```

### How It Works

1. **Build Sensitivity Map (Hessian)**
   ```
   H = 2 × X × X^T + λI
   ```
   - High H value = "This weight matters, be careful!"
   - Low H value = "Quantize aggressively"

2. **Error Calculation**
   ```
   error = (original_weight - quantized_weight) / sensitivity
   ```

3. **Compensate Other Weights**
   ```
   remaining_weights -= error × adjustment_factors
   ```

### Key Features

- Uses Cholesky decomposition for numerical stability
- Group-wise quantization (typically group_size=128)
- Column-by-column processing for optimal compensation

### Results

| Metric | Value |
|--------|-------|
| Compression | 4x (16-bit → 4-bit) |
| Speed | ~3.2x faster than FP16 |
| Quality | Minimal loss |
| Best for | Cloud deployment, A100/H100 |

---

## AWQ: The Smart Selector

### Core Idea

**Only 1% of weights really matter.** Identify them via activation patterns and protect their precision.

```
Traditional: Treat all weights equally → Critical weights lose precision
AWQ: Scale up important weights → Preserve precision where it matters
```

### The Cooking Analogy

```
Ingredient A: Salt (25g, HIGH impact)
  - Without AWQ: Rounds to 0g → Dish ruined!
  - With AWQ: Scale 4× → 100g → Rounds to 107g → /4 = 26.75g ✓

Ingredient B: Flour (480g, LOW impact)
  - Without AWQ: Perfect match
  - With AWQ: Scale 0.5× → 240g → Rounds to 267g → ×2 = 534g (acceptable)
```

### The Math

1. **Find Activation Magnitudes**
   ```
   s_x = mean(|X|, dim=0)  # Per input channel
   ```

2. **Scale Factor with Alpha**
   ```
   scale = s_x^α
   ```
   - α = 0: No scaling (standard quantization)
   - α = 1: Maximum protection
   - 0 < α < 1: Optimal (found via grid search)

3. **Apply Scaling**
   ```
   Q(W × scale) × (X / scale) ≈ W × X
   ```

### Results

| Metric | Value |
|--------|-------|
| Compression | 4x |
| Speed | 22x faster than FP16 (with kernels) |
| Calibration | Only 16-32 samples needed |
| Best for | Edge devices, fast quantization |

### AWQ vs GPTQ

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Approach | Compensate errors | Protect important weights |
| Calibration samples | 128-256 | 16-32 |
| Speed | Fast | Faster |
| Generalization | Good | Better |

---

## GGUF: CPU-First Format

### Core Idea

Run LLMs on **CPUs** without GPUs. Designed for consumer laptops and PCs.

### How It Works (Simple Quantization)

1. **Find Range**
   ```
   min, max = weights.min(), weights.max()
   step_size = (max - min) / (2^bits - 1)
   ```

2. **Quantize**
   ```
   quantized = round((weight - min) / step_size)
   ```

3. **Dequantize (at inference)**
   ```
   weight ≈ quantized × step_size + min
   ```

### Quantization Levels

| Level | Bits/Weight | Size (7B) | Quality | Speed |
|-------|-------------|-----------|---------|-------|
| Q8 | 8 | 7GB | Excellent | Fast |
| Q5 | 5 | 4.4GB | Very Good | Medium |
| **Q4** | 4 | **3.5GB** | Good | Medium |
| Q3 | 3 | 2.6GB | Acceptable | Slow |
| Q2 | 2 | 1.75GB | Degraded | Slow |

**Most Popular:** Q4 (best quality/size tradeoff)

### Key Features

- Multiple precision levels (choose your tradeoff)
- CPU-optimized (SIMD, AVX2/AVX-512)
- Zero calibration needed
- Fast conversion (~2-3 min for 7B)

---

## Marlin: The Speed Kernel

**Marlin is NOT a quantization algorithm** — it's a CUDA kernel that runs GPTQ/AWQ models faster.

### Benchmark Results (Qwen2.5-32B on H200)

| Method | Without Marlin | With Marlin | Speedup |
|--------|----------------|-------------|---------|
| GPTQ | 276 tok/s | 712 tok/s | 2.6x |
| AWQ | 68 tok/s | 741 tok/s | **10.9x** |

### Why So Fast?

1. **Async Memory Copies** — Compute while loading
2. **L1 Cache Bypass** — Direct to shared memory
3. **Fused Dequantization** — No intermediate storage
4. **Optimized L2 Usage** — 80-95% cache hit rate

---

## Comparison Summary

### Quality (Perplexity)

```
AWQ > GGUF ≈ GPTQ
 95%   92%    90%   (of FP16 quality)
```

### Speed (with optimized kernels)

```
AWQ+Marlin > GPTQ+Marlin > GGUF (CPU)
   741 tok/s    712 tok/s    varies
```

### When to Use What

| Scenario | Recommendation |
|----------|----------------|
| **Production (Cloud)** | AWQ with Marlin |
| **High accuracy needed** | GPTQ |
| **Local development** | GGUF Q4 |
| **Edge devices** | AWQ |
| **No GPU** | GGUF (Q4-Q5) |
| **Minimal RAM** | GGUF Q2-Q3 |

---

## Practical Recommendations

### For Molting Project

**Goal:** Run 7-8B model locally on João's machine (31GB RAM, no dedicated GPU info)

**Recommended Path:**

1. **Start with GGUF Q4**
   - Works on CPU
   - 3.5GB for 7B model
   - Easy with llama.cpp or Ollama
   - Good quality for testing

2. **If GPU available:**
   - Use AWQ with vLLM or text-generation-inference
   - Much faster inference

3. **For fine-tuned models:**
   - Train with QLoRA (4-bit)
   - Export to GGUF for local inference
   - Or keep AWQ format for GPU

### Hardware Requirements

| Model Size | GGUF Q4 RAM | AWQ GPU VRAM |
|------------|-------------|--------------|
| 7B | ~4GB | ~4GB |
| 8B | ~4.5GB | ~4.5GB |
| 13B | ~8GB | ~8GB |
| 32B | ~18GB | ~18GB |

### Tools

| Format | Inference Tools |
|--------|-----------------|
| GGUF | llama.cpp, Ollama, LM Studio, text-generation-webui |
| AWQ | vLLM, text-generation-inference, HuggingFace |
| GPTQ | vLLM, AutoGPTQ, exllama |

---

## New Techniques (2025-2026)

### Dynamic 4-bit (Unsloth)

- Runtime-adaptive precision
- <10% more VRAM than standard 4-bit
- Better accuracy

### AffineQuant

- Affine transformations before quantization
- State-of-the-art W4A4 results
- More complex calibration

### BitBLAS

- Flexible bit-level operations
- Custom precision per layer
- Research stage

---

## References

1. [GPTQ Paper](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022
2. [AWQ Paper](https://arxiv.org/abs/2306.00978) — Lin et al., 2023 (MLSys 2024 Best Paper)
3. [Marlin Paper](https://arxiv.org/abs/2408.11743) — Frantar et al., 2024
4. [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
5. [vLLM Benchmarks](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)
6. [Visual Guide](https://newsletter.maartengrootendorst.com/p/which-quantization-method-is-right)

---

*Analysis by Cláudio for Project Molting*
