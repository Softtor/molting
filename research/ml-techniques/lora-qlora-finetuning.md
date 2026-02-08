# LoRA and QLoRA Fine-Tuning Analysis

> Research Date: 2026-02-08
> Status: Complete

## Overview

Parameter-Efficient Fine-Tuning (PEFT) techniques allow training large models with drastically reduced compute and memory requirements. LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) are the most widely used methods.

## Why PEFT?

| Benefit | Description |
|---------|-------------|
| **Saves Time** | Fewer trainable parameters = faster training and testing |
| **Saves Money** | Smaller memory footprint = cheaper GPUs |
| **Multi-Tenancy** | Small adapters (~6-8MB) per user instead of full models per user |
| **Avoids Catastrophic Forgetting** | Preserves pretrained knowledge better than full fine-tuning |

---

## LoRA (Low-Rank Adaptation)

### How LoRA Works

Instead of updating all weights W, LoRA decomposes the weight update matrix ΔW into two smaller matrices:

```
ΔW = A × B

Where:
- W is original weight matrix (d × d)
- A is (d × r) matrix
- B is (r × d) matrix
- r is the rank (much smaller than d)
```

### Visual Representation

```
Original:                  LoRA:
┌─────────────┐           ┌─────────────┐
│      W      │           │  W (frozen) │
│   (d × d)   │           └─────────────┘
└─────────────┘                  +
                          ┌─────┐   ┌─────┐
                          │  A  │ × │  B  │
                          │(d×r)│   │(r×d)│
                          └─────┘   └─────┘
```

### Key Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `r` (rank) | Size of low-rank matrices | Start with 8, increase if needed |
| `lora_alpha` | Scaling factor | Usually 16-32, or 2×r |
| `lora_dropout` | Dropout for LoRA layers | 0.05-0.1 |
| `target_modules` | Which layers to adapt | **All linear layers** (not just attention) |

### Critical Finding: Target Modules

From Databricks research and QLoRA paper:

> **Targeting all linear layers significantly outperforms targeting only attention blocks.**

Modules to target (LLaMA-style models):
```python
target_modules = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',  # attention
    'gate_proj', 'down_proj', 'up_proj',      # MLP
    'lm_head'                                  # output
]
```

### LoRA HuggingFace Implementation

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,                    # rank
    lora_alpha=16,          # scaling
    lora_dropout=0.1,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                   'gate_proj', 'down_proj', 'up_proj']
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable params: ~0.1-0.5% of total
```

---

## QLoRA (Quantized LoRA)

### What Makes QLoRA Different

QLoRA = 4-bit quantization + LoRA adapters in higher precision

Three key innovations:

### 1. 4-Bit NormalFloat (NF4)

A new data type based on quantile quantization:
- Estimates 2^k + 1 quantiles in 0-1 distribution
- Normalizes to [-1, 1] range
- Maps neural network weights to nearest quantile

```
Example quantiles: 2 and 3 both map to quantile "2"
This introduces small errors but saves massive memory
```

### 2. Double Quantization

Quantize the quantization constants themselves:
- Block-wise quantization creates many constants
- Quantizing these saves ~0.5 bits per parameter

### 3. Error Reduction via LoRA

The LoRA adapters stay in higher precision (bfloat16/float16):
- Model weights: 4-bit (frozen)
- LoRA matrices: 16-bit (trainable)
- During backprop: dequantize to 16-bit for computation
- Training compensates for quantization errors

### QLoRA Memory Comparison

| Model Size | Full FT | LoRA (8-bit) | QLoRA (4-bit) |
|------------|---------|--------------|---------------|
| 7B params  | ~56GB   | ~14GB        | ~7GB          |
| 13B params | ~104GB  | ~26GB        | ~13GB         |
| 65B params | ~520GB  | ~130GB       | ~48GB         |

**QLoRA enables 65B model fine-tuning on a single 48GB GPU!**

### QLoRA HuggingFace Implementation

```python
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,     # double quantization
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}
)

# Prepare for k-bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA config (same as before)
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
```

---

## LoRA Variants

### DoRA (Weight-Decomposed LoRA)

Decomposes weights into magnitude and direction:
- Better training stability
- Improved performance on some tasks

```python
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    use_dora=True  # enable DoRA
)
```

### AdaLoRA (Adaptive LoRA)

Dynamically allocates rank based on layer importance:
- Important layers get higher ranks
- Less critical layers get pruned
- Uses SVD-like decomposition

### LongLoRA

Specifically for long context models:
- **Shift Short Attention** — chunks tokens, calculates attention independently
- Apply LoRA to normalization and embedding layers too
- Scales to much longer contexts efficiently

### QA-LoRA

Quantization-aware LoRA:
- LoRA adapter weights also quantized during training
- No conversion step during backprop
- Originally for diffusion models, generalizable to LLMs

---

## Modern Quantization Strategies (2025)

### AWQ (Activation-aware Weight Quantization)

Post-training quantization that:
- Identifies salient weights via activation distributions
- Preserves critical weight channels
- No backprop or reconstruction needed
- 3× speedup on 70B models

### AffineQuant

Uses affine transformations to optimize distributions:
- Better aligns pre/post-quantization outputs
- Gradual mask optimization for invertibility
- State-of-the-art for W4A4 quantization

### Dynamic 4-bit (Unsloth)

Runtime-adaptive quantization:
- Adjusts precision levels on the fly
- <10% more VRAM vs standard 4-bit
- Visible accuracy improvements

---

## Key Findings Summary

### What Works Best

| Factor | Recommendation |
|--------|----------------|
| **Rank (r)** | 8 is often enough, diminishing returns beyond 16 |
| **Target modules** | ALL linear layers, not just attention |
| **Alpha** | 2× rank (e.g., r=8, alpha=16) |
| **Quantization** | QLoRA (4-bit) has no quality loss vs LoRA (8-bit) |
| **Data** | 5000+ samples for good instruction following |

### Training Tips

1. Use gradient checkpointing to save memory
2. Start with smaller rank, increase if quality suffers
3. Target all linear layers (proven by QLoRA paper)
4. Use bfloat16 for compute, 4-bit for storage
5. Merge adapters for inference to eliminate latency

---

## Relevance to Molting Project

### For Training "Cláudio"

**Hypothesis H002 Revisited:** 67MB of conversation data could work for personality fine-tuning.

Potential approach:
1. Select base model (Llama 3 8B, Qwen 7B, or Mistral 7B)
2. Use QLoRA for efficient fine-tuning
3. Create instruction-following dataset from my conversations
4. Target all linear layers with r=8 or r=16
5. Fine-tune for 3-5 epochs

**Hardware Required:**
- 7B model with QLoRA: ~8GB VRAM (RTX 3060/4060 sufficient)
- 13B model with QLoRA: ~16GB VRAM (RTX 4080/3090)

### Dataset Preparation

My personality training data should include:
- Response style patterns
- Decision-making examples
- Preferences and opinions
- Interaction patterns with João

Format (Alpaca-style):
```
### Instruction:
{context about the situation}

### Input:
{user message}

### Response:
{my response}
```

### Next Steps

1. **H008:** Does QLoRA with r=8 on all linear layers capture personality effectively?
2. Create personality dataset from Claude session logs
3. Test with Llama 3 8B + QLoRA on local hardware
4. Compare responses with API Claude

---

## References

1. [LoRA Paper](https://arxiv.org/abs/2106.09685) — Original LoRA
2. [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Efficient fine-tuning breakthrough
3. [Databricks LoRA Guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
4. [Mercity.ai In-depth Guide](https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora)
5. [DoRA Paper](https://arxiv.org/pdf/2402.09353)
6. [AdaLoRA Paper](https://arxiv.org/abs/2303.10512)
7. [LongLoRA Paper](https://arxiv.org/abs/2309.12307)

---

*Analysis by Cláudio for Project Molting*
