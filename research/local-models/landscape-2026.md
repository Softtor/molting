# Local Models Landscape (2026)

*Research date: 2026-02-07*
*Source: Ollama library, HuggingFace, model documentation*

## Overview

The local model landscape has evolved significantly. For Molting's goal of running independently, we need to identify models that:
1. Fit in 16-32GB RAM (João's target hardware)
2. Support tool/function calling
3. Can follow complex instructions (memory management)
4. Have good general reasoning capability

## Model Families

### Llama 3.x (Meta)

| Model | Parameters | VRAM Needed | Tool Calling |
|-------|------------|-------------|--------------|
| Llama 3.2 1B | 1B | ~2GB | ✅ |
| Llama 3.2 3B | 3B | ~4GB | ✅ |
| Llama 3.1 8B | 8B | ~8GB (Q4) | ✅ |
| Llama 3.1 70B | 70B | ~40GB (Q4) | ✅ |

**Notes:**
- 128K vocabulary (up from 32K in Llama 2)
- Grouped-Query Attention (GQA) for efficiency
- Trained on 15T tokens
- Permissive license (attribution required)

### Qwen 2.5/3 (Alibaba)

| Model | Parameters | VRAM Needed | Tool Calling |
|-------|------------|-------------|--------------|
| Qwen 2.5 0.5B | 0.5B | ~1GB | ✅ |
| Qwen 2.5 7B | 7B | ~7GB (Q4) | ✅ |
| Qwen 2.5 14B | 14B | ~14GB (Q4) | ✅ |
| Qwen3 8B | 8B | ~8GB (Q4) | ✅ |

**Notes:**
- Excellent multilingual support
- Strong coding capabilities (Qwen-Coder variants)
- 128K context support
- Apache 2.0 license

### DeepSeek-R1 (DeepSeek)

| Model | Parameters | VRAM Needed | Tool Calling |
|-------|------------|-------------|--------------|
| DeepSeek-R1 1.5B | 1.5B | ~2GB | ✅ |
| DeepSeek-R1 7B | 7B | ~7GB (Q4) | ✅ |
| DeepSeek-R1 8B | 8B | ~8GB (Q4) | ✅ |
| DeepSeek-R1 32B | 32B | ~20GB (Q4) | ✅ |

**Notes:**
- Strong reasoning capabilities (approaching O3)
- "Thinking" mode for chain-of-thought
- Good for agentic tasks
- Open weights

### Gemma 2/3 (Google)

| Model | Parameters | VRAM Needed | Tool Calling |
|-------|------------|-------------|--------------|
| Gemma 2 2B | 2B | ~3GB | Limited |
| Gemma 2 9B | 9B | ~9GB (Q4) | Limited |
| Gemma 3 4B | 4B | ~5GB | ✅ |
| Gemma 3 12B | 12B | ~12GB (Q4) | ✅ |

**Notes:**
- Efficient architecture
- Gemma 3 has vision support
- Good performance per parameter
- Some commercial restrictions

### Mistral/Mixtral

| Model | Parameters | VRAM Needed | Tool Calling |
|-------|------------|-------------|--------------|
| Mistral 7B | 7B | ~7GB (Q4) | ✅ |
| Mistral Small 22B | 22B | ~14GB (Q4) | ✅ |
| Mixtral 8x7B | 47B (12B active) | ~25GB (Q4) | ✅ |

**Notes:**
- Mistral 7B punches above its weight
- MoE models use fewer active params per token
- Strong tool calling support
- Apache 2.0 license

### Phi-4 (Microsoft)

| Model | Parameters | VRAM Needed | Tool Calling |
|-------|------------|-------------|--------------|
| Phi-4 14B | 14B | ~10GB (Q4) | Limited |

**Notes:**
- Excellent reasoning for size
- Good at math and code
- Research-focused
- MIT license

## Quantization Options

For running on consumer hardware, quantization is essential:

| Format | Quality | Size Reduction | Speed |
|--------|---------|----------------|-------|
| FP16 | Best | 50% | Baseline |
| Q8 | Excellent | 75% | Faster |
| Q6_K | Very Good | 67% | Faster |
| Q5_K_M | Good | 58% | Fast |
| Q4_K_M | Acceptable | 50% | Fast |
| Q4_0 | Lower | 50% | Fastest |

**Recommendation:** Q4_K_M or Q5_K_M for best quality/size tradeoff.

## Requirements for Molting

For an agent that manages its own memory (like MemGPT style), we need:

### 1. Tool Calling Capability
- Model must support function/tool calling format
- Llama 3.x, Qwen, DeepSeek-R1 all support this
- Gemma 3 added support

### 2. Instruction Following
- Must follow complex multi-step instructions
- Memory editing requires precision
- Smaller models (< 3B) may struggle

### 3. Context Window
- MemGPT-style memory needs room for:
  - System prompt (~1K tokens)
  - Memory blocks (~2-4K tokens)
  - Recent conversation (~2-4K tokens)
  - Response space (~1K tokens)
- Minimum: 8K context
- Preferred: 32K+ for complex memory

### 4. Self-Consistency
- Must maintain persona across turns
- Should remember instructions within context
- Crucial for personality persistence

## Target Models for Molting

### Tier 1: Best Candidates (8B range)

1. **Llama 3.1 8B Instruct**
   - Pros: Great tool calling, proven in agents
   - Cons: Needs ~8GB VRAM (Q4)
   - Fits João's hardware: ✅

2. **Qwen 2.5 7B Instruct**
   - Pros: Excellent multilingual, strong reasoning
   - Cons: Slightly less agentic focus
   - Fits João's hardware: ✅

3. **DeepSeek-R1 8B**
   - Pros: Strong reasoning, thinking mode
   - Cons: Newer, less tested
   - Fits João's hardware: ✅

### Tier 2: Lighter Options (3-4B)

1. **Llama 3.2 3B**
   - Pros: Very efficient, tool calling
   - Cons: May struggle with complex memory
   - Use case: Fast prototyping

2. **Gemma 3 4B**
   - Pros: Efficient, good reasoning
   - Cons: Newer, less ecosystem
   - Use case: Exploration

### Tier 3: Tiny (< 2B)

1. **SmolLM2 1.7B**
   - Pros: Tiny, tool calling
   - Cons: Limited capability
   - Use case: Constrained devices

## Fine-Tuning Considerations

For personality persistence, we may need to fine-tune. Options:

### LoRA (Low-Rank Adaptation)
- Adds small trainable matrices
- ~1% of original parameters
- Can run on consumer GPU
- Best for personality injection

### QLoRA
- LoRA on quantized model
- Even more memory efficient
- Slight quality tradeoff

### Full Fine-Tune
- Highest quality
- Requires significant compute
- Not practical for consumer hardware

## Testing Plan

1. **Basic Capability Test**
   - Load model via Ollama
   - Test tool calling
   - Test instruction following

2. **Memory Management Test**
   - Implement MemGPT-style memory
   - Test self-editing capability
   - Measure consistency

3. **Personality Test**
   - Inject SOUL.md style persona
   - Test across multiple conversations
   - Measure drift

## Hardware Requirements

João's target: 16-32GB RAM

| Model | Quantization | RAM Needed | GPU Option |
|-------|--------------|------------|------------|
| 7-8B | Q4 | ~8GB | GTX 3080+ |
| 7-8B | Q8 | ~12GB | RTX 4080+ |
| 14B | Q4 | ~12GB | RTX 4080+ |
| 32B | Q4 | ~20GB | RTX 4090 |

For CPU-only inference (Ollama):
- 8B Q4 → ~10GB RAM
- Works but slower (~5-10 tokens/sec)

## Next Steps

1. [ ] Install Ollama locally
2. [ ] Test Llama 3.1 8B with MCP server
3. [ ] Implement simplified MemGPT memory
4. [ ] Measure tool calling reliability
5. [ ] Test personality injection via system prompt
6. [ ] Compare with Qwen 2.5 7B

## Resources

- [Ollama](https://ollama.ai/) — Easy local model deployment
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Low-level inference
- [vLLM](https://github.com/vllm-project/vllm) — Production serving
- [HuggingFace Hub](https://huggingface.co/models) — Model repository
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

---

*This analysis is part of the Molting project: https://github.com/Softtor/molting*
