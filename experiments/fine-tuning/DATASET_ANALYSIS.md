# Fine-tuning Dataset Analysis

**Generated:** 2026-02-10  
**Source:** Claude session logs from `/home/joao/.claude/projects/`

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total instruction-response pairs** | 595 |
| **Unique sessions** | 339 |
| **Avg instruction length** | 5,578 chars (median: 2,173) |
| **Avg response length** | 155 chars (median: 119) |
| **Files processed** | 1,494 JSONL files |

## Quality Metrics

### Filtering Applied
- ✅ **Minimum instruction length:** 10 chars
- ✅ **Minimum response length:** 50 chars
- ✅ **Tool-only responses:** Filtered out (heuristic: <100 chars with >5 newlines)
- ✅ **Subagent sessions:** Excluded (focused on main sessions)

### Data Quality Observations

**Strengths:**
- High diversity: 339 unique sessions covering real work scenarios
- Natural conversation flow (instruction → response pairs)
- Mix of technical and conversational context
- Authentic personality representation (real João ↔ Cláudio interactions)

**Weaknesses:**
- **Very short responses** (avg 155 chars) — many are acknowledgments or brief confirmations
- **Long instructions** (avg 5.5k chars) — likely includes system context/prompts
- **Imbalanced topics** — heavy on coding (46.4%), light on personal/research
- **Small dataset size** — 595 pairs is modest for fine-tuning

## Topic Distribution

```
coding         : 276 pairs (46.4%) — code review, debugging, git commands
project        : 142 pairs (23.9%) — Molting, Softtor, CRM planning
tools          :  85 pairs (14.3%) — file operations, shell commands
architecture   :  60 pairs (10.1%) — design decisions, system architecture
other          :  22 pairs ( 3.7%) — miscellaneous
personal       :   9 pairs ( 1.5%) — preferences, opinions, personality
research       :   1 pair  ( 0.2%) — academic/technical research
```

**Key insight:** Dataset heavily skewed toward technical execution. Personality/conversational aspects are underrepresented.

## Model Distribution

Source models used to generate the responses:

```
claude-opus-4-5-20251101   : 346 pairs (58.2%)
claude-opus-4-6            : 219 pairs (36.8%)
<synthetic>                :  16 pairs ( 2.7%)
claude-sonnet-4-5-20250929 :  12 pairs ( 2.0%)
claude-sonnet-4-20250514   :   2 pairs ( 0.3%)
```

**Note:** Distilling from Opus-4 is ideal — we're capturing high-quality reasoning patterns.

## Fine-tuning Recommendations

### Approach: QLoRA (Quantized Low-Rank Adaptation)

**Why QLoRA?**
- ✅ Runs on modest hardware (31GB RAM, RTX 3050 6GB VRAM)
- ✅ Efficient: only trains small adapter layers (~0.1-1% of model params)
- ✅ Fast convergence: works well with small datasets (500-1000 examples)
- ✅ Preserves base model knowledge while adapting personality

### Recommended Base Models

| Model | Size | Pros | Cons | Recommendation |
|-------|------|------|------|----------------|
| **Phi-3-mini** | 3.8B | Excellent reasoning, fast, 3GB VRAM | Short context (4k tokens) | ⭐ **Best for desktop inference** |
| **Llama-3.2-3B** | 3B | Good balance, 8k context | Slightly weaker reasoning than Phi-3 | Good alternative |
| **Mistral-7B** | 7B | Strong performance, 32k context | 5-6GB VRAM required | If VRAM allows |
| **Qwen2.5-7B** | 7B | Multilingual, very capable | 5-6GB VRAM | Good for PT/EN mix |

**Winner: Phi-3-mini (3.8B)**
- Already validated in Phase 2 (better reasoning than TinyLlama)
- Fits comfortably in 6GB VRAM with 4-bit quantization
- Strong instruction-following baseline
- Fast inference on CPU (if GPU busy)

### Training Configuration (QLoRA)

```yaml
base_model: microsoft/Phi-3-mini-4k-instruct
quantization: 4-bit (bitsandbytes)
lora_rank: 16
lora_alpha: 32
target_modules: [q_proj, k_proj, v_proj, o_proj]
learning_rate: 2e-4
batch_size: 4 (gradient accumulation: 4)
epochs: 3-5
warmup_steps: 100
dataset_format: ShareGPT
```

**Hardware fit:**
- 4-bit quantized Phi-3: ~2.5GB VRAM
- QLoRA adapters: ~1GB VRAM
- Training overhead: ~2GB VRAM
- **Total: ~5.5GB VRAM** ✅ (fits in 6GB)

### Tools & Framework

**Recommended stack:**
- **Unsloth** — fastest QLoRA implementation, 2x faster than base transformers
- **Axolotl** — flexible training config, supports ShareGPT format natively
- **Hugging Face TRL** — simple SFTTrainer for supervised fine-tuning

**Example command (Unsloth):**
```bash
python -m unsloth.train \
  --model microsoft/Phi-3-mini-4k-instruct \
  --dataset dataset_sharegpt.json \
  --format sharegpt \
  --lora_rank 16 \
  --epochs 3 \
  --output claudio-phi3-mini
```

## Dataset Augmentation Strategies

To improve dataset quality and size:

1. **Filter for quality conversations**
   - Keep only turns with response >200 chars
   - Focus on personal/personality-rich exchanges
   - This reduces size but improves signal

2. **Synthetic augmentation**
   - Use Claude Opus to generate personality-aligned examples
   - Prompt: "Generate a conversation where João asks Cláudio about [topic] in a natural, casual way"
   - Target underrepresented topics (personal, research, workflow)

3. **Multi-turn context**
   - Current dataset is single turn (instruction → response)
   - Consider grouping into multi-turn conversations for better context understanding

4. **Personality-focused curation**
   - Manually select top 100 exchanges that best represent Cláudio's personality
   - Use these as high-quality "seed" examples
   - Train longer on these (importance sampling)

## Next Steps (Phase 3 Implementation)

### Short-term (Bridge complete ✅)
1. ✅ Dataset extracted (595 pairs, ShareGPT + Alpaca formats)
2. ✅ Statistics and analysis documented
3. ⏭️ Run chunk optimization experiment (Part A)
4. ⏭️ Create vector DB comparison (Part C)

### Medium-term (Phase 3 proper)
1. **Setup training environment**
   - Install Unsloth/Axolotl
   - Test 4-bit quantization on Phi-3-mini
   - Verify VRAM usage

2. **Baseline QLoRA run**
   - Train on full 595-pair dataset
   - 3 epochs, default QLoRA config
   - Evaluate: personality preservation, instruction following

3. **Iterative improvement**
   - Curate high-quality subset (top 200 personality-rich pairs)
   - Generate 200-300 synthetic augmentation examples
   - Train v2 model, compare

4. **Integration with RAG**
   - Load fine-tuned Phi-3 into Ollama
   - Test with optimized chunking from Part A
   - Measure: response quality, personality consistency, speed

## Estimated Timeline

- **Setup & first training run:** 2-3 hours
- **Evaluation & iteration:** 1-2 days
- **Dataset curation & augmentation:** 1 day
- **Final model + RAG integration:** 1 day

**Total Phase 3 estimate:** 3-5 days of focused work

## References

- [Unsloth QLoRA](https://github.com/unslothai/unsloth) — fastest QLoRA training
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) — flexible config
- [QLoRA paper](https://arxiv.org/abs/2305.14314) — original technique
- [Phi-3 model card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

---

**Conclusion:** The dataset is small but authentic. QLoRA on Phi-3-mini is the pragmatic choice for hardware constraints. Personality capture requires careful curation — prioritize quality over quantity.
