# Phase 5: Next Steps - Retrain with Curated Dataset

**Date:** 2026-02-11 14:24 BRT  
**Status:** ✅ Dataset curation complete

## What Was Done (Day 1)

✅ **Audited original dataset** - Identified agent-like patterns ("I'll start by...", "Let me...")  
✅ **Created curation script** - `curate_dataset.py` with 21 regex patterns (EN + PT-BR)  
✅ **Generated curated dataset** - 153 high-quality examples (from 484 original)  
✅ **Documented process** - Full report in `CURATION_REPORT.md`

## Curation Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total examples** | 484 | 153 | -68.4% |
| **Agent patterns removed** | - | 331 | 268 EN + 35 PT |
| **Quality threshold** | Mixed | High | Direct responses only |

### Top Removed Patterns

1. **"I'll start/continue by..."** → 168 cases
2. **"Let me read/check/analyze..."** → 107 cases  
3. **"Vou implementar/analisar..."** → 32 cases (Portuguese)

## Ready for Retraining

**Input file:** `dataset_sharegpt_curated.json` (153 examples)

### Recommended Training Config

```yaml
base_model: microsoft/Phi-3-mini-4k-instruct
dataset: dataset_sharegpt_curated.json
format: sharegpt
quantization: 4bit
lora_rank: 16
lora_alpha: 32
learning_rate: 2e-4
batch_size: 4
gradient_accumulation: 4
epochs: 5-7  # More epochs for smaller dataset
warmup_steps: 50
```

**Key change:** Increase epochs from 3 to 5-7 since dataset is smaller but higher quality.

## Expected Improvements

With curated dataset, the model should:

✅ **Reduce agent-like behavior** - No more "I'll start by..." or "Let me first..."  
✅ **More direct responses** - Jump straight to the answer  
✅ **Better personality preservation** - Training on authentic Cláudio responses only  
✅ **Less meta-commentary** - Fewer planning/thinking-out-loud statements

## Timeline Estimate

| Task | Duration | Notes |
|------|----------|-------|
| **Setup training env** | 30 min | Verify Unsloth/Axolotl dependencies |
| **First training run** | 1-2 hours | 5 epochs @ ~4 min/epoch |
| **Evaluation** | 30 min | Run `test_personality.py` on new model |
| **Compare results** | 15 min | Side-by-side vs Phase 3 model |
| **Document findings** | 30 min | Update evaluation docs |

**Total:** ~3-4 hours for complete retrain + eval cycle

## How to Run Training

```bash
cd /home/joao/Documentos/code/softtor/molting/experiments/fine-tuning

# Activate venv
source venv/bin/activate

# Run training with curated dataset
python train_qlora.py \
  --dataset dataset_sharegpt_curated.json \
  --output output/phase5-curated \
  --epochs 5 \
  --model_name microsoft/Phi-3-mini-4k-instruct
```

## Success Criteria

Compare Phase 5 vs Phase 3 models:

| Metric | Phase 3 | Phase 5 Target |
|--------|---------|----------------|
| **Personality score** | 7.4/10 | 8.5/10 |
| **Agent-like behavior** | Moderate | Minimal |
| **Response directness** | Mixed | High |
| **Portuguese fluency** | Good | Good+ |

## Iteration Plan

If results are still not satisfactory:

1. **Add more filters** - Remove remaining edge cases (e.g., `<thinking>` tags)
2. **Manual curation** - Hand-pick top 100 best examples
3. **Synthetic augmentation** - Generate 50-100 personality-rich examples with Opus
4. **Longer training** - Try 10 epochs with early stopping

## Files Generated

- ✅ `curate_dataset.py` - Curation script (executable)
- ✅ `dataset_sharegpt_curated.json` - Curated dataset (153 examples)
- ✅ `CURATION_REPORT.md` - Full curation statistics and patterns
- ✅ `PHASE5_NEXT_STEPS.md` - This file (retrain guide)

---

**Status:** Ready for retrain. Dataset is clean, script is documented, next session can jump straight into training.

**Recommendation:** Run training in a fresh session (allows for long GPU training without session timeout issues).
