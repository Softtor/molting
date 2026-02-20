# Known Issues — Fine-tuning Experiments

## Status as of Phase 7 (2026-02-20)

---

## Issue 1: Phi-3-mini inference OOM on 4GB GPU (Partially resolved)

### Root Cause (Phase 7 diagnosis)
Two separate issues were blocking Phi-3-mini inference:

**Issue 1a: rope_scaling KeyError (FIXED)**
- Old cached modeling_phi3.py expected `rope_scaling["type"]`
- New HuggingFace config uses `rope_scaling["rope_type"]` format
- Fix: patched `~/.cache/huggingface/modules/.../modeling_phi3.py`
  - Changed `self.config.rope_scaling["type"]` to `.get("type") or .get("rope_type", "default")`
  - Added handling for `"default"` rope_type → use standard Phi3RotaryEmbedding

**Issue 1b: transformers 5.x materialization OOM (ACTIVE)**
- transformers 5.x uses new `core_model_loading.py` with threaded tensor materialization
- This path bypasses bitsandbytes 4-bit quantization hooks (which worked in transformers 4.x)
- Result: model materializes in fp16 on CUDA → OOM on 4GB GPU for 3.8B model
- TinyLlama (1.1B) is unaffected — small enough to load anyway
- Phi-3-mini (3.8B) requires ~1.9GB at 4-bit but ~7.6GB at fp16

### Fix Options (for Phase 8)
1. **Downgrade transformers to 4.47.x** — `pip install transformers==4.47.0` (safest)
2. **Use llama.cpp backend** — `ollama pull phi3` or direct GGUF inference
3. **Load on CPU then convert to 4-bit** — but needs 7.6GB RAM
4. **Wait for bitsandbytes compatibility update** — track HF issue tracker

### Training Status
- TinyLlama: ✅ Training works, eval works
- Phi-3-mini: ✅ Training worked (Phase 6, transformers 4.x era), ❌ inference blocked

---

## Issue 2: TinyLlama 1.1B response quality ceiling

### Symptom
Phase 7 TinyLlama (fixed template, system prompt, 8 epochs): 5.5/10
Phase 6 TinyLlama (broken template): 2.9/10
Still failing on: factual accuracy (25%), response coherence (43%)

### Root Cause
1.1B parameters insufficient for maintaining identity + factual accuracy simultaneously
Notable failures:
- "João as father/son" hallucination (confused relationship)
- System prompt structure bleeding into responses (Q6 — overfitting artifact)
- Truncation at 512 tokens (system prompt takes ~400 tokens itself!)

### Fix Options (Phase 8)
1. **max_length: 512 → 1024** (system prompt alone is ~400 tokens)
2. **Shorter system prompt** — summarize to ~100 tokens, keep full prompt as context only
3. **Phi-3-mini** (3.8B) once inference is unblocked — primary solution
4. **Negative examples** in dataset — show wrong João descriptions to counteract hallucination
5. **LoRA r=32** for more capacity (current: r=16)
6. **Dataset: 79 → 200+ examples** for better generalization vs memorization

---

## Issue 3: Chat template inconsistency (RESOLVED)

### What was wrong
Phase 6 training used `<|endoftext|>` as separator instead of TinyLlama's correct `</s>`.
This caused the model to learn from a malformed format, degrading all responses.

### Resolution (Phase 7)
- Fixed `format_sharegpt_fixed()` to use `</s>` as EOS separator
- Verified format: `<|system|>\n{sys}</s>\n<|user|>\n{q}</s>\n<|assistant|>\n{a}</s>`
- Training loss improved: previous run unknown, Phase 7: 0.9171

---

## Summary Table

| Issue | Status | Phase |
|-------|--------|-------|
| Chat template `<\|endoftext\|>` → `</s>` | ✅ Fixed | 7 |
| System prompt missing during training | ✅ Fixed | 7 |
| grad_accum 8→4 | ✅ Fixed | 7 |
| Epochs 5→8 | ✅ Fixed | 7 |
| rope_scaling KeyError (Phi-3 cached code) | ✅ Fixed | 7 |
| transformers 5.x OOM for Phi-3 inference | ❌ Active | 7 |
| TinyLlama quality ceiling (1.1B capacity) | ⚠️ Mitigated | 7 |
| max_length too small (512, sys prompt=400) | ❌ Active | 7 |
