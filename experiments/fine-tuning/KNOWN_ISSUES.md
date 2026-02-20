# Known Issues — Fine-tuning Environment

## Issue: Phi-3-mini Inference OOM with transformers 5.x + BNB 0.49

**Status:** Open  
**Versions:** transformers 5.1.0 + bitsandbytes 0.49.1  
**GPU:** RTX 3050 4GB  

### Symptom
Loading Phi-3-mini (3.8B) with `quantization_config=BitsAndBytesConfig(load_in_4bit=True)` fails 
with CUDA OOM at ~52% of weight loading. OOM occurs in `core_model_loading.py:materialize_tensors()`.

### Root Cause
Transformers 5.x rewrote the model loading backend (`core_model_loading.py`). The new 
materialization pipeline does not correctly apply BnB 4-bit quantization layer-by-layer during loading.
Instead, tensors are materialized in float16 (2 bytes/param) before quantization, requiring:
- 3.8B × 2 bytes ≈ 7.6 GB VRAM  
- Available: 3.8 GB → OOM at ~52%

Training works because it uses a different code path:
- `prepare_model_for_kbit_training()` uses older gradient checkpointing integration
- Training peak: 3.31 GB (working within 3.8 GB)

### Workaround
- Use TinyLlama (1.1B) for inference — loads fine (0.82 GB with 4-bit)
- For Phi-3-mini: downgrade transformers to 4.x or use CPU-only inference (7.6 GB RAM needed)
- Or: wait for transformers 5.x + bitsandbytes compatibility fix

### Impact
Cannot evaluate Phi-3-mini adapter quality via inference.
**Mitigation:** Use training loss as quality proxy.
- TinyLlama final loss: 2.47 (poor convergence)
- Phi-3-mini final loss: 1.41 (better convergence, ~43% lower)

The loss improvement strongly suggests Phi-3-mini adapter is better, but cannot confirm via rubric.
