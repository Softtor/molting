# Deployment Strategy - Molting Phase 4+

**Status:** 🟡 PoC Validated (7.4/10) - Not Production-Ready  
**Blocker:** Agent-like behavior in responses (embedded in model weights)  
**ETA to Production:** 2-4 days (after retraining with cleaned data)

---

## Current State (Phase 4 Complete)

### ✅ Achievements

- **Phase 1-2:** Training pipeline working (QLoRA on TinyLlama-1.1B)
- **Phase 3:** Personality transfer validated (7.4/10)
- **Phase 4:** Prompt engineering tested (limited impact)

### ⚠️ Blockers

1. **Agent-like behavior:** Model responds with task-planning language ("I'll first...", "Let me analyze...")
2. **Coherence issues:** Some responses go off-track (Q6 in Phase 3 eval)
3. **Verbosity:** Responses sometimes too long

### 🎯 Root Cause

Training data contains agent-like examples. Prompt engineering cannot override this at inference time.

---

## Deployment Options

### Option 1: Retrain with Cleaned Data (Recommended)

**Effort:** 1-2 days  
**Impact:** High (fixes root cause)  
**Risk:** Low (same pipeline, better data)

**Steps:**
1. Review training dataset (`experiments/fine-tuning/data/`)
2. Remove/rephrase agent-like examples
3. Add conversational Q&A pairs
4. Retrain (same QLo RA config)
5. Re-evaluate (Phase 3 test questions)

**Expected Outcome:** 8+/10 quality, production-ready

---

### Option 2: Post-Processing Filter

**Effort:** 4-6 hours  
**Impact:** Medium (masks symptoms)  
**Risk:** Medium (may truncate valid responses)

**Implementation:**
```python
import re

def filter_agent_language(response):
    # Remove meta-language sentences
    meta_patterns = [
        r"I'll (?:start|first|begin|now)\s+(?:by|with)",
        r"Let me (?:start|first|begin|analyze|explore)",
        r"I'm going to",
        r"My approach (?:is|would be)"
    ]
    
    # Split into sentences
    sentences = response.split('. ')
    
    # Keep only non-meta sentences
    filtered = []
    for sent in sentences:
        if not any(re.search(p, sent, re.I) for p in meta_patterns):
            filtered.append(sent)
    
    return '. '.join(filtered)
```

**Pros:**
- Quick to implement
- No retraining needed
- Can deploy immediately

**Cons:**
- Doesn't fix root cause
- May remove valid content
- Feels like a hack

---

### Option 3: Upgrade to 7B Model

**Effort:** 2-3 days  
**Impact:** High (better quality overall)  
**Risk:** Medium-High (requires more resources)

**Requirements:**
- GPU: A100 40GB or dual A6000
- Training time: 4-6 hours (vs. 2 hours for 1.1B)
- Inference: Slower (7B vs 1.1B params)

**Benefits:**
- Better instruction following
- Less overfitting on behavioral patterns
- Higher quality responses overall

**Drawbacks:**
- More expensive to run
- Slower inference
- May still need data cleanup

---

## Recommended Path

### Phase 5 Plan (Next 2-4 days)

#### Day 1: Data Cleanup
- [ ] Review training dataset line-by-line
- [ ] Remove agent-like patterns
- [ ] Add 20-30 conversational examples
- [ ] Split train/val sets (90/10)

#### Day 2: Retrain + Evaluate
- [ ] Retrain QLoRA with cleaned data
- [ ] Run Phase 3 evaluation (8 test questions)
- [ ] Target: 8+/10 quality score
- [ ] Compare: cleaned vs. original model

#### Day 3: Post-Processing Fallback
- [ ] Implement filter (Option 2)
- [ ] Test on edge cases
- [ ] Combine: cleaned model + light filtering

#### Day 4: Production Prep
- [ ] Create deployment package
- [ ] Write inference API wrapper
- [ ] Document usage examples
- [ ] Deploy to staging environment

---

## Production Architecture

### Inference Stack

```
User Query
   ↓
[API Gateway]
   ↓
[Load Balancer]
   ↓
[Model Server] ← QLoRA adapter + TinyLlama base
   ↓
[Post-Processing] ← Optional filter
   ↓
Response
```

### Model Serving Options

1. **Hugging Face Inference Endpoints** (easiest)
   - Managed service
   - Auto-scaling
   - ~$0.60/hour (GPU)

2. **Custom FastAPI + vLLM** (flexible)
   - Full control
   - Batch processing
   - Requires infrastructure

3. **Modal.com** (serverless)
   - Pay per request
   - Cold start: ~2-5s
   - Good for low-traffic

---

## Success Metrics

### Qualitative (Phase 5 Evaluation)

- [ ] No agent-like language in 4/4 test questions
- [ ] Coherent responses (on-topic, logical flow)
- [ ] Natural conversational tone
- [ ] Appropriate length (200-400 chars average)

### Quantitative

| Metric | Phase 3 (Current) | Phase 5 (Target) |
|--------|-------------------|------------------|
| Overall Quality | 7.4/10 | 8.5+/10 |
| Personality Transfer | 8/10 | 8+/10 (maintain) |
| Factual Accuracy | 7/10 | 8+/10 |
| Response Coherence | 6/10 | 8+/10 |
| Meta-Language (lower=better) | 50% | <20% |

---

## Rollback Plan

If cleaned model performs worse:
1. Keep original QLoRA adapter as fallback
2. Use post-processing filter on original model
3. Deploy with warning: "Responses may be verbose"

---

## Long-term Roadmap

### Phase 6: Production Hardening (Week 2-3)
- A/B testing (cleaned vs. filtered models)
- Human evaluation (5-10 users)
- Performance optimization (inference speed)
- Monitoring + logging

### Phase 7: Continuous Improvement (Month 2+)
- Collect production feedback
- Expand training data (user interactions)
- Fine-tune on new examples
- Experiment with larger models (7B)

---

**Last Updated:** 2026-02-11  
**Next Review:** After Phase 5 (retraining complete)
