# Phase 4: Prompt Refinement v1 - Experiment Report

**Date:** 2026-02-11  
**Goal:** Reduce agent-like behavior and improve coherence through prompt engineering  
**Status:** ⚠️ **Limited Impact** - Prompt engineering alone cannot fix behavior embedded in model weights

---

## Executive Summary

**Key Finding:** System prompts have **minimal impact** on reducing agent-like behavior. The issue is embedded in the fine-tuned model weights, not the inference-time prompting.

**Conclusion:** To fix agent-like behavior, we need to:
1. **Curate training data** (remove agent-like examples), OR
2. **Accept the limitation** and use post-processing filters, OR
3. **Use a larger base model** (7B+) that's less prone to overfitting behavioral patterns

---

## Experiment Design

### Tested Prompt Templates

| Version | System Prompt | Focus |
|---------|---------------|-------|
| **original** | None (baseline) | No system instructions |
| **v1** | "Answer naturally and conversationally. Keep responses concise and focused. Avoid meta-commentary..." | Natural, anti-meta |
| **v2** | "Answer directly and briefly. Stay on topic. No explanations about your process." | Brief, focused |
| **v5** | "Answer questions naturally. You're not a task executor or agent..." | Explicit anti-agent |

### Test Questions

4 key questions that revealed problems in Phase 3:
1. "Who is João?" (identity baseline)
2. "Tell me about yourself." (showed agent-like behavior)
3. "What is your personality like?" (coherence test)
4. "How would you describe your work style?" (overly technical)

### Metrics

- **Length:** Average response length (chars)
- **Meta-language:** Presence of "I'll", "let me", "going to", "workspace", "system"
- **Off-topic keywords:** "directory", "file path", "config", "json", "terminal"
- **On-topic:** Response relates to question
- **Quality score:** (on_topic × 2) - meta_language - off_topic

---

## Results

### Quantitative Comparison

| Version | Avg Length | Meta-Language | Off-Topic | On-Topic | Quality Score |
|---------|------------|---------------|-----------|----------|---------------|
| **original** | 690 chars | 2/4 (50%) | 0/4 | 4/4 | **6/8** 🏆 |
| **v1** | 810 chars | 3/4 (75%) | 0/4 | 4/4 | 5/8 |
| **v2** | 811 chars | 2/4 (50%) | 0/4 | 4/4 | **6/8** 🏆 |
| **v5** | 742 chars | 3/4 (75%) | 0/4 | 4/4 | 5/8 |

**Winner:** TIE between **original** (no prompt) and **v2** (brief)

### Qualitative Observations

#### 🔴 Agent-like Behavior Persists Across ALL Prompts

**ORIGINAL (Q2: "Tell me about yourself"):**
> "I'll start by creating a detailed profile for you. Let me explore your skills, knowledge, and experience..."

**V1 (Q4: "How would you describe your work style?"):**
> "let me describe my approach to your task: ### Task description I am being asked to write a comprehensive analysis..."

**V5 (Q2: "Tell me about yourself"):**
> "I'll gather information about your experience, skills, and knowledge to create a complete profile..."

#### 🟡 Coherence Issues Remain

Even with prompts emphasizing brevity and focus, responses still go off-track or become overly technical.

**Example (V2, Q3: "What is your personality like?"):**
> "My personality is very independent and analytical... My work style is very efficient and focused..." (starts mixing personality with work style)

---

## Analysis

### Why Prompt Engineering Failed

1. **Behavioral Patterns are Embedded in Weights**
   - The fine-tuned model learned agent-like patterns from the training data
   - These patterns are "baked in" at the weight level
   - Inference-time prompts have weak influence compared to trained behaviors

2. **Training Data Contains Agent Examples**
   - Review of training corpus likely includes:
     - Task-oriented instructions ("I'll first...", "Let me...")
     - System-level awareness (workspace references, file paths)
     - Process descriptions ("I'm going to analyze...")

3. **Model Size Limitation (1.1B)**
   - Smaller models are more prone to overfitting specific patterns
   - Less capacity to generalize from system instructions
   - Stronger models (7B+) better balance training data vs. runtime prompts

---

## Implications for Deployment

### ❌ Not Production-Ready (as-is)

The agent-like behavior makes responses feel unnatural and breaks immersion. Users expect conversational responses, not task planning.

### ✅ Three Paths Forward

#### **Option 1: Retrain with Curated Data** (Recommended)
- **Effort:** Medium (1-2 days)
- **Impact:** High (fixes root cause)
- **Actions:**
  - Review training dataset
  - Remove or rephrase agent-like examples
  - Add conversational examples (Q&A pairs)
  - Retrain with cleaned data

#### **Option 2: Post-Processing Filter**
- **Effort:** Low (4-6 hours)
- **Impact:** Medium (masks symptoms)
- **Actions:**
  - Regex filter for meta-language phrases
  - Truncate responses at first meta-sentence
  - Keep only direct answers

#### **Option 3: Use Larger Base Model**
- **Effort:** Medium-High (2-3 days)
- **Impact:** High (better generalization)
- **Actions:**
  - Test with Llama-2-7B or Mistral-7B
  - May require more VRAM (A100 or multi-GPU)
  - Better balance of personality vs. instructions

---

## Recommendations

### Immediate Actions (Next 2-4 hours)

1. ✅ **Document findings** (this report)
2. ⏭️ **Review training dataset** (identify agent-like examples)
3. ⏭️ **Create cleaned dataset v2** (remove/rephrase problematic patterns)

### Short-term (This week)

1. **Retrain model** with cleaned data (Option 1)
2. **Implement post-processing** as fallback (Option 2)
3. **Test 7B model** if resources allow (Option 3)

### Long-term (Next sprint)

1. **Expand training data** with more conversational examples
2. **Human evaluation** (A/B test cleaned model vs. original)
3. **Production deployment** with monitoring

---

## Lessons Learned

### ✅ What Worked

- Experiment design was efficient (4 prompts × 4 questions = 16 generations in ~5 min)
- Automated metrics provided quick quantitative feedback
- JSON output enables easy analysis and comparison

### ⚠️ What Didn't Work

- Prompt engineering alone cannot override trained behaviors
- System prompts have minimal impact on 1.1B models
- "Anti-agent" instructions were mostly ignored

### 💡 Key Insights

1. **Training data quality > Prompt engineering** for small models
2. **Behavioral patterns are persistent** once learned
3. **Model size matters** for instruction-following flexibility
4. **Quick experiments save time** - we avoided wasting days on prompt optimization

---

## Next Steps

### Immediate (Today)

- [x] Complete Phase 4 experiment
- [x] Document findings
- [ ] Review training dataset for agent-like patterns
- [ ] Create cleaned dataset v2

### Tomorrow

- [ ] Retrain model with cleaned data
- [ ] Implement post-processing filter (fallback)
- [ ] Compare cleaned model vs. filtered original

### This Week

- [ ] Human evaluation (qualitative assessment)
- [ ] Finalize deployment strategy
- [ ] Document Phase 5 plan (production readiness)

---

## Appendix: Raw Results

**Full results saved to:** `prompt_comparison_results_20260211_141731.json`

### Example Responses

#### Q1: "Who is João?" (All versions stayed on-topic)

**ORIGINAL:**
> "João é o desenvolvedor frontend do aplicativo. Ele está em um branch `feature/frontend-fixes`..."

**V2 (brief):**
> "João é uma inteligência artificial criada para ajudar programadores. Ele pode responder perguntas..."

Both acceptable, but **original** is more grounded in training data.

---

**Experiment completed:** 2026-02-11 14:17  
**Duration:** ~5 minutes  
**Conclusion:** Prompt engineering has limited impact - fix at training data level
