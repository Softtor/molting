# Dataset Curation Report - Phase 5

**Generated:** 2026-02-11 14:23:50  
**Agent:** Cláudio (subagent molting-phase5-feb11)

## Goal

Remove "agent-like" patterns from training data to improve personality preservation:
- Patterns like "I'll start by...", "Let me first...", "I need to..."
- Tool-only responses without personality
- Overly short acknowledgments

## Input/Output

- **Input:** `dataset_sharegpt_filtered.json`
- **Output:** `dataset_sharegpt_curated.json`

## Results

| Metric | Value |
|--------|-------|
| **Total examples** | 484 |
| **Kept** | 153 (31.6%) |
| **Removed** | 331 (68.4%) |

## Removal Breakdown

- **agent_pattern: I'?ll\s+(start|begin|first|now|continue)\s+(by|wit:** 168 (50.8% of removed)
- **agent_pattern: Let me\s+(start|begin|first|read|check|analyze|exa:** 107 (32.3% of removed)
- **agent_pattern: Vou\s+(implementar|analisar|começar|primeiro|ler|v:** 32 (9.7% of removed)
- **agent_pattern: I'?ll\s+(read|check|examine|analyze|look at|review:** 16 (4.8% of removed)
- **agent_pattern: Deixe-me\s+(ler|analisar|começar|verificar|primeir:** 3 (0.9% of removed)
- **agent_pattern: I need to\s+(first|start|begin|read|check|analyze):** 2 (0.6% of removed)
- **tool_only:** 1 (0.3% of removed)
- **agent_pattern: Primeiro,?\s+(vou|deixe-me):** 1 (0.3% of removed)
- **agent_pattern: I'?ll\s+use\s+the\s+\w+\s+skill:** 1 (0.3% of removed)


## Patterns Detected

The following regex patterns were used to identify agent-like responses:

- `I'?ll\s+(start|begin|first|now|continue)\s+(by|with|the)`
- `Let me\s+(start|begin|first|read|check|analyze|examine|look|see|understand|pick up)`
- `First,?\s+I'?ll`
- `I need to\s+(first|start|begin|read|check|analyze)`
- `Vou\s+(implementar|analisar|começar|primeiro|ler|verificar|examinar)`
- `Deixe-me\s+(ler|analisar|começar|verificar|primeiro)`
- `Primeiro,?\s+(vou|deixe-me)`
- `Preciso\s+(primeiro|começar|ler|analisar)`
- `I'?ll\s+(read|check|examine|analyze|look at|review|continue|pick up)\s+`
- `Let's\s+(start|begin)\s+(by|with)`
- `Here'?s\s+(what|how)\s+I'?ll`
- `My\s+plan\s+is\s+to`
- `I'?m\s+going\s+to\s+(start|begin|first)`
- `I'?ll\s+continue\s+(with|the|where)`
- `picking up where`
- `^Excellent!?\s*$`
- `^Perfect!?\s*$`
- `^Great!?\s*$`
- `by\s+invoking\s+the\s+relevant\s+skills?`
- `I'?ll\s+use\s+the\s+\w+\s+skill`
- `vou\s+usar\s+o\s+skill`


## Quality Criteria

**Kept examples must:**
- ✅ Have responses ≥50 characters
- ✅ NOT contain agent-like planning language
- ✅ NOT be tool-only invocations (heuristic: <20 words + >5 newlines)

## Next Steps

1. **Review curated dataset** - Spot check samples to verify quality
2. **Retrain model** - Use dataset_sharegpt_curated.json for QLoRA fine-tuning
3. **Evaluate personality** - Compare base vs curated-trained model
4. **Iterate if needed** - Adjust patterns or thresholds based on results

## Files Generated

- `dataset_sharegpt_curated.json` - Curated dataset (153 examples)
- `CURATION_REPORT.md` - This report

---

**Status:** ✅ Curation complete. Dataset ready for retraining.
