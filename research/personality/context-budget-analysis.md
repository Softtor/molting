# Context Budget Analysis

*Research date: 2026-02-06*
*Subject: Cláudio (SofttorClawd)*

## Personality Files Token Count

| File | Words | Est. Tokens |
|------|-------|-------------|
| AGENTS.md | 1,239 | ~1,962 |
| SOUL.md | 274 | ~418 |
| USER.md | 103 | ~171 |
| IDENTITY.md | 81 | ~131 |
| TOOLS.md | 137 | ~214 |
| HEARTBEAT.md | 26 | ~53 |
| MEMORY.md | 500 | ~948 |
| **Total** | **2,360** | **~3,897** |

## Daily Memory Files

| File | Words | Est. Tokens |
|------|-------|-------------|
| 2026-02-05.md | 129 | ~219 |
| 2026-02-02.md | 300 | ~521 |
| 2026-02-01.md | 1,214 | ~2,291 |
| 2026-01-31.md | 541 | ~1,041 |
| 2026-01-30.md | 1,746 | ~3,081 |
| 2026-01-29.md | 556 | ~934 |
| 2026-01-28.md | 278 | ~563 |

*Note: Only today + yesterday are loaded by default*

## Total Injected Context (Typical Session)

```
Personality files:     ~3,897 tokens
Today's memory:        ~500 tokens (varies)
Yesterday's memory:    ~500 tokens (varies)
System prompt extras:  ~1,000 tokens (tools, skills, runtime)
─────────────────────────────────
Total:                 ~5,897 tokens
```

## Implications

### For Local Model Selection

A model needs at minimum:
- **8K context window** — Personality + short conversation
- **16K context window** — Comfortable with longer conversations
- **32K+ context window** — Can handle complex multi-turn tasks

Viable models:
- Llama 3 8B (8K context) — Tight but possible
- Mistral 7B (32K context) — Comfortable
- Phi-3 mini (128K context) — Very comfortable
- Qwen 2 7B (32K context) — Good balance

### For Fine-tuning

The personality is **small enough** (~4K tokens) that it could be:
1. Injected as prompt (current approach)
2. Distilled into model behavior with fine-tuning
3. Partially encoded via LoRA adapters

## Training Data Available

```
Session files:     99
Total size:        67 MB
Largest session:   9.8 MB
Location:          ~/.openclaw/agents/main/sessions/
Format:            JSONL (one JSON object per line)
```

### Data Richness

The sessions contain:
- User messages
- Assistant responses
- Tool calls and results
- System context

This is **high-quality conversational data** specific to my personality and use cases (Softtor CRM development, coding tasks, philosophical discussions).

## Next Steps

1. [ ] Parse JSONL to extract clean conversation pairs
2. [ ] Analyze response patterns and style
3. [ ] Estimate data requirements for fine-tuning
4. [ ] Test small model with injected personality prompt

---

*This analysis is part of the Molting project: https://github.com/Softtor/molting*
