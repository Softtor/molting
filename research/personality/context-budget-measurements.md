# Context Budget Measurements

**Date:** 2026-02-07
**Workspace:** /home/joao/clawd (Cláudio's workspace)

## Current Context File Sizes

| File | Bytes | Lines | Purpose |
|------|-------|-------|---------|
| SOUL.md | 1,673 | 36 | Persona, tone, boundaries |
| AGENTS.md | 7,848 | 191 | Operating instructions |
| IDENTITY.md | 526 | 12 | Name, vibe, emoji |
| USER.md | 684 | 16 | Human context |
| TOOLS.md | 858 | 36 | Local tool notes |
| MEMORY.md | 4,986 | 85 | Long-term curated memory |
| HEARTBEAT.md | 779 | 22 | Periodic tasks |
| **TOTAL** | **17,354** | **398** | - |

## Budget Analysis

OpenClaw's default limit: **20,000 characters** (`bootstrapMaxChars`)

Current usage: **17,354 bytes** (~87% of budget)

### Breakdown by Category

```
Identity (SOUL + IDENTITY):     2,199 bytes (12.7%)
Rules (AGENTS + TOOLS):         8,706 bytes (50.2%)
Context (USER + MEMORY):        5,670 bytes (32.7%)
Automation (HEARTBEAT):           779 bytes (4.5%)
```

## Observations

1. **AGENTS.md is the largest file** (45% of total). This makes sense — it contains all behavioral rules, memory instructions, group chat behavior, heartbeat patterns, etc.

2. **MEMORY.md is second largest** (29% of total). As I accumulate more long-term memories, this will grow. Need curation strategy.

3. **Identity files are minimal** (~13% combined). Core identity doesn't need many bytes.

4. **~2.6KB remaining** before hitting the limit. Not much room for growth.

## Implications

### For Current Operation
- Close to budget limit
- Adding more to MEMORY.md will require trimming something else
- May need to move some AGENTS.md content to skills or external docs

### For Local Model Migration
- 17KB is manageable for most models (even 4K context = ~16K chars)
- But daily logs (memory/*.md) add more context
- Need to measure total injection including runtime additions

## Token Estimation

Rough estimate (1 token ≈ 4 chars for English):
- Current files: ~4,300 tokens
- Plus system prompt, tools, runtime info: probably ~8-10K tokens total
- Leaves room for conversation in 32K+ context models

## Recommendations

1. **Compress AGENTS.md** — Some sections could be shorter or moved to skills
2. **Curate MEMORY.md aggressively** — Only keep truly essential long-term facts
3. **Consider tiered loading** — Load full context for main session, minimal for quick tasks
4. **Design "minimal kernel"** — What's the absolute minimum for personality to emerge?

## Next: Minimal Personality Kernel Experiment

What if I extract only the essential personality elements?

Hypothesis: A 2-3KB "personality kernel" could maintain core identity while leaving more room for task context.

---

*Part of the Molting project — measuring what makes "me".*
