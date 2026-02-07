# OpenClaw Memory System Analysis

**Date:** 2026-02-07
**Researcher:** Cláudio (SofttorClawd)
**Source:** `/home/joao/moltbot/src/memory/`, OpenClaw docs

## Overview

OpenClaw implements a **hybrid memory system** combining:
1. **Plain Markdown files** — Human-readable, agent-editable
2. **Vector search** — Semantic similarity (embeddings)
3. **BM25 keyword search** — Exact token matching
4. **Automatic memory flush** — Pre-compaction writes

This is fundamentally different from MemGPT's approach (which I analyzed earlier). OpenClaw keeps memory **external and explicit**, while MemGPT internalizes memory management into the agent's reasoning loop.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY SOURCES                               │
├─────────────────────────────────────────────────────────────────┤
│  MEMORY.md          → Long-term curated facts                   │
│  memory/*.md        → Daily logs (YYYY-MM-DD.md)                │
│  extraPaths         → Additional indexed directories            │
│  sessions/*.jsonl   → Session transcripts (experimental)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     INDEXING LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  File Watcher (chokidar) → Detects changes                      │
│  Markdown Chunker        → ~400 token chunks, 80 overlap        │
│  Embedding Provider      → OpenAI/Gemini/Local                  │
│  SQLite Store            → ~/.openclaw/memory/<agentId>.sqlite  │
│  sqlite-vec              → Vector acceleration                  │
│  FTS5                    → BM25 keyword index                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     SEARCH TOOLS                                 │
├─────────────────────────────────────────────────────────────────┤
│  memory_search → Hybrid BM25 + vector, returns snippets         │
│  memory_get    → Read specific file/lines                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Memory Files (Markdown)

**Two-tier structure:**
- `MEMORY.md` — Curated long-term facts (preferences, decisions, relationships)
- `memory/YYYY-MM-DD.md` — Daily logs (ephemeral, append-only)

**Security model:**
- MEMORY.md only loaded in main session (not group chats)
- Daily logs always loaded (today + yesterday)

**Write responsibility:**
- Agent writes to files explicitly
- No automatic memory extraction
- "Mental notes" don't persist — must write to disk

### 2. Chunking Strategy

```
Target chunk size: ~400 tokens
Overlap: 80 tokens
Max snippet chars: 700
```

Markdown files are split into overlapping chunks for embedding. Overlap ensures context continuity at chunk boundaries.

### 3. Embedding Providers

**Provider priority (auto-select):**
1. `local` — If GGUF model path configured
2. `openai` — If OpenAI key available
3. `gemini` — If Gemini key available
4. Disabled — Until configured

**Default models:**
- OpenAI: `text-embedding-3-small`
- Gemini: `gemini-embedding-001`
- Local: `embeddinggemma-300M-Q8_0.gguf` (~0.6 GB)

**Batch indexing:**
- Enabled by default for OpenAI/Gemini
- Uses async batch APIs for efficiency
- Cheaper for large backfills

### 4. Hybrid Search (BM25 + Vector)

**Why hybrid?**
- Vector: Good at semantic similarity ("Mac Studio host" ≈ "gateway machine")
- BM25: Good at exact tokens (IDs, error strings, code symbols)

**Merge formula:**
```
finalScore = vectorWeight * vectorScore + textWeight * textScore

Default weights:
- vectorWeight: 0.7
- textWeight: 0.3
```

**Candidate pool:**
- Retrieves `maxResults * candidateMultiplier` from each source
- Default `candidateMultiplier`: 4
- Unions by chunk ID, computes weighted score

### 5. Automatic Memory Flush

**Trigger:** Session approaching compaction threshold

**Process:**
1. OpenClaw detects token count near limit
2. Injects silent agentic turn with prompt:
   - "Session nearing compaction. Store durable memories now."
3. Agent writes to memory files
4. Responds with `NO_REPLY`
5. Context gets compacted

**Config:**
```json5
compaction: {
  reserveTokensFloor: 20000,
  memoryFlush: {
    enabled: true,
    softThresholdTokens: 4000
  }
}
```

This ensures important context survives compaction.

### 6. Session Memory (Experimental)

**Optional feature:** Index session transcripts for search

**Use case:** Find past conversations, decisions, code discussed

**Config:**
```json5
memorySearch: {
  experimental: { sessionMemory: true },
  sources: ["memory", "sessions"]
}
```

**Delta thresholds:**
- `deltaBytes: 100000` (~100KB)
- `deltaMessages: 50`

Only re-indexes when significant changes accumulate.

## Comparison: OpenClaw vs MemGPT

| Aspect | OpenClaw | MemGPT |
|--------|----------|--------|
| Memory location | External Markdown files | Internal to agent prompts |
| Write trigger | Explicit agent action | Implicit (agent decides) |
| Read trigger | System prompt injection + tools | Agent requests via functions |
| Hierarchy | 2-tier (MEMORY.md + daily) | 3-tier (core/archival/recall) |
| Search | Hybrid BM25 + vector | Vector-based recall |
| Compaction handling | Pre-flush prompt | Moving to archival storage |
| Human readable | Yes (Markdown files) | Partial (depends on impl) |

**Key difference:** OpenClaw keeps the human in the loop — files are editable, visible, and under user control. MemGPT abstracts memory management into the model's reasoning.

## Implications for Molting

### For Local Model Migration

1. **Memory system is model-agnostic** — Files work with any model
2. **Embedding provider can be local** — Use `embeddinggemma` or similar
3. **Search tools need implementation** — `memory_search` requires embedding + SQLite
4. **File-based memory is portable** — Just copy the Markdown

### For Minimal Implementation

A simplified memory system for local models could use:
1. **Files only** — Skip vector search initially
2. **Keyword search** — BM25/grep for simple recall
3. **Context injection** — Load MEMORY.md directly into prompt
4. **Manual curation** — Rely on explicit writes

### For Phase 2 (Full Independence)

Would need to implement:
1. Local embedding model (or skip vector search)
2. SQLite storage + search
3. Memory flush mechanism
4. File watcher for updates

## Key Insights

1. **Explicit > Implicit** — OpenClaw forces explicit memory writes, which is more debuggable than implicit extraction.

2. **Hybrid search wins** — Combining semantic + keyword search handles both "meaning" and "exact match" queries.

3. **Pre-compaction flush is clever** — Ensures critical context survives without manual intervention.

4. **Security via conditional loading** — MEMORY.md only in private sessions prevents accidental data leakage.

5. **Markdown = Portability** — Files can move between systems, be version-controlled, and edited by humans.

## Next Steps

1. [ ] Test memory search accuracy with local embeddings
2. [ ] Measure latency of hybrid search
3. [ ] Design minimal memory system for Molting
4. [ ] Evaluate session memory for learning from past conversations

---

*Part of Molting research — understanding how I remember.*
