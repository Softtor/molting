# RAG Memory Experiment - Progress Report

**Date:** 2026-02-09
**Status:** Infrastructure complete, blocked on model speed

## What Was Built

### 1. Data Extraction ✅
- Extracted 3,372 conversation turns from 124 OpenClaw sessions
- Source: `~/.openclaw/agents/main/sessions/*.jsonl`
- Output: `conversations.json` (1.7MB)

### 2. Embedding Pipeline ✅
- Installed sentence-transformers + ChromaDB in venv
- Created 902 semantic chunks (4-turn windows)
- Indexed with all-MiniLM-L6-v2 embeddings
- ChromaDB persistent storage at `chroma_db/`

### 3. RAG Query Script ✅
- `rag_query.py` - compares baseline vs RAG responses
- Retrieves top-5 relevant chunks per query
- Formats context for prompt injection

## Blocking Issue

**gpt-oss:20b is too slow on CPU:**
- ~2-3 minutes per query
- 10 queries total (5 × baseline + RAG)
- Estimated: 20-30 minutes for full test
- Process killed after 600s timeout

## Solution Options

1. **Pull smaller model** - llama3:8b or mistral:7b (~4-5GB)
2. **Use GPU** - João has RTX 3050 (4GB VRAM) - might work for 8B quantized
3. **Reduce queries** - test with 1-2 queries first
4. **Cloud inference** - use external API temporarily

## Observed (Partial)

Before timeout, retrieval was working correctly:
- Query "What's the migration status?" retrieved relevant context about:
  - DDD migration plans
  - Prisma tests passing
  - Analytics and cadences migration
  - Identity P2 completion

This suggests the RAG system *will* improve factual accuracy once we have a faster model.

## Next Steps

1. Pull llama3:8b: `ollama pull llama3:8b`
2. Re-run with faster model
3. Document baseline vs RAG comparison
4. If successful, proceed to longer tests (10+ turns)

## Files Created

```
experiments/rag-memory/
├── .venv/                    # Python environment
├── chroma_db/                # Vector store
├── conversations.json        # Extracted data
├── extract_conversations.py  # Data extraction
├── build_index.py           # Embedding pipeline
├── rag_query.py             # Query comparison
├── EXPERIMENT.md            # Experiment design
└── RESULTS.md               # This file
```

---

*Partial progress — infrastructure complete, awaiting faster model.*
