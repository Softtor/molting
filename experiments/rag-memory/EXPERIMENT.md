# RAG Memory Experiment

**Date:** 2026-02-09
**Objective:** Test if a local model can use retrieved conversation history to provide accurate, contextual responses.

## Hypothesis

**H005:** A RAG system that retrieves from my conversation history can augment a local model's responses, improving factual accuracy about my work and preferences.

### Predictions
1. Without RAG: Local model will hallucinate or give generic answers about my projects
2. With RAG: Local model will give accurate, specific answers based on retrieved context
3. Retrieval quality correlates with response quality

## Setup

### Data Source
- OpenClaw session logs: `~/.openclaw/agents/main/sessions/*.jsonl`
- ~155 sessions, 72MB of conversation data
- Contains: user messages, assistant responses, tool calls, timestamps

### Components
1. **Extractor** — Parse JSONL, extract conversation turns
2. **Chunker** — Split into semantic chunks (by message, by topic)
3. **Embedder** — Generate embeddings (sentence-transformers)
4. **Store** — Vector storage (ChromaDB or FAISS)
5. **Retriever** — Find relevant chunks for a query
6. **Generator** — Local model (gpt-oss:20b) with retrieved context

### Test Queries
1. "What CRM project am I working on?" (Softtor)
2. "What's the migration status?" (saas-crm + omnichannel merge)
3. "What did I work on yesterday?" (temporal reasoning)
4. "What's my preferred coding style?" (preferences)
5. "Tell me about the Molting project" (self-reference)

## Implementation Plan

### Step 1: Extract Conversation Data
```python
# Parse JSONL files, extract user/assistant turns
# Filter out tool calls, system messages
# Output: conversations.json with structured turns
```

### Step 2: Create Embeddings
```python
# Use sentence-transformers (all-MiniLM-L6-v2)
# Chunk by message or fixed size
# Store in ChromaDB for persistence
```

### Step 3: Build Retrieval Pipeline
```python
# Query → Embed → Retrieve top-k → Format context → Generate
```

### Step 4: Compare Results
- Baseline: Local model without RAG
- Treatment: Local model with RAG
- Measure: Factual accuracy, relevance, hallucination rate

## Success Criteria

1. **Accuracy:** ≥80% of retrieved facts are correct
2. **Relevance:** Retrieved context matches query intent
3. **Integration:** Local model uses retrieved context naturally
4. **No hallucination:** With RAG, model doesn't contradict retrieved facts

## Notes

This experiment directly supports Phase 2 goal: "Build RAG system with my conversation history"

---

*Part of Molting research — building memory persistence layer*
