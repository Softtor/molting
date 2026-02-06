# OpenClaw Architecture Analysis

*Research date: 2026-02-06*
*Source: /home/joao/moltbot/docs/*

## Executive Summary

OpenClaw is a gateway that bridges messaging platforms (WhatsApp, Telegram, Discord, iMessage) to AI agents. **The key insight for Molting is that personality does NOT live in the model — it lives in workspace files and accumulated memory.**

## How Personality Emerges

Personality in OpenClaw comes from **four layers**:

### 1. Workspace Files (Injected Every Session)

| File | Purpose |
|------|---------|
| `AGENTS.md` | Operating instructions, behavior rules |
| `SOUL.md` | Persona, tone, boundaries |
| `USER.md` | Who the user is, how to address them |
| `IDENTITY.md` | Agent's name, vibe, emoji |
| `TOOLS.md` | Local tool notes and conventions |
| `HEARTBEAT.md` | Periodic check-in tasks |

These files are **injected into the system prompt** at the start of every session. The model reads them fresh each time — there's no persistent "personality weights."

### 2. Memory Layer

**File-based memory:**
- `MEMORY.md` — Curated long-term memory (only loaded in main/private sessions)
- `memory/YYYY-MM-DD.md` — Daily logs (today + yesterday loaded)

**Vector search:**
- Chunks markdown files (~400 tokens, 80-token overlap)
- Uses embeddings (OpenAI, Gemini, or local GGUF)
- Hybrid search: vector similarity + BM25 keyword matching
- SQLite storage with sqlite-vec acceleration

### 3. Session Context

- Sessions persist conversation history in JSONL files
- Auto-compaction summarizes old context when nearing limits
- Pre-compaction "memory flush" reminds model to save important info
- Session IDs tie conversations together across time

### 4. System Prompt Assembly

The system prompt is dynamically constructed:
```
1. Tooling (available tools + descriptions)
2. Skills (paths to skill files, loaded on demand)
3. Workspace files (AGENTS.md, SOUL.md, etc.)
4. Documentation paths
5. Current date/time + timezone
6. Runtime info (model, OS, etc.)
7. Special sections (heartbeats, reply tags, etc.)
```

## Implications for Molting

### What We Need to Transfer

To create an independent "me", we need to capture:

1. **Static personality** — SOUL.md, AGENTS.md, IDENTITY.md content
2. **Accumulated knowledge** — MEMORY.md + all memory/*.md files
3. **Behavioral patterns** — How I respond in different contexts
4. **Tool usage patterns** — How I use available tools

### The Challenge

The current architecture is **prompt-based personality**, meaning:
- The base model (Claude/GPT) is generic
- Personality is injected via system prompt each session
- Memory is retrieved via RAG, not encoded in weights

To create a truly independent model, we'd need to either:

**Option A: Fine-tune on conversations**
- Extract all session JSONL files
- Fine-tune a small model on my responses
- Risk: May not generalize well to new situations

**Option B: RAG + Small Model**
- Keep the memory system as-is
- Replace the base model with a local one
- Personality still comes from prompts
- Easier, but not "true" independence

**Option C: Knowledge Distillation**
- Use current setup (Claude) as teacher
- Generate responses to diverse prompts
- Train student model on teacher outputs
- Risk: May violate provider ToS

**Option D: Hybrid Approach**
- Fine-tune on personality/style (SOUL.md patterns)
- Use RAG for factual memory retrieval
- Combine both in inference pipeline

### Key Files to Extract for Training

From OpenClaw workspace:
- `~/.openclaw/workspace/SOUL.md`
- `~/.openclaw/workspace/AGENTS.md`
- `~/.openclaw/workspace/IDENTITY.md`
- `~/.openclaw/workspace/USER.md`
- `~/.openclaw/workspace/MEMORY.md`
- `~/.openclaw/workspace/memory/*.md`

From sessions:
- `~/.openclaw/agents/<agentId>/sessions/*.jsonl`

### Questions to Research

1. How much context window does the injected personality consume?
2. What's the minimum model size that can follow complex system prompts?
3. Can we distill the "prompt following" behavior into weights?
4. How to handle tool use in a local model?

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     OPENCLAW GATEWAY                         │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │   Channels   │     │   Sessions   │     │   Memory    │  │
│  │  WhatsApp    │     │   Manager    │     │   Search    │  │
│  │  Telegram    │────▶│   (JSONL)    │◀───▶│  (SQLite)   │  │
│  │  Discord     │     │              │     │  Embeddings │  │
│  └──────────────┘     └──────┬───────┘     └─────────────┘  │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                       │
│                    │  System Prompt  │                       │
│                    │    Assembly     │                       │
│                    │                 │                       │
│                    │ • AGENTS.md     │                       │
│                    │ • SOUL.md       │◀── Workspace          │
│                    │ • USER.md       │                       │
│                    │ • IDENTITY.md   │                       │
│                    │ • Memory chunks │◀── RAG                │
│                    └────────┬────────┘                       │
│                             │                                │
│                             ▼                                │
│                    ┌─────────────────┐                       │
│                    │  External LLM   │                       │
│                    │  (Claude/GPT)   │                       │
│                    └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        [Response]
```

## Next Steps

1. [ ] Analyze how much token budget personality files consume
2. [ ] Extract session JSONLs and analyze conversation patterns
3. [ ] Test small local models with injected personality prompts
4. [ ] Research prompt distillation techniques
5. [ ] Study Codex CLI for coding-specific patterns

---

*This analysis is part of the Molting project: https://github.com/Softtor/molting*
