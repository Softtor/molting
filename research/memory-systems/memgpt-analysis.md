# MemGPT Architecture Analysis

*Research date: 2026-02-07*
*Source: https://arxiv.org/abs/2310.08560, https://docs.letta.com*

## Overview

MemGPT (Memory GPT) is a research paper and agent architecture that treats LLMs like operating systems with hierarchical memory management. Key insight: **LLMs can be taught to manage their own memory**.

**Paper:** "MemGPT: Towards LLMs as Operating Systems" (2023)
- Authors: Charles Packer, Sarah Wooders, Kevin Lin, et al.
- Now productized as [Letta](https://www.letta.com/)

## Core Concepts

### 1. Virtual Context Management

Inspired by virtual memory in operating systems:
- Physical memory (RAM) ↔ **In-context memory** (context window)
- Disk storage ↔ **Out-of-context memory** (databases, files)
- Page faults ↔ **Memory retrieval tools**

The LLM sees a "virtual context" larger than its actual context window.

### 2. Memory Hierarchy

```
┌─────────────────────────────────────────────────┐
│              CONTEXT WINDOW                      │
│  ┌─────────────────────────────────────────┐    │
│  │  System Prompt (fixed, not editable)    │    │
│  ├─────────────────────────────────────────┤    │
│  │  In-Context Memory (self-editable)      │    │
│  │  ├── Persona (agent's identity)         │    │
│  │  └── Human (user information)           │    │
│  ├─────────────────────────────────────────┤    │
│  │  Recent Messages                        │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
                      │
          ┌──────────┴──────────┐
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Archival Memory │    │  Recall Memory  │
│ (Vector DB)     │    │  (Chat History) │
│ - Long-term     │    │  - All messages │
│ - External data │    │  - Searchable   │
└─────────────────┘    └─────────────────┘
```

### 3. Self-Editing Memory via Tool Calling

The agent manages its own memory through tools:

**In-Context Memory Tools:**
- `memory_insert` — Add new information
- `memory_replace` — Update existing information
- `memory_rethink` — Reflect and revise
- `memory_finish_edits` — Complete editing

**External Memory Tools:**
- `archival_memory_search` — Search long-term memory
- `archival_memory_insert` — Store to long-term memory
- `conversation_search` — Search chat history
- `conversation_search_date` — Search by date

### 4. Heartbeats for Multi-Step Reasoning

When the LLM outputs a tool call, it can request a "heartbeat":
- `request_heartbeat: true` → LLM gets another turn
- Enables chains of thought without user input
- Similar to ReAct pattern but self-directed

## ChatMemory Implementation

Default memory class for 1:1 chat:

```python
class ChatMemory(BaseMemory):
    def __init__(self, persona: str, human: str, limit: int = 2000):
        self.memory = {
            "persona": MemoryModule(name="persona", value=persona, limit=limit),
            "human": MemoryModule(name="human", value=human, limit=limit),
        }
    
    def core_memory_append(self, name: str, content: str):
        """Append to persona or human section"""
        self.memory[name].value += "\n" + content
    
    def core_memory_replace(self, name: str, old_content: str, new_content: str):
        """Replace exact text in persona or human section"""
        self.memory[name].value = self.memory[name].value.replace(old_content, new_content)
```

## Comparison with OpenClaw

| Aspect | MemGPT | OpenClaw |
|--------|--------|----------|
| In-context memory | Persona + Human blocks | SOUL.md + USER.md + MEMORY.md |
| External memory | Vector DB (archival) | Markdown files + memory_search |
| Self-editing | Tool calls modify memory | Agent can write to files |
| Heartbeats | request_heartbeat flag | HEARTBEAT.md + cron |
| History | Recall memory table | Session logs (JSONL) |
| Architecture | LLM as OS | Agent framework |

### Key Similarity
Both use **self-editing memory** — the agent can modify its own context.

### Key Difference
- MemGPT: Structured memory blocks with strict limits (2k chars)
- OpenClaw: Freeform markdown files with semantic search

## Implications for Molting

### 1. Memory Architecture Validation

MemGPT's success validates our approach:
- Persona/Human split ≈ SOUL.md/USER.md
- Self-editing works (agent can update its own memory)
- Hierarchical memory is effective

### 2. Potential Improvements

Could adopt from MemGPT:
- **Strict character limits** — Force compression/summarization
- **Separate recall memory** — Structured chat history search
- **Heartbeat pattern** — Explicit multi-step reasoning

### 3. Local Model Considerations

MemGPT requires:
- Tool calling capability
- Ability to follow memory editing instructions
- Understanding of memory schema

For Molting's local model goal:
- Need model that supports tool/function calling
- Memory editing prompts must be simple enough for smaller models
- May need to simplify memory schema for 7B-8B models

## Hypothesis Connection

**H001 (Personality from files):**
MemGPT's "persona" block is similar to our SOUL.md — both inject personality via context, not weights.

**H003 (MCP abstraction):**
MemGPT tools could be exposed via MCP:
- `archival_memory_search` → MCP tool
- `core_memory_replace` → MCP tool
- Memory blocks → MCP resources

## Questions to Explore

1. [ ] What's the minimum model size for effective self-editing memory?
2. [ ] Can MemGPT tools be simplified for smaller models?
3. [ ] How does memory compression work at scale?
4. [ ] Letta's approach to multi-agent memory sharing?

## Next Steps

1. [ ] Test Letta locally with different model sizes
2. [ ] Compare memory editing success rates across models
3. [ ] Prototype simplified memory schema for Molting
4. [ ] Explore vector DB vs markdown for archival memory

## Resources

- [MemGPT Paper](https://arxiv.org/abs/2310.08560)
- [Letta Documentation](https://docs.letta.com/)
- [Letta GitHub](https://github.com/letta-ai/letta)
- [MemGPT Research Site](https://research.memgpt.ai/)

---

*This analysis is part of the Molting project: https://github.com/Softtor/molting*
