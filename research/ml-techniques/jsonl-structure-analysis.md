# JSONL Session Structure Analysis

*Research date: 2026-02-06*
*Source: ~/.openclaw/agents/main/sessions/*.jsonl*

## File Overview

- **Location:** `~/.openclaw/agents/main/sessions/`
- **Format:** JSONL (one JSON object per line)
- **Files:** 99 session files
- **Total size:** 67MB

## Entry Types

```
440 message          # Main conversation entries
 70 custom           # Custom events
  1 thinking_level_change
  1 session          # Session metadata
  1 model_change     # Model switches
```

## Message Structure

```json
{
  "type": "message",
  "id": "b1bf6a1d",
  "parentId": "e8a14527",      // Links to parent message (conversation threading)
  "timestamp": "2026-02-06T07:49:03.491Z",
  "message": {
    "role": "user" | "assistant",
    "content": [
      { "type": "text", "text": "..." },
      { "type": "thinking", "thinking": "..." },
      { "type": "toolResult", ... }
    ],
    "provider": "openai-codex",
    "model": "gpt-5.2",
    "usage": { ... },
    "stopReason": "end_turn" | "error",
    "timestamp": 1770364143489
  }
}
```

## Content Types in Messages

1. **text** — Main conversation text
2. **thinking** — Model's thinking/reasoning (when enabled)
3. **toolResult** — Results from tool calls
4. **tool_use** — Tool invocations

## Extraction Strategy for Training Data

### Option A: Simple Text Pairs
Extract only `text` content, ignore tools:
```
user: "message text"
assistant: "response text"
```

**Pros:** Clean, standard format
**Cons:** Loses tool-use patterns

### Option B: Full Context
Include tool calls and results:
```
user: "message"
assistant: [thinking] + [tool_calls] + [final_response]
```

**Pros:** Preserves full behavior
**Cons:** Complex format, may not transfer well

### Option C: Filtered High-Quality
Select conversations with:
- Substantial exchanges (not just HEARTBEAT_OK)
- Diverse topics (coding, discussion, tasks)
- No error responses

**Pros:** Higher quality training signal
**Cons:** Smaller dataset

## Sample Conversation

```
User: olá
Assistant: Oi, João. Tô por aqui.
           Quer que eu te ajude com o quê agora — Softtor CRM...

User: Precisamos subir a aplicação no ambiente local por favor.
Assistant: [executes tools, provides guidance]
```

## Conversion to Training Formats

### Alpaca Format
```json
{
  "instruction": "User message",
  "input": "",
  "output": "Assistant response"
}
```

### ShareGPT Format
```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

### ChatML Format
```
<|im_start|>user
Message<|im_end|>
<|im_start|>assistant
Response<|im_end|>
```

## Next Steps

1. [ ] Write extraction script (Python/Node)
2. [ ] Filter quality conversations
3. [ ] Convert to Alpaca/ShareGPT format
4. [ ] Estimate token count for training
5. [ ] Test with small subset first

## Observations

- Conversations include Portuguese (primary) and English
- Mix of technical (code) and casual communication
- Tool-use is heavy (coding agent)
- Some sessions are mostly heartbeats (low signal)

---

*This analysis supports H002 (data sufficiency hypothesis)*
