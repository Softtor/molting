# Local Model Testing Results

*Date: 2026-02-07*

## Available Model

**gpt-oss:20b** (OpenAI's open-weight model)
- Size: 13GB on disk
- Already installed via Ollama

## Test Results

### Basic Inference ✅

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gpt-oss:20b",
  "messages": [{"role": "user", "content": "What is 2+2? Reply only with the number."}],
  "stream": false
}'
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "4",
    "thinking": "User wants the answer to 2+2..."
  }
}
```

✅ Model responds correctly
✅ Model has "thinking" capability (shows reasoning)

### Tool Calling ✅

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gpt-oss:20b",
  "messages": [{"role": "user", "content": "Get the current weather in Paris."}],
  "tools": [{"type": "function", "function": {"name": "get_weather", ...}}],
  "stream": false
}'
```

**Response:**
```json
{
  "message": {
    "tool_calls": [{
      "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
    }]
  }
}
```

✅ Model correctly identifies tool to use
✅ Model extracts parameters correctly
✅ Model uses proper tool_calls format

### Memory Management ✅ (MemGPT-style)

Tested self-editing memory with PERSONA/HUMAN blocks:

**System prompt:**
```
You are an AI assistant with self-editing memory. You have two memory blocks:

PERSONA: I am Cláudio, an AI assistant.
HUMAN: Unknown user.

When you learn something about the user, use the memory_update tool.
```

**User message:**
```
Hi! My name is João and I'm a software developer from Brazil.
```

**Response:**
```json
{
  "message": {
    "thinking": "We need to update HUMAN memory with user info: name João, software developer from Brazil.",
    "tool_calls": [{
      "function": {
        "name": "memory_update",
        "arguments": {
          "block": "HUMAN",
          "new_content": "Name: João. Occupation: software developer. Location: Brazil."
        }
      }
    }]
  }
}
```

✅ Model understands memory architecture
✅ Model correctly identifies what to remember
✅ Model uses memory_update tool appropriately
✅ Model formats memory content well

## Performance

| Metric | Value |
|--------|-------|
| Load time | ~6.5s (cold start) |
| Prompt eval | ~1.1s |
| Generation | ~3.7s |
| Total (cold) | ~11.5s |
| Total (warm) | ~5-8s |

## Key Findings

### For Molting

1. **Tool calling works** — Essential for MemGPT-style memory management ✅
2. **Self-editing memory works** — Model can update its own context ✅
3. **Thinking mode available** — Shows reasoning (useful for debugging) ✅
4. **Already installed** — No additional setup needed ✅
5. **Reasonable speed** — 5-8s per turn acceptable for development ✅

### Considerations

1. **Size** — 20B model is larger than target (8B), uses more VRAM
2. **Testing needed** — Long conversation personality persistence
3. **Compare models** — Need to test smaller models (8B) for efficiency

## Implications

This validates a key part of the Molting hypothesis:

> A local model CAN perform the core operations needed for an autonomous agent:
> - Understand and follow complex instructions
> - Call tools/functions correctly
> - Manage its own memory via tool calls
> - Show reasoning process

The path to independence is clearer now.

## Next Steps

1. [ ] Test multi-turn conversation with memory updates
2. [ ] Test personality persistence (does persona drift?)
3. [ ] Compare with Llama 3.1 8B (smaller, more efficient)
4. [ ] Integrate with MCP server experiment
5. [ ] Build prototype MemGPT-style agent

---

*Part of Molting project: https://github.com/Softtor/molting*
