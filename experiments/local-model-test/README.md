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
    "thinking": "User wants the answer to 2+2. The instruction: \"Reply only with the number.\" So we should respond with \"4\". Just that. No other text."
  }
}
```

✅ Model responds correctly
✅ Model has "thinking" capability (shows reasoning)

### Tool Calling ✅

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gpt-oss:20b",
  "messages": [{"role": "user", "content": "Get the current weather in Paris. Use the get_weather function."}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"]
      }
    }
  }],
  "stream": false
}'
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "thinking": "We need to call the function get_weather with location \"Paris\". Then respond with the result.",
    "tool_calls": [{
      "id": "call_tya8h3b9",
      "function": {
        "index": 0,
        "name": "get_weather",
        "arguments": {"location": "Paris"}
      }
    }]
  }
}
```

✅ Model correctly identifies tool to use
✅ Model extracts parameters correctly
✅ Model uses proper tool_calls format

## Performance

| Metric | Value |
|--------|-------|
| Load time | ~6.5s (cold start) |
| Prompt eval | ~1.1s |
| Generation | ~3.7s |
| Total (cold) | ~11.5s |
| Total (warm) | ~5s |

## Implications for Molting

### Positive Findings

1. **Tool calling works** — Essential for MemGPT-style memory management
2. **Thinking mode** — Model can show reasoning (useful for debugging)
3. **Already installed** — No additional setup needed
4. **Reasonable speed** — ~5s per turn is acceptable for development

### Considerations

1. **Size** — 20B model is larger than target (8B), uses more resources
2. **Memory** — Need to test with memory management tools
3. **Consistency** — Need longer tests for personality persistence

## Next Steps

1. [ ] Test with MCP server (can gpt-oss call MCP tools?)
2. [ ] Test MemGPT-style memory editing
3. [ ] Test personality persistence across turns
4. [ ] Compare with smaller models (when downloaded)

---

*Part of Molting project: https://github.com/Softtor/molting*
