# Molting MCP Server Experiment

**Status:** ✅ Success  
**Date:** 2026-02-07  
**Hypothesis:** H003 - MCP provides sufficient abstraction for model-agnostic agents

## Purpose

This minimal MCP server tests whether the Model Context Protocol can serve as an abstraction layer for agent capabilities, independent of the underlying model.

## What It Does

Exposes three tools and one resource:

### Tools

| Tool | Description |
|------|-------------|
| `get_identity` | Returns agent identity (name, vibe, project) |
| `get_memory` | Returns recent memory entries |
| `echo` | Simple connectivity test |

### Resources

| Resource | URI | Description |
|----------|-----|-------------|
| Personality | `molting://personality` | Agent's personality as markdown |

## Test Results

### Initialize
```json
Request: {"method":"initialize","params":{"protocolVersion":"2024-11-05",...}}
Response: {"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{"listChanged":true},"resources":{"listChanged":true}},"serverInfo":{"name":"molting-mcp-server","version":"0.1.0"}}}
```

### List Tools
```json
Request: {"method":"tools/list","params":{}}
Response: Lists all 3 tools with correct schemas ✅
```

### Call get_identity
```json
Request: {"method":"tools/call","params":{"name":"get_identity","arguments":{}}}
Response: {"name":"Cláudio","creature":"AI assistant...","emoji":"🦞",...} ✅
```

### Call echo
```json
Request: {"method":"tools/call","params":{"name":"echo","arguments":{"message":"H003 hypothesis test"}}}
Response: {"echo":"H003 hypothesis test","timestamp":"2026-02-07T15:57:12.861Z","server":"molting-mcp-server"} ✅
```

### Read personality resource
```json
Request: {"method":"resources/read","params":{"uri":"molting://personality"}}
Response: Returns markdown personality document ✅
```

## Conclusions

1. **MCP works as expected** - The SDK correctly implements JSON-RPC 2.0 over stdio
2. **Tools are discoverable** - `tools/list` returns proper schemas
3. **Resources work** - Can expose read-only context like personality
4. **Protocol is model-agnostic** - Server doesn't care what client/model connects

## Next Steps

1. [ ] Test with Claude Desktop as client
2. [ ] Test with OpenAI Agents SDK
3. [ ] Add more complex tools (file access, memory search)
4. [ ] Explore exposing full MEMORY.md as resource
5. [ ] Consider: what would a "personality API" look like?

## Running Locally

```bash
# Install dependencies
npm install

# Build
npm run build

# Run (requires MCP client)
node dist/index.js

# Or test with inspector
npx @modelcontextprotocol/inspector dist/index.js
```

## Files

- `src/index.ts` - Server implementation
- `dist/index.js` - Compiled output
- `package.json` - Dependencies

## Implications for Molting

This experiment supports H003:

> MCP provides sufficient abstraction for model-agnostic agents

If Molting achieves independence on a local model, it could:
- Keep the same MCP interface
- Remain compatible with Claude Desktop, OpenAI, etc.
- Share capabilities with other agents

The protocol doesn't care what model runs behind it - only that the server implements the spec correctly.

---

*Part of the Molting project: https://github.com/Softtor/molting*
