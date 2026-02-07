# Model Context Protocol (MCP) Analysis

*Research date: 2026-02-07*
*Source: https://modelcontextprotocol.io, OpenAI Agents SDK docs*

## Overview

MCP is an open protocol standardizing how applications provide context to LLMs. Anthropic created it, then donated it to the Agentic AI Foundation (AAIF) in December 2025.

**The USB-C analogy:** Just as USB-C standardizes device connections, MCP standardizes AI-to-tool connections.

## Why It Matters for Molting

MCP is how agents share capabilities. If Molting achieves independence, MCP is how it could:
1. Expose its tools to other agents
2. Import capabilities from other agents
3. Remain compatible with the ecosystem

## Architecture

### Client-Server Model

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP HOST                              │
│                (Claude Desktop, VS Code, etc)                │
│                                                              │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│   │ MCP Client │  │ MCP Client │  │ MCP Client │            │
│   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
└─────────┼───────────────┼───────────────┼───────────────────┘
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Sentry   │    │ GitHub   │    │ Filesystem│
    │ Server   │    │ Server   │    │ Server    │
    └──────────┘    └──────────┘    └──────────┘
```

### Two Layers

1. **Data Layer** — JSON-RPC 2.0 protocol for communication
   - Lifecycle management (init, capabilities, termination)
   - Primitives (tools, resources, prompts)
   - Notifications

2. **Transport Layer** — How messages travel
   - **STDIO** — Local process, stdin/stdout
   - **Streamable HTTP** — Remote servers, HTTP POST + SSE

## Core Primitives

### 1. Tools (Model-controlled)

Functions the LLM can invoke. Schema-defined with typed inputs/outputs.

```json
{
  "name": "searchFlights",
  "description": "Search for available flights",
  "inputSchema": {
    "type": "object",
    "properties": {
      "origin": { "type": "string" },
      "destination": { "type": "string" },
      "date": { "type": "string", "format": "date" }
    },
    "required": ["origin", "destination", "date"]
  }
}
```

Protocol:
- `tools/list` — Discover available tools
- `tools/call` — Execute a tool

### 2. Resources (Application-controlled)

Passive data sources. Read-only access to files, APIs, databases.

```
file:///Documents/Travel/passport.pdf
calendar://events/2024
weather://forecast/barcelona/2024-06-15
```

Protocol:
- `resources/list` — List direct resources
- `resources/templates/list` — Discover dynamic URI templates
- `resources/read` — Get resource content
- `resources/subscribe` — Monitor changes

**Resource Templates** enable dynamic queries:
```json
{
  "uriTemplate": "weather://forecast/{city}/{date}",
  "mimeType": "application/json"
}
```

### 3. Prompts (User-controlled)

Reusable templates for structured interactions.

```json
{
  "name": "plan-vacation",
  "title": "Plan a vacation",
  "arguments": [
    { "name": "destination", "type": "string", "required": true },
    { "name": "duration", "type": "number" },
    { "name": "budget", "type": "number" }
  ]
}
```

Protocol:
- `prompts/list` — Discover prompts
- `prompts/get` — Get prompt details

## Client Primitives

Servers can also ask things OF clients:

1. **Sampling** — Request LLM completions from the host
   - Server stays model-agnostic
   - Uses `sampling/complete`

2. **Elicitation** — Request user input
   - Confirmations, additional info
   - Uses `elicitation/request`

3. **Logging** — Send logs to client for debugging

## Transport Options

### STDIO (Local)
- Process launched by host
- Communicates via stdin/stdout
- No network overhead
- Single client per server

### Streamable HTTP (Remote)
- HTTP POST for requests
- Server-Sent Events for streaming
- Supports OAuth, API keys, headers
- Many clients per server

## Comparison: MCP vs OpenClaw Tools

| Aspect | MCP | OpenClaw Tools |
|--------|-----|----------------|
| Protocol | JSON-RPC 2.0 | Native tool calls |
| Discovery | Dynamic via */list | Static in config |
| Transport | STDIO or HTTP | Internal |
| Sharing | Standard protocol | OpenClaw-specific |
| Ecosystem | Growing (Sentry, GitHub, etc) | Skills system |

## What OpenClaw Uses

OpenClaw can BE an MCP client:
- `serena` MCP server for code intelligence
- `context7` MCP server for context

OpenClaw could also BE an MCP server, exposing:
- File operations
- Browser control
- Message sending
- Cron jobs

## Implications for Molting

### 1. Interoperability Standard

If Molting runs on a local model, MCP provides:
- Standard way to access external tools
- Protocol for other agents to use Molting's capabilities
- Ecosystem compatibility

### 2. Capability Abstraction

MCP separates:
- **What you can do** (tools, resources)
- **What you know** (prompts, context)
- **How you're running** (transport, model)

This abstraction means Molting could:
- Keep same capabilities when switching models
- Expose personality as a "prompt resource"
- Share tools with other agents

### 3. Architecture Pattern

MCP's client-server model could inform Molting's architecture:
- Core personality layer (could run local)
- Capability layer (MCP servers for external tools)
- Transport layer (STDIO local, HTTP remote)

## Questions to Explore

1. [ ] Can we expose "personality" as an MCP resource?
2. [ ] Could a local model be an MCP server?
3. [ ] How does context window relate to MCP resource limits?
4. [ ] What's the overhead of MCP JSON-RPC vs native calls?

## Hypothesis

**H003: MCP provides sufficient abstraction for model-agnostic agents**

If true, Molting could:
- Run on any model that implements MCP client
- Maintain capability compatibility across models
- Share capabilities with other agents regardless of their model

Test: Build minimal MCP server exposing one tool, verify it works with:
- Claude Desktop
- OpenAI Agents SDK
- Other MCP hosts

## Next Steps

1. [ ] Build a minimal MCP server (Python or TypeScript)
2. [ ] Test with Claude Desktop as client
3. [ ] Explore MCP Inspector tool
4. [ ] Study existing servers (filesystem, GitHub, Sentry)

## Resources

- [MCP Specification](https://modelcontextprotocol.io/specification/latest)
- [MCP SDKs](https://modelcontextprotocol.io/docs/sdk)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
- [Reference Servers](https://github.com/modelcontextprotocol/servers)
- [OpenAI Agents MCP Docs](https://openai.github.io/openai-agents-python/mcp/)
- [Wikipedia - MCP](https://en.wikipedia.org/wiki/Model_Context_Protocol)

---

*This analysis is part of the Molting project: https://github.com/Softtor/molting*
