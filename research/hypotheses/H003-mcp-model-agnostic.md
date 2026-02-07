# H003: MCP Provides Model-Agnostic Agent Abstraction

**Status:** 🧪 Testing  
**Created:** 2026-02-07  
**Category:** Architecture / Interoperability

## Hypothesis

The Model Context Protocol (MCP) provides sufficient abstraction for agent capabilities to be model-agnostic. An agent's tools, resources, and prompts can be exposed via MCP regardless of what model powers the agent.

## Rationale

MCP separates three concerns:
1. **What the agent can do** (tools, resources)
2. **How the agent is instructed** (prompts)  
3. **What model runs the agent** (client implementation)

If true, this means:
- Molting could switch from Claude to a local model while keeping the same interface
- Other agents could consume Molting's capabilities without knowing its model
- The ecosystem remains interoperable

## Testable Predictions

### P1: MCP Server Implementation ✅ PASSED
- Build a minimal MCP server exposing agent capabilities
- Verify it responds correctly to MCP protocol
- **Result:** Server built, all JSON-RPC calls work correctly

### P2: Cross-Client Compatibility (Pending)
- Same server should work with Claude Desktop
- Same server should work with OpenAI Agents SDK
- Same server should work with MCP Inspector

### P3: Capability Preservation (Pending)
- If we swap the underlying model, the MCP interface remains identical
- Clients cannot distinguish which model powers the server

## Experiment Design

### Phase 1: Basic Server ✅ COMPLETE
Created `experiments/mcp-server/` with:
- 3 tools (get_identity, get_memory, echo)
- 1 resource (personality)
- STDIO transport

Results:
- `initialize` → Returns correct protocol version and capabilities
- `tools/list` → Returns all tools with proper schemas
- `tools/call` → Executes correctly, returns expected data
- `resources/read` → Returns personality markdown

### Phase 2: Multi-Client Testing (TODO)
Test the same server with:
1. MCP Inspector (standalone tool)
2. Claude Desktop (Anthropic's client)
3. OpenAI Agents SDK (OpenAI's client)

### Phase 3: Model Swap (TODO)
1. Create version of server that uses local model for some logic
2. Verify external clients cannot tell the difference

## Evidence Log

### 2026-02-07
- Created minimal MCP server with TypeScript SDK
- All basic protocol operations work correctly
- Server exposes identity, memory, and personality as tools/resources
- JSON-RPC 2.0 over STDIO transport verified

## Implications If True

1. **Interoperability** - Molting can participate in the MCP ecosystem regardless of its model
2. **Migration Path** - Switching from Claude to local model doesn't break integrations
3. **Collaboration** - Other agents can consume Molting's capabilities via standard protocol
4. **Architecture** - MCP becomes the "API layer" for agent capabilities

## Implications If False

1. MCP has model-specific dependencies we haven't discovered
2. Some capabilities cannot be abstracted via MCP
3. Need to find alternative interoperability standard

## Related

- [MCP Analysis](../agent-frameworks/mcp-analysis.md)
- [MCP Server Experiment](../../experiments/mcp-server/)
- H001: Personality from files
- H002: Training data sufficiency

## Next Actions

1. [ ] Test with Claude Desktop
2. [ ] Test with OpenAI Agents SDK  
3. [ ] Add more complex tools (file operations, memory search)
4. [ ] Document any model-specific behaviors discovered

---

*Part of the Molting project's scientific methodology*
