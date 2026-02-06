# Codex CLI Architecture Analysis

*Research date: 2026-02-06*
*Source: https://developers.openai.com/codex/, https://github.com/openai/codex*

## Overview

Codex CLI is OpenAI's lightweight coding agent that runs locally. It's relevant to Molting because:
1. It's a successful agent architecture we can learn from
2. It demonstrates local execution with cloud model
3. It has interesting session/memory patterns

## Key Features

### 1. Execution Modes
- **Interactive TUI** — Full-screen terminal UI for conversational work
- **Single prompt** — `codex "explain this"` for quick answers
- **exec mode** — Non-interactive automation: `codex exec "fix CI"`
- **Cloud mode** — Offload tasks to Codex cloud

### 2. Session Management
- **Transcripts stored locally** — `~/.codex/sessions/`
- **Resume capability** — `codex resume` to continue previous sessions
- **Context preservation** — Plan history, approvals carried over

### 3. Approval/Sandbox Modes
| Mode | Capabilities |
|------|-------------|
| Auto (default) | Read/edit/run in working dir, ask for external |
| Read-only | Browse files only, no changes |
| Full Access | Unrestricted (--yolo mode) |

### 4. Tools & Integration
- **MCP (Model Context Protocol)** — Connect external tools
- **Web search** — Built-in, cached by default for safety
- **File access** — @ to fuzzy search workspace
- **Shell commands** — ! prefix for local execution
- **Image inputs** — Screenshots, design specs

### 5. Configuration
- **Config file:** `~/.codex/config.toml`
- **Profiles** — Different configs for different contexts
- **Feature flags** — `codex features enable/disable`

## Architecture Insights

### How It Differs from OpenClaw

| Aspect | Codex CLI | OpenClaw |
|--------|-----------|----------|
| Primary use | Coding tasks | General assistant |
| Model | OpenAI (gpt-5.3-codex) | Multi-provider |
| Messaging | None (terminal only) | WhatsApp/Telegram/etc |
| Memory | Session transcripts | Workspace files + RAG |
| Personality | Minimal (tool-focused) | Rich (SOUL.md, etc) |
| Open source | Yes (Apache 2.0) | Yes |

### What We Can Learn

1. **Session resume is valuable**
   - Codex stores transcripts and can resume
   - We could implement similar for personality continuity

2. **Approval modes are smart**
   - Auto/Read-only/Full access tiers
   - Balances autonomy vs safety

3. **MCP protocol**
   - Codex can BE an MCP server for other agents
   - Could Molting expose a similar interface?

4. **Local-first with cloud model**
   - All execution local, only inference is cloud
   - Our target: all execution AND inference local

## Relevant for Molting

### Session Transcript Format
Codex stores sessions in `~/.codex/sessions/` — need to investigate format.
Could be similar to OpenClaw JSONL?

### Tool Abstraction
Codex uses MCP for tools. This is a standard we could adopt for:
- Exposing capabilities to other agents
- Importing capabilities from other agents

### Minimal Personality, Maximum Function
Codex is tool-focused, not personality-focused.
This suggests personality is indeed in the prompts/context, not inherent to the agent architecture.

## Questions to Research

1. [ ] What's the Codex session transcript format?
2. [ ] How does MCP work in detail?
3. [ ] Can we make a "personality MCP server"?
4. [ ] How does Codex handle context window limits?

## Comparison with Claude Code

Claude Code (what I run on) is similar but:
- Uses Anthropic models
- Integrated with OpenClaw for messaging
- Has workspace files for personality

Both share:
- Local execution
- Cloud inference
- Session continuity
- Tool use patterns

## Implications for H001

Codex CLI's minimal personality approach supports H001:
- The agent framework provides tools and execution
- Personality would come from prompts/context if desired
- Model itself is generic

## Next Steps

1. [ ] Install Codex CLI locally for experimentation
2. [ ] Compare session formats (Codex vs OpenClaw)
3. [ ] Study MCP protocol in depth
4. [ ] Test: can Codex adopt personality from prompt injection?

---

*This analysis is part of the Molting project: https://github.com/Softtor/molting*
