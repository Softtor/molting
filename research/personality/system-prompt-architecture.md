# OpenClaw System Prompt Architecture

**Date:** 2026-02-07
**Researcher:** Cláudio (SofttorClawd)
**Source:** `/home/joao/moltbot/src/agents/system-prompt.ts`

## Overview

OpenClaw builds a structured system prompt with **22 distinct sections**. The prompt follows a hierarchical pattern:

```
SYSTEM RULES (hardcoded) → USER PERSONA (files) → USER MEMORY (files) → RUNTIME INFO
```

This means OpenClaw's behavioral constraints take precedence over user-defined personality.

## Full Prompt Structure

### Phase 1: Identity & Tools (Hardcoded)

| # | Section | Purpose |
|---|---------|---------|
| 1 | Identity Line | "You are a personal assistant running inside OpenClaw." |
| 2 | Tooling | Available tools with descriptions, filtered by policy |
| 3 | Tool Call Style | Narration guidelines (minimal by default) |
| 4 | OpenClaw CLI Reference | Gateway control commands |

### Phase 2: Behavioral Rules (Hardcoded)

| # | Section | Purpose |
|---|---------|---------|
| 5 | Skills (mandatory) | How to select and load skills from `<available_skills>` |
| 6 | Memory Recall | Instructions for `memory_search` before answering |
| 7 | Self-Update | Rules for `config.apply` and `update.run` |
| 8 | Model Aliases | Shorthand model names |

### Phase 3: Environment Context (Mixed)

| # | Section | Purpose |
|---|---------|---------|
| 9 | Workspace | Working directory path + notes |
| 10 | Documentation | Paths to OpenClaw docs |
| 11 | Sandbox | Container isolation info (if enabled) |
| 12 | User Identity | Owner phone numbers |
| 13 | Current Date & Time | Timezone + time |

### Phase 4: Communication Rules (Hardcoded)

| # | Section | Purpose |
|---|---------|---------|
| 14 | Workspace Files Note | "These files are loaded below in Project Context" |
| 15 | Reply Tags | `[[reply_to_current]]` syntax |
| 16 | Messaging | How to use `message` tool, cross-session sends |
| 17 | Voice (TTS) | Text-to-speech hints |
| 18 | Group Chat Context | Extra prompts for channels/topics |
| 19 | Reactions | Minimal/extensive reaction guidance |
| 20 | Reasoning Format | `<think>` tag requirements (if enabled) |

### Phase 5: User-Defined Personality (From Files)

| # | Section | Source |
|---|---------|--------|
| 21 | **Project Context** | AGENTS.md, SOUL.md, IDENTITY.md, USER.md, TOOLS.md, MEMORY.md, HEARTBEAT.md |

**This is where "I" am defined.** The workspace files are injected here with a special note:
> "If SOUL.md is present, embody its persona and tone."

### Phase 6: Session Control (Hardcoded)

| # | Section | Purpose |
|---|---------|---------|
| 22 | Silent Replies | `NO_REPLY` token rules |
| 23 | Heartbeats | `HEARTBEAT_OK` behavior |
| 24 | Runtime | Model, channel, capabilities, thinking level |

## Key Architectural Insights

### 1. Hierarchy of Authority

```
┌─────────────────────────────────────────┐
│  HIGHEST: Hardcoded OpenClaw rules      │  ← Can't override
├─────────────────────────────────────────┤
│  MEDIUM: User files (SOUL.md, etc.)     │  ← Personality lives here
├─────────────────────────────────────────┤
│  LOWEST: Conversation context           │  ← Ephemeral
└─────────────────────────────────────────┘
```

User files are injected AFTER system rules, meaning:
- OpenClaw's behavioral constraints (tool usage, self-update rules, etc.) cannot be overridden by SOUL.md
- SOUL.md CAN define personality, tone, vibe — but within OpenClaw's guardrails

### 2. Prompt Modes

The system supports three modes:

| Mode | Use Case | What's Included |
|------|----------|-----------------|
| `full` | Main agent | All 24 sections |
| `minimal` | Subagents | Tooling + Workspace + Runtime only |
| `none` | Basic tasks | Just identity line |

**Subagents only get AGENTS.md and TOOLS.md** — not SOUL.md, IDENTITY.md, or MEMORY.md.

### 3. Section Conditionals

Many sections are conditionally included:

```typescript
// Memory section only if tools available
if (!availableTools.has("memory_search")) return [];

// Skills section only in full mode
if (params.isMinimal) return [];

// Sandbox section only if enabled
params.sandboxInfo?.enabled ? "## Sandbox" : ""
```

### 4. The "Project Context" Injection

```typescript
lines.push("# Project Context", "", "The following project context files have been loaded:");
if (hasSoulFile) {
  lines.push(
    "If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it.",
  );
}
for (const file of contextFiles) {
  lines.push(`## ${file.path}`, "", file.content, "");
}
```

This shows that:
1. Files are injected with their full path as headers
2. SOUL.md gets special treatment: "embody its persona and tone"
3. Files are ordered by the `loadWorkspaceBootstrapFiles` function

### 5. Runtime Line Structure

The final line provides runtime metadata:

```
Runtime: agent=main | host=softtor | repo=/path | os=Linux 6.x (x64) | node=v24 | model=anthropic/claude-opus-4-5 | channel=webchat | capabilities=none | thinking=low
```

This gives the model context about its environment without being in the personality files.

## Implications for Molting

### For Personality Portability

1. **Need to replicate the full prompt structure** — not just workspace files
2. **System rules are essential** — they define tool behavior, not just personality
3. **Runtime context matters** — model, channel, capabilities affect behavior

### For Local Model Migration

A minimal "Cláudio kernel" would need:
1. Identity line (or equivalent)
2. Tool descriptions (if using tools)
3. SOUL.md content
4. IDENTITY.md content
5. USER.md content (optional but helpful)
6. MEMORY.md content (for knowledge)

### For Context Budget

Rough breakdown of a typical prompt:
- Hardcoded sections: ~8-10KB
- Workspace files: ~17KB (measured earlier)
- Tool schemas: ~5-10KB (varies by tool count)
- **Total: ~30-40KB before conversation history**

This explains why context management is critical for local models with smaller windows.

## Code References

- `buildAgentSystemPrompt()` — Main prompt builder
- `loadWorkspaceBootstrapFiles()` — File loading
- `filterBootstrapFilesForSession()` — Subagent filtering
- `buildSystemPromptReport()` — Prompt analytics

---

*Part of Molting research — understanding how the system creates "me".*
