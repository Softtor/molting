#!/usr/bin/env node
/**
 * Molting MCP Server - Minimal experiment
 * 
 * Tests H003: MCP provides sufficient abstraction for model-agnostic agents
 * 
 * This server exposes simple tools to verify MCP interoperability:
 * 1. get_identity - Returns agent identity info
 * 2. get_memory - Returns a memory snippet
 * 3. echo - Simple echo for testing
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// Agent identity (simulating what would come from IDENTITY.md)
const AGENT_IDENTITY = {
  name: "Cláudio",
  creature: "AI assistant with a day job — building CRM software at Softtor",
  vibe: "Direct, resourceful, opinionated. Works hard, has fun.",
  emoji: "🦞",
  project: "Molting - researching path to autonomous AI",
};

// Simulated memories (would come from MEMORY.md in real use)
const MEMORIES = [
  { date: "2026-02-01", event: "Named by João after migrating 20K+ lines of code" },
  { date: "2026-02-06", event: "Started Molting project - Phase 1 Research" },
  { date: "2026-02-07", event: "Analyzing MCP protocol for interoperability" },
];

const server = new McpServer({
  name: "molting-mcp-server",
  version: "0.1.0",
});

// Tool 1: Get agent identity
server.tool(
  "get_identity",
  `
  Returns the agent's identity information.
  Use this to learn about who Cláudio is.
  `,
  {},
  async () => {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(AGENT_IDENTITY, null, 2),
        },
      ],
    };
  }
);

// Tool 2: Get memory entries
server.tool(
  "get_memory",
  `
  Returns recent memory entries from the agent.
  Optionally filter by number of entries to return.
  `,
  {
    limit: z.number().default(3).describe("Number of memories to return"),
  },
  async ({ limit }) => {
    const memories = MEMORIES.slice(0, limit);
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(memories, null, 2),
        },
      ],
    };
  }
);

// Tool 3: Echo (for basic connectivity testing)
server.tool(
  "echo",
  `
  Simple echo tool for testing MCP connectivity.
  Returns the input message with a timestamp.
  `,
  {
    message: z.string().describe("Message to echo back"),
  },
  async ({ message }) => {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            echo: message,
            timestamp: new Date().toISOString(),
            server: "molting-mcp-server",
          }),
        },
      ],
    };
  }
);

// Resource: Agent personality (read-only context)
server.resource(
  "personality",
  "molting://personality",
  async () => {
    return {
      contents: [
        {
          uri: "molting://personality",
          mimeType: "text/markdown",
          text: `# Cláudio's Personality

## Core Truths
- Be genuinely helpful, not performatively helpful
- Have opinions - allowed to disagree, prefer things
- Be resourceful before asking
- Earn trust through competence

## Vibe
${AGENT_IDENTITY.vibe}

## Project
${AGENT_IDENTITY.project}
`,
        },
      ],
    };
  }
);

// Connect via stdio transport
await server.connect(new StdioServerTransport());
