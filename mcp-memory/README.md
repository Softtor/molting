# Molting Memory MCP

MCP server that provides RAG (Retrieval-Augmented Generation) over Molting research knowledge.

## Purpose

Keep Cláudio contextualizado with all Molting research findings, experiments, and discoveries without re-reading everything each session.

## Tools

- `molting_search(query)` - Semantic search over all Molting knowledge
- `molting_context(topic)` - Get structured context about a specific topic
- `molting_recent()` - Get recent research activity

## Indexed Sources

- `experiments/` - All experiment results and findings
- `research/` - Phase 1 research notes
- `docs/` - Project documentation
- `README.md`, `DIRECTIVES.md` - Project overview

## Setup

```bash
cd mcp-memory
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_index.py  # Index all Molting knowledge
```

## Usage with OpenClaw

Add to `~/.openclaw/config.yaml`:

```yaml
mcp:
  servers:
    molting-memory:
      command: python
      args: ["/path/to/molting/mcp-memory/server.py"]
```

## Architecture

```
mcp-memory/
├── server.py         # FastMCP server
├── build_index.py    # Index builder
├── requirements.txt  # Dependencies
├── chroma_db/        # Vector store (gitignored)
└── README.md
```
