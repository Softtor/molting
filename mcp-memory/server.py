#!/usr/bin/env python3
"""
Molting Memory MCP Server

Provides RAG-powered access to all Molting research knowledge.
"""

from pathlib import Path
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP

CHROMA_DIR = Path(__file__).parent / "chroma_db"
MOLTING_ROOT = Path(__file__).parent.parent

# Initialize MCP server
mcp = FastMCP("molting-memory")

# Lazy-loaded components
_model = None
_collection = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection("molting_knowledge")
    return _collection

def _search(query: str, n_results: int = 5) -> str:
    """Internal search function (not wrapped by MCP)."""
    model = get_model()
    collection = get_collection()
    
    query_embedding = model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    output = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        source = metadata.get('source', 'unknown')
        output.append(f"### Result {i+1} (from {source})\n{doc[:500]}...")
    
    return "\n\n".join(output) if output else "No results found."

@mcp.tool()
def molting_search(query: str, n_results: int = 5) -> str:
    """
    Search Molting knowledge base semantically.
    
    Args:
        query: What to search for (natural language)
        n_results: Number of results to return (default 5)
    
    Returns:
        Relevant excerpts from Molting research with sources
    """
    return _search(query, n_results)

@mcp.tool()
def molting_context(topic: str) -> str:
    """
    Get structured context about a specific Molting topic.
    
    Args:
        topic: Topic to get context for (e.g., "personality portability", "RAG experiment", "OpenClaw architecture")
    
    Returns:
        Comprehensive context about the topic from multiple sources
    """
    results = _search(topic, n_results=8)
    return f"# Context: {topic}\n\n{results}"

@mcp.tool()
def molting_hypotheses() -> str:
    """
    List all Molting hypotheses and their current status.
    
    Returns:
        Table of hypotheses with status (validated/pending/failed)
    """
    results = _search("hypothesis H001 H002 H003 H004 H005 validated status", n_results=10)
    return f"# Molting Hypotheses\n\n{results}"

@mcp.tool()
def molting_experiments() -> str:
    """
    List all experiments and their results.
    
    Returns:
        Summary of experiments conducted
    """
    results = _search("experiment results VALIDATED findings setup test", n_results=10)
    return f"# Molting Experiments\n\n{results}"

@mcp.tool()  
def molting_reindex() -> str:
    """
    Rebuild the Molting knowledge index.
    
    Use this after adding new research or experiments.
    
    Returns:
        Status of reindexing operation
    """
    import subprocess
    result = subprocess.run(
        ["python", str(Path(__file__).parent / "build_index.py")],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent)
    )
    
    if result.returncode == 0:
        # Reset cached collection
        global _collection
        _collection = None
        return f"✅ Reindexing complete!\n\n{result.stdout}"
    else:
        return f"❌ Reindexing failed:\n{result.stderr}"

if __name__ == "__main__":
    mcp.run()
