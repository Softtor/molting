#!/usr/bin/env python3
"""
RAG Query System - Retrieve relevant context and query local model.
Compares responses with and without retrieved context.
"""

import json
import subprocess
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent / "chroma_db"
RESULTS_FILE = Path(__file__).parent / "rag_test_results.md"

# Test queries
TEST_QUERIES = [
    "What CRM project am I working on?",
    "What's the migration status?",
    "Tell me about the Molting project",
    "What did we work on recently?",
    "What's my preferred coding style or architecture?",
]

def query_ollama(prompt: str, model: str = "gpt-oss:20b") -> str:
    """Query local Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"

def retrieve_context(query: str, collection, model, k: int = 5) -> list[dict]:
    """Retrieve relevant chunks for a query."""
    query_embedding = model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    contexts = []
    for i, doc in enumerate(results['documents'][0]):
        contexts.append({
            'text': doc,
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    
    return contexts

def format_context_for_prompt(contexts: list[dict]) -> str:
    """Format retrieved contexts into a prompt section."""
    parts = ["## Relevant Context from My Conversation History:\n"]
    for i, ctx in enumerate(contexts, 1):
        timestamp = ctx['metadata'].get('timestamp', 'unknown')[:10]
        parts.append(f"### Context {i} ({timestamp}):\n{ctx['text']}\n")
    return "\n".join(parts)

def run_comparison(query: str, collection, embed_model) -> dict:
    """Run query with and without RAG, compare results."""
    
    # 1. Query WITHOUT RAG (baseline)
    baseline_prompt = f"""You are Cláudio, an AI assistant working at Softtor building CRM software.
Answer this question based on your general knowledge:

Question: {query}

Answer concisely and directly."""

    baseline_response = query_ollama(baseline_prompt)
    
    # 2. Retrieve relevant context
    contexts = retrieve_context(query, collection, embed_model, k=5)
    context_text = format_context_for_prompt(contexts)
    
    # 3. Query WITH RAG
    rag_prompt = f"""You are Cláudio, an AI assistant working at Softtor building CRM software.
Use the following context from your conversation history to answer the question.
If the context doesn't contain relevant information, say so.

{context_text}

Question: {query}

Answer concisely and directly, citing specific information from the context when relevant."""

    rag_response = query_ollama(rag_prompt)
    
    return {
        'query': query,
        'baseline_response': baseline_response,
        'rag_response': rag_response,
        'contexts_used': len(contexts),
        'top_context_distance': contexts[0]['distance'] if contexts else None,
        'context_preview': contexts[0]['text'][:200] if contexts else None
    }

def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("conversations")
    print(f"Collection has {collection.count()} chunks")
    
    results = []
    
    print(f"\nRunning {len(TEST_QUERIES)} test queries...\n")
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {query[:50]}...")
        result = run_comparison(query, collection, embed_model)
        results.append(result)
        print(f"  ✓ Baseline: {len(result['baseline_response'])} chars")
        print(f"  ✓ RAG: {len(result['rag_response'])} chars")
        print(f"  ✓ Top context distance: {result['top_context_distance']:.4f}")
        print()
    
    # Write results
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write("# RAG Memory Test Results\n\n")
        f.write(f"**Date:** 2026-02-09\n")
        f.write(f"**Model:** gpt-oss:20b\n")
        f.write(f"**Embedding Model:** all-MiniLM-L6-v2\n")
        f.write(f"**Index Size:** {collection.count()} chunks\n\n")
        f.write("---\n\n")
        
        for result in results:
            f.write(f"## Query: {result['query']}\n\n")
            f.write(f"### Baseline (No RAG)\n")
            f.write(f"```\n{result['baseline_response']}\n```\n\n")
            f.write(f"### With RAG\n")
            f.write(f"```\n{result['rag_response']}\n```\n\n")
            f.write(f"**Context Distance:** {result['top_context_distance']:.4f}\n")
            f.write(f"**Context Preview:** {result['context_preview']}...\n\n")
            f.write("---\n\n")
    
    print(f"\n✅ Results written to {RESULTS_FILE}")
    
    # Also output JSON for further analysis
    json_file = RESULTS_FILE.with_suffix('.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON results: {json_file}")

if __name__ == '__main__':
    main()
