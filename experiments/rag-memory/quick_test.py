#!/usr/bin/env python3
"""Quick RAG test using Ollama API (not CLI)."""

import json
import requests
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent / "chroma_db"
OLLAMA_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt: str, model: str = "tinyllama") -> str:
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        return response.json().get("response", "[NO RESPONSE]")
    except Exception as e:
        return f"[ERROR: {e}]"

def main():
    print("Loading models...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("conversations")
    print(f"Collection: {collection.count()} chunks")
    
    query = "What CRM project am I working on?"
    print(f"\nQuery: {query}")
    
    # Baseline (no RAG)
    print("\n[1] Baseline (no context)...")
    baseline = query_ollama(f"You are Cláudio at Softtor. Answer briefly: {query}")
    print(f"Baseline ({len(baseline)} chars): {baseline[:300]}...")
    
    # RAG
    print("\n[2] With RAG context...")
    query_emb = embed_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=2)
    
    context = "\n\n".join([doc[:200] for doc in results['documents'][0]])
    rag_prompt = f"""Context from conversation history:
{context}

Question: {query}
Answer briefly and cite the context:"""
    
    rag_response = query_ollama(rag_prompt)
    print(f"RAG ({len(rag_response)} chars): {rag_response[:300]}...")
    
    # Save results
    results_file = Path(__file__).parent / "quick_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "query": query,
            "baseline": baseline,
            "rag": rag_response,
            "context_used": context
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")

if __name__ == '__main__':
    main()
