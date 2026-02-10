#!/usr/bin/env python3
"""Full RAG comparison: 5+ diverse queries, 2 models, baseline vs RAG."""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent / "chroma_db"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = ["tinyllama", "phi3:mini"]

# 5+ diverse queries covering different aspects
QUERIES = [
    {
        "id": "technical",
        "question": "What technology stack and frameworks do we use at Softtor?",
        "aspect": "Technical knowledge"
    },
    {
        "id": "personal",
        "question": "Who is João and what is his role?",
        "aspect": "Personal/team knowledge"
    },
    {
        "id": "project",
        "question": "What is the Molting project about and what are its goals?",
        "aspect": "Project-specific knowledge"
    },
    {
        "id": "architecture",
        "question": "What architectural decisions were made for the CRM system?",
        "aspect": "Architecture/design decisions"
    },
    {
        "id": "history",
        "question": "What happened during the migration to the new infrastructure?",
        "aspect": "Historical events"
    },
    {
        "id": "workflow",
        "question": "What is my typical development workflow and tools I use?",
        "aspect": "Workflow/practices"
    }
]

def query_ollama(prompt: str, model: str = "tinyllama", timeout: int = 120) -> tuple[str, float]:
    """Query Ollama API and return (response, time_taken)."""
    start = time.time()
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 300  # Limit response length
            }
        }, timeout=timeout)
        elapsed = time.time() - start
        return response.json().get("response", "[NO RESPONSE]"), elapsed
    except Exception as e:
        elapsed = time.time() - start
        return f"[ERROR: {e}]", elapsed

def run_comparison():
    """Run full comparison across all queries and models."""
    print("=" * 80)
    print("RAG EXPERIMENT: FULL COMPARISON")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Queries: {len(QUERIES)}")
    print()
    
    # Load models
    print("Loading embedding model and ChromaDB...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("conversations")
    chunk_count = collection.count()
    print(f"✓ ChromaDB loaded: {chunk_count} chunks")
    print()
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "chunk_count": chunk_count,
            "embedding_model": "all-MiniLM-L6-v2",
            "models_tested": MODELS,
            "queries_count": len(QUERIES)
        },
        "results": []
    }
    
    for query_info in QUERIES:
        query_id = query_info["id"]
        question = query_info["question"]
        aspect = query_info["aspect"]
        
        print("─" * 80)
        print(f"Query [{query_id}]: {question}")
        print(f"Aspect: {aspect}")
        print("─" * 80)
        
        # Get RAG context
        query_emb = embed_model.encode([question])[0].tolist()
        rag_results = collection.query(query_embeddings=[query_emb], n_results=3)
        context_docs = rag_results['documents'][0]
        context = "\n\n".join([f"[Chunk {i+1}] {doc[:250]}" for i, doc in enumerate(context_docs)])
        
        query_result = {
            "id": query_id,
            "question": question,
            "aspect": aspect,
            "context_chunks_used": len(context_docs),
            "models": {}
        }
        
        for model in MODELS:
            print(f"\n[{model.upper()}]")
            
            # Baseline (no context)
            print("  Baseline (no context)...", end=" ", flush=True)
            baseline_prompt = f"""You are Cláudio, an AI assistant at Softtor.
Answer this question briefly and accurately: {question}"""
            baseline_response, baseline_time = query_ollama(baseline_prompt, model)
            print(f"✓ ({baseline_time:.1f}s)")
            
            # RAG (with context)
            print("  RAG (with context)...", end=" ", flush=True)
            rag_prompt = f"""You are Cláudio at Softtor. Use the following context from your conversation history to answer the question.

Context:
{context}

Question: {question}

Answer based on the context above. Be specific and cite what you remember:"""
            rag_response, rag_time = query_ollama(rag_prompt, model)
            print(f"✓ ({rag_time:.1f}s)")
            
            query_result["models"][model] = {
                "baseline": {
                    "response": baseline_response,
                    "time_seconds": round(baseline_time, 2),
                    "length_chars": len(baseline_response)
                },
                "rag": {
                    "response": rag_response,
                    "time_seconds": round(rag_time, 2),
                    "length_chars": len(rag_response)
                }
            }
        
        results["results"].append(query_result)
        print()
    
    # Save results
    output_file = Path(__file__).parent / "full_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 80)
    print(f"✅ COMPLETE! Results saved to {output_file}")
    print("=" * 80)
    
    # Summary statistics
    print("\nSUMMARY:")
    for model in MODELS:
        baseline_times = [r["models"][model]["baseline"]["time_seconds"] for r in results["results"]]
        rag_times = [r["models"][model]["rag"]["time_seconds"] for r in results["results"]]
        
        avg_baseline = sum(baseline_times) / len(baseline_times)
        avg_rag = sum(rag_times) / len(rag_times)
        
        print(f"\n{model.upper()}:")
        print(f"  Avg baseline time: {avg_baseline:.2f}s")
        print(f"  Avg RAG time: {avg_rag:.2f}s")
        print(f"  Overhead: {avg_rag - avg_baseline:.2f}s ({((avg_rag/avg_baseline - 1) * 100):.1f}%)")
    
    return results

if __name__ == '__main__':
    results = run_comparison()
