#!/usr/bin/env python3
"""
Chunk size optimization experiment.
Tests different chunk sizes and overlap percentages to find optimal retrieval quality.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import tempfile
import shutil

CONVERSATIONS_FILE = Path(__file__).parent / "conversations.json"

# Test parameters
CHUNK_SIZES = [256, 512, 1024, 2048]  # tokens
OVERLAPS = [0.0, 0.1, 0.25, 0.5]  # percentage

# Reuse queries from full_comparison.py
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


def load_conversations():
    """Load extracted conversations."""
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return len(text) // 4


def chunk_by_tokens(data: Dict, chunk_size: int, overlap_pct: float) -> List[Dict]:
    """
    Create token-based chunks with configurable overlap.
    
    Args:
        data: Conversation data
        chunk_size: Target chunk size in tokens
        overlap_pct: Overlap percentage (0.0-1.0)
    
    Returns:
        List of chunks with text and metadata
    """
    chunks = []
    overlap_tokens = int(chunk_size * overlap_pct)
    stride = chunk_size - overlap_tokens
    
    for session in data['sessions']:
        session_id = session['session_id']
        
        # Concatenate all turns into continuous text
        full_text = []
        for turn in session['turns']:
            role = turn['role'].upper()
            content = turn['content']
            full_text.append(f"{role}: {content}")
        
        session_text = '\n\n'.join(full_text)
        session_chars = len(session_text)
        
        # Sliding window chunking
        start_char = 0
        chunk_idx = 0
        
        while start_char < session_chars:
            # Calculate chunk boundaries in chars (approximation)
            chunk_chars = chunk_size * 4  # ~4 chars per token
            end_char = min(start_char + chunk_chars, session_chars)
            
            chunk_text = session_text[start_char:end_char]
            actual_tokens = estimate_tokens(chunk_text)
            
            if actual_tokens > 50:  # Skip very small chunks
                chunks.append({
                    'id': f"{session_id}_chunk{chunk_idx}",
                    'text': chunk_text,
                    'metadata': {
                        'session_id': session_id,
                        'chunk_size': chunk_size,
                        'overlap': overlap_pct,
                        'tokens': actual_tokens
                    }
                })
                chunk_idx += 1
            
            # Move to next chunk with stride
            start_char += stride * 4
            
            # Prevent infinite loop
            if stride <= 0:
                break
    
    return chunks


def build_temp_index(chunks: List[Dict], embed_model: SentenceTransformer) -> chromadb.Collection:
    """Build a temporary ChromaDB index for this configuration."""
    temp_dir = tempfile.mkdtemp(prefix="chroma_temp_")
    client = chromadb.PersistentClient(path=temp_dir)
    
    collection = client.create_collection(
        name="temp_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Batch insert
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        ids = [c['id'] for c in batch]
        texts = [c['text'] for c in batch]
        metadatas = [c['metadata'] for c in batch]
        
        embeddings = embed_model.encode(texts, show_progress_bar=False)
        
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
    
    return collection, temp_dir


def evaluate_retrieval(collection: chromadb.Collection, 
                       embed_model: SentenceTransformer,
                       queries: List[Dict],
                       k: int = 3) -> Dict:
    """
    Evaluate retrieval quality for given queries.
    
    Metrics:
    - Average relevance score (cosine similarity)
    - Average chunk length
    - Retrieval diversity (unique sessions in top-k)
    """
    total_relevance = 0
    total_length = 0
    unique_sessions = 0
    
    for query_info in queries:
        question = query_info["question"]
        
        # Retrieve
        query_emb = embed_model.encode([question])[0].tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=k
        )
        
        # Calculate metrics
        if results['distances'] and results['distances'][0]:
            # ChromaDB returns distances (lower is better), convert to similarity
            similarities = [1 - d for d in results['distances'][0]]
            avg_similarity = sum(similarities) / len(similarities)
            total_relevance += avg_similarity
        
        # Avg chunk length
        if results['documents'] and results['documents'][0]:
            avg_len = sum(len(doc) for doc in results['documents'][0]) / len(results['documents'][0])
            total_length += avg_len
            
            # Count unique sessions
            if results['metadatas'] and results['metadatas'][0]:
                sessions = set(meta['session_id'] for meta in results['metadatas'][0])
                unique_sessions += len(sessions)
    
    n_queries = len(queries)
    
    return {
        'avg_relevance': round(total_relevance / n_queries, 4),
        'avg_chunk_chars': round(total_length / n_queries, 1),
        'avg_unique_sessions': round(unique_sessions / n_queries, 2)
    }


def run_experiment():
    """Run full chunk optimization experiment."""
    print("=" * 80)
    print("CHUNK OPTIMIZATION EXPERIMENT")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Chunk sizes: {CHUNK_SIZES} tokens")
    print(f"Overlaps: {[f'{o*100:.0f}%' for o in OVERLAPS]}")
    print(f"Queries: {len(QUERIES)}")
    print()
    
    # Load data and model
    print("Loading conversations and embedding model...")
    data = load_conversations()
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✓ Loaded {data['total_sessions']} sessions, {data['total_turns']} turns")
    print()
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "chunk_sizes_tested": CHUNK_SIZES,
            "overlaps_tested": OVERLAPS,
            "queries": len(QUERIES),
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "configurations": []
    }
    
    total_configs = len(CHUNK_SIZES) * len(OVERLAPS)
    config_num = 0
    
    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            config_num += 1
            print(f"[{config_num}/{total_configs}] Testing chunk_size={chunk_size}, overlap={overlap*100:.0f}%")
            
            start_time = time.time()
            
            # Create chunks
            print(f"  Creating chunks...", end=" ", flush=True)
            chunks = chunk_by_tokens(data, chunk_size, overlap)
            print(f"✓ {len(chunks)} chunks")
            
            # Build index
            print(f"  Building index...", end=" ", flush=True)
            collection, temp_dir = build_temp_index(chunks, embed_model)
            print(f"✓")
            
            # Evaluate
            print(f"  Evaluating retrieval...", end=" ", flush=True)
            metrics = evaluate_retrieval(collection, embed_model, QUERIES)
            print(f"✓")
            
            elapsed = time.time() - start_time
            
            config_result = {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunk_count": len(chunks),
                "metrics": metrics,
                "time_seconds": round(elapsed, 2)
            }
            
            results["configurations"].append(config_result)
            
            print(f"  Results: relevance={metrics['avg_relevance']:.4f}, "
                  f"avg_chars={metrics['avg_chunk_chars']:.0f}, "
                  f"diversity={metrics['avg_unique_sessions']:.2f}")
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            print()
    
    # Save results
    output_file = Path(__file__).parent / "chunk_optimization_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 80)
    print(f"✅ COMPLETE! Results saved to {output_file}")
    print("=" * 80)
    
    # Find best configuration
    print("\nTOP 5 CONFIGURATIONS (by avg_relevance):")
    sorted_configs = sorted(results["configurations"], 
                           key=lambda x: x["metrics"]["avg_relevance"],
                           reverse=True)
    
    for i, config in enumerate(sorted_configs[:5], 1):
        print(f"\n{i}. chunk_size={config['chunk_size']}, overlap={config['overlap']*100:.0f}%")
        print(f"   Chunks: {config['chunk_count']}")
        print(f"   Relevance: {config['metrics']['avg_relevance']:.4f}")
        print(f"   Avg chars: {config['metrics']['avg_chunk_chars']:.0f}")
        print(f"   Diversity: {config['metrics']['avg_unique_sessions']:.2f}")
    
    return results


if __name__ == '__main__':
    results = run_experiment()
