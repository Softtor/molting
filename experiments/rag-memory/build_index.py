#!/usr/bin/env python3
"""
Build ChromaDB index from extracted conversations.
Creates embeddings using sentence-transformers.
"""

import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CONVERSATIONS_FILE = Path(__file__).parent / "conversations.json"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

def load_conversations():
    """Load extracted conversations."""
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def infer_source_type(text: str) -> str:
    """
    Infer source type from conversation content.
    Types: research, experiment, hypothesis, log, general
    """
    text_lower = text.lower()
    
    # Research indicators
    research_keywords = ['paper', 'study', 'research', 'survey', 'literature', 'academic', 'publication']
    if any(kw in text_lower for kw in research_keywords):
        return 'research'
    
    # Experiment indicators
    experiment_keywords = ['experiment', 'test', 'benchmark', 'result', 'performance', 'metric', 'comparison']
    if any(kw in text_lower for kw in experiment_keywords):
        return 'experiment'
    
    # Hypothesis indicators
    hypothesis_keywords = ['hypothesis', 'theory', 'predict', 'expect', 'assume', 'believe', 'might work']
    if any(kw in text_lower for kw in hypothesis_keywords):
        return 'hypothesis'
    
    # Log indicators (technical/implementation)
    log_keywords = ['implemented', 'fixed', 'bug', 'error', 'commit', 'deploy', 'merge', 'push']
    if any(kw in text_lower for kw in log_keywords):
        return 'log'
    
    return 'general'


def chunk_conversations(data, max_chunk_len=1000):
    """
    Create chunks from conversations.
    Each chunk is a user-assistant exchange with context.
    Now includes source_type metadata for filtering.
    """
    chunks = []
    
    for session in data['sessions']:
        session_id = session['session_id']
        turns = session['turns']
        
        # Group into user-assistant pairs
        i = 0
        while i < len(turns):
            chunk_text = []
            chunk_turns = []
            
            # Collect a conversation window (up to 4 turns)
            window_start = i
            while i < len(turns) and i < window_start + 4:
                turn = turns[i]
                role = turn['role']
                content = turn['content'][:500]  # Truncate long messages
                chunk_text.append(f"{role.upper()}: {content}")
                chunk_turns.append(turn)
                i += 1
            
            if chunk_text:
                full_text = '\n\n'.join(chunk_text)
                
                # Get timestamp from first turn
                timestamp = chunk_turns[0].get('timestamp', '')
                
                # Infer source type from content
                source_type = infer_source_type(full_text)
                
                chunks.append({
                    'id': f"{session_id}_{window_start}",
                    'text': full_text,
                    'metadata': {
                        'session_id': session_id,
                        'timestamp': timestamp,
                        'turn_count': len(chunk_turns),
                        'source_type': source_type
                    }
                })
    
    return chunks

def main():
    print("Loading conversations...")
    data = load_conversations()
    print(f"Loaded {data['total_sessions']} sessions, {data['total_turns']} turns")
    
    print("\nChunking conversations...")
    chunks = chunk_conversations(data)
    print(f"Created {len(chunks)} chunks")
    
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Delete existing collection if exists
    try:
        client.delete_collection("conversations")
    except:
        pass
    
    collection = client.create_collection(
        name="conversations",
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"\nEmbedding and indexing {len(chunks)} chunks...")
    batch_size = 100
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        ids = [c['id'] for c in batch]
        texts = [c['text'] for c in batch]
        metadatas = [c['metadata'] for c in batch]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"  Indexed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
    
    print(f"\n✅ Index built successfully!")
    print(f"   Location: {CHROMA_DIR}")
    print(f"   Total chunks: {collection.count()}")

if __name__ == '__main__':
    main()
