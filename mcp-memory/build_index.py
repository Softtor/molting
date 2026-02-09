#!/usr/bin/env python3
"""Build ChromaDB index from all Molting research knowledge."""

import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

MOLTING_ROOT = Path(__file__).parent.parent
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# Directories to index
INDEX_PATHS = [
    "experiments",
    "research", 
    "docs",
    "README.md",
    "DIRECTIVES.md",
]

# File extensions to index
EXTENSIONS = {".md", ".txt"}

# Directories to skip
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "chroma_db", "node_modules", ".cache"}

def get_files_to_index():
    """Get all files that should be indexed."""
    files = []
    for path in INDEX_PATHS:
        full_path = MOLTING_ROOT / path
        if full_path.is_file():
            files.append(full_path)
        elif full_path.is_dir():
            for ext in EXTENSIONS:
                for f in full_path.rglob(f"*{ext}"):
                    # Skip files in excluded directories
                    if not any(skip in f.parts for skip in SKIP_DIRS):
                        files.append(f)
    return files

def chunk_file(filepath: Path, chunk_size: int = 1000, overlap: int = 200):
    """Split file into overlapping chunks."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except:
        return []
    
    # Skip empty or very small files
    if len(content) < 100:
        return []
    
    # Create chunks
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        
        # Add metadata
        rel_path = filepath.relative_to(MOLTING_ROOT)
        chunks.append({
            "text": chunk,
            "source": str(rel_path),
            "start_char": start,
        })
        
        start = end - overlap
        if end >= len(content):
            break
    
    return chunks

def main():
    print("🦞 Building Molting Memory Index...")
    
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Delete existing collection if exists
    try:
        client.delete_collection("molting_knowledge")
    except:
        pass
    
    collection = client.create_collection(
        name="molting_knowledge",
        metadata={"description": "Molting research knowledge base"}
    )
    
    # Get files to index
    files = get_files_to_index()
    print(f"Found {len(files)} files to index")
    
    # Process files
    all_chunks = []
    for filepath in files:
        chunks = chunk_file(filepath)
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    if not all_chunks:
        print("No content to index!")
        return
    
    # Generate embeddings
    print("Generating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Add to collection
    print("Adding to ChromaDB...")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": c["source"], "start": c["start_char"]} for c in all_chunks]
    )
    
    print(f"✅ Indexed {len(all_chunks)} chunks from {len(files)} files")
    print(f"📁 Index saved to {CHROMA_DIR}")

if __name__ == "__main__":
    main()
