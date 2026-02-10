# Vector Database Comparison for Personality Memory

**Context:** Project Molting Phase 1-2 gap analysis  
**Use case:** Storing ~1000 conversation chunks with embeddings for RAG-based personality memory  
**Date:** 2026-02-10

## Requirements

Our specific needs for personality memory:
- **Scale:** Small (~1000-5000 chunks, growing slowly)
- **Query pattern:** Semantic search with metadata filtering (source_type, timestamp)
- **Embedding model:** all-MiniLM-L6-v2 (384-dim vectors)
- **Integration:** Local-first, Python-friendly, minimal dependencies
- **Performance:** Sub-50ms retrieval latency acceptable
- **Persistence:** Must survive restarts (not in-memory only)

## Comparison Matrix

| Feature | ChromaDB | FAISS | PGVector |
|---------|----------|-------|----------|
| **Type** | Document DB | Vector index | Postgres extension |
| **Setup complexity** | ⭐⭐⭐ Easy (pip install) | ⭐⭐⭐ Easy (pip install) | ⭐⭐ Moderate (needs Postgres) |
| **Persistence** | Built-in (SQLite) | Manual (save/load index) | Built-in (Postgres) |
| **Metadata filtering** | ✅ Native support | ❌ Requires custom layer | ✅ Native SQL queries |
| **Index types** | HNSW (cosine/L2) | Multiple (IVF, HNSW, Flat) | IVFFlat, HNSW (pg 0.7.0+) |
| **Query speed (1k vecs)** | ~10-30ms | ~1-5ms (in-memory) | ~20-50ms |
| **Scalability** | Good (<10M vecs) | Excellent (billions) | Good (<10M vecs) |
| **Production-ready** | ✅ Yes | ⚠️ Requires wrapper | ✅ Yes (if using Postgres) |
| **Dependencies** | Lightweight | Minimal (numpy) | PostgreSQL server |
| **Memory footprint** | Low | Medium (index in RAM) | Low (DB-managed) |
| **Document storage** | ✅ Stores text + metadata | ❌ Vectors only | ✅ Full relational DB |
| **Similarity metrics** | Cosine, L2, IP | Cosine, L2, IP | Cosine, L2, IP |
| **Batch operations** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Cloud deployment** | ✅ Easy (Docker) | ⚠️ DIY | ✅ Standard Postgres |

## Deep Dive

### 1. ChromaDB

**Pros:**
- ✅ **Zero-config persistence** — just point to a directory, done
- ✅ **Document-centric** — stores embeddings + text + metadata together
- ✅ **Metadata filtering** — `where={"source_type": "research"}` works natively
- ✅ **Good DX** — intuitive API, great for prototyping
- ✅ **Small scale optimized** — perfect for <1M vectors
- ✅ **Active development** — well-maintained, growing ecosystem

**Cons:**
- ⚠️ **Performance ceiling** — slower than FAISS at large scale
- ⚠️ **Less flexible** — fixed architecture (HNSW + SQLite)
- ⚠️ **Embedding model coupling** — must manage embedding generation separately

**Best for:** Prototyping, small-to-medium RAG apps, when you want metadata filtering without SQL.

**Our verdict:** ⭐⭐⭐⭐⭐ **Perfect fit for Molting Phase 2-3.** Already using it, works well, no need to change.

---

### 2. FAISS (Facebook AI Similarity Search)

**Pros:**
- ✅ **Blazing fast** — optimized for billion-scale search
- ✅ **Flexible indexing** — IVF, HNSW, PQ (product quantization), etc.
- ✅ **GPU support** — can offload to CUDA for massive speedups
- ✅ **Mature & battle-tested** — used in production at Meta, OpenAI, etc.
- ✅ **Low-level control** — tune index parameters for exact needs

**Cons:**
- ❌ **No built-in persistence** — must manually save/load `.index` files
- ❌ **No metadata storage** — vectors only, need separate DB for metadata
- ❌ **No metadata filtering** — must pre-filter IDs, then search
- ⚠️ **Requires wrapper layer** — not a complete solution, more like a library
- ⚠️ **Overkill for small scale** — complexity not justified for <10k vectors

**Architecture pattern (if using FAISS):**
```
FAISS index (vectors) + SQLite (metadata) + custom glue code
```

**Best for:** Large-scale production systems (>10M vectors), when raw speed is critical, when you have engineering bandwidth for custom integration.

**Our verdict:** ⭐⭐ **Overkill for Molting.** Too low-level, no metadata filtering, requires custom persistence layer. Only consider if scaling to millions of chunks.

---

### 3. PGVector (Postgres Extension)

**Pros:**
- ✅ **Full relational DB** — vectors + metadata + joins + transactions
- ✅ **SQL queries** — `WHERE source_type = 'research' ORDER BY embedding <-> query LIMIT 5`
- ✅ **Production-grade** — Postgres reliability, backups, replication
- ✅ **Flexible schema** — add columns, indexes, constraints as needed
- ✅ **Ecosystem integration** — works with existing Postgres tooling (pg_dump, Hasura, etc.)
- ✅ **HNSW support** — fast approximate search (pg 0.7.0+)

**Cons:**
- ⚠️ **Requires Postgres** — must run/manage a database server
- ⚠️ **Setup overhead** — install extension, configure, manage connections
- ⚠️ **Slightly slower** — 20-50ms vs FAISS's 1-5ms (but fine for our scale)
- ⚠️ **Vector-specific features lag** — ChromaDB/FAISS more specialized

**Best for:** When already using Postgres, when you need relational queries + vectors, enterprise deployments.

**Our verdict:** ⭐⭐⭐ **Good, but overkill for Molting.** If we were building a multi-user system with users, sessions, permissions, etc. — Postgres would make sense. For a single-user research tool, ChromaDB is simpler.

---

## Recommendation for Molting

**🏆 Stick with ChromaDB**

**Rationale:**
1. **Already working** — 257 chunks indexed, retrieval validated in Phase 2
2. **Right-sized** — perfect for 1k-10k chunk scale, our use case
3. **Metadata filtering** — enables source-type filtering (just added in Part A)
4. **Simple deployment** — no external services, just a directory
5. **Good enough performance** — 10-30ms retrieval is fine for interactive queries
6. **Future-proof** — if we scale to 100k+ chunks, ChromaDB will still work (just slower than FAISS)

**When to reconsider:**
- ❌ **NOT** if we hit 10k chunks — ChromaDB handles this fine
- ⚠️ **MAYBE** if we exceed 100k chunks — consider FAISS then
- ⚠️ **MAYBE** if we need <5ms latency — FAISS would help
- ✅ **YES** if we build a multi-user SaaS — then PGVector makes sense (user isolation, relational data)

## Alternative: Hybrid Approach (Future)

If we need both speed AND metadata filtering at scale:

**FAISS (vectors) + DuckDB (metadata)**
- FAISS for fast vector search
- DuckDB for SQL-like metadata filtering (embedded, no server)
- Glue: search FAISS → get IDs → filter in DuckDB

But this is **premature optimization** for Molting Phase 3.

---

## Benchmarking (Simulated)

*Hypothetical performance on our hardware (31GB RAM, RTX 3050):*

| Operation | ChromaDB | FAISS | PGVector |
|-----------|----------|-------|----------|
| Index 1000 chunks | ~2s | ~0.5s | ~3s |
| Query (top-5) | ~15ms | ~2ms | ~30ms |
| Metadata filter query | ~20ms | ~50ms (pre-filter) | ~35ms |
| Disk usage (1000 chunks) | ~5MB | ~3MB (index) + metadata DB | ~10MB (Postgres overhead) |

*Note: ChromaDB latency acceptable for interactive use. FAISS faster but requires more code.*

---

## Conclusion

**For Molting Phase 3:**
- ✅ **Keep ChromaDB** — it's working, simple, and sufficient
- ✅ **Focus effort on fine-tuning** — that's the real Phase 3 goal
- ✅ **Optimize chunking (Part A)** — better retrieval quality > switching DBs
- ⏭️ **Defer DB migration** — only revisit if hitting clear performance walls

**Engineering principle:** Don't optimize what's not broken. ChromaDB is not our bottleneck — retrieval quality (chunk optimization) and model personality (fine-tuning) are.

---

## References

- [ChromaDB](https://github.com/chroma-core/chroma) — embeddings database
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [PGVector](https://github.com/pgvector/pgvector) — Postgres extension
- [LlamaIndex Vector Store comparison](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/)
- [Pinecone Vector DB benchmarks](https://www.pinecone.io/learn/vector-database-comparison/)
