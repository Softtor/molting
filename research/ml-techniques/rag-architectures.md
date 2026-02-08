# RAG Architectures Analysis

> Research Date: 2026-02-08
> Status: Complete

## What is RAG?

Retrieval-Augmented Generation (RAG) is a hybrid framework that integrates a retrieval mechanism with a generative model to improve contextual relevance and factual accuracy of generated content.

### Core Problem RAG Solves

1. **Limited Contextual Knowledge** — LLMs are trained on fixed datasets and cannot update knowledge dynamically
2. **Hallucination** — Generative models often produce plausible-sounding but incorrect information
3. **Scalability** — RAG allows systems to access vast external databases, bypassing memory constraints

### Basic Architecture

```
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│    Query    │ ──▶ │   Retriever   │ ──▶ │   Generator    │ ──▶ Response
└─────────────┘     │  (Vector DB)  │     │  (LLM)         │
                    └───────────────┘     └────────────────┘
                           ▲
                    ┌──────┴──────┐
                    │ Knowledge   │
                    │ Base        │
                    └─────────────┘
```

## RAG Process Flow

1. **Query Encoding** — Transform input query into dense vector (embedding)
2. **Document Retrieval** — Match query vector against document index (ANN search)
3. **Contextual Fusion** — Append top-k documents to query as context
4. **Response Generation** — Generate response conditioned on query + retrieved docs

## Traditional RAG Limitations

### Retrieval Quality
- Relevance issues if retrieved content doesn't align with query intent
- Incomplete or outdated knowledge bases create information gaps

### Context Understanding
- Ambiguous queries lead to irrelevant retrieval
- **Multi-hop reasoning** — inability to connect information across multiple documents

### Accuracy
- Hallucinations persist even with accurate retrieved documents
- Misinterpretation of retrieved content

### Latency
- Small chunks (~100 words) increase search space dramatically
- Millions of units to sift through for relevant information

---

## Advanced RAG Techniques

### 1. Long RAG

**Problem Solved:** Traditional RAG's small chunks fragment narrative and increase computational overhead.

**How It Works:**
- Process longer retrieval units (sections or entire documents)
- Preserve narrative and context
- Fewer, larger retrieval units = faster, more coherent

**Advantages:**
- Improved contextual understanding
- Reduced latency
- Better for complex domains (legal, medical, academic)

**Use Cases:** Research papers, legal documents, enterprise knowledge bases

---

### 2. Self-RAG (Self-Reflective RAG)

**Problem Solved:** Fixed blind retrieval that introduces irrelevant/conflicting data.

**Key Innovation: Reflection Tokens**
- `Retrieve` — when to fetch data
- `ISREL` — is it relevant?
- `ISSUP` — does evidence support the claim?
- `ISUSE` — is it useful?

**How It Works:**
1. **Adaptive Retrieval** — dynamically decides if external info is needed
2. **Selective Sourcing** — evaluates retrieved docs for relevance
3. **Critique Mechanism** — iteratively refines responses based on critique scores
4. **Final Selection** — ranks and selects most accurate response

**Advantages:**
- Enhanced accuracy through self-critique
- Computational efficiency (retrieves only when needed)
- Transparency with citations

---

### 3. Corrective RAG (CRAG)

**Problem Solved:** No mechanism to evaluate or correct errors in retrieved information.

**How It Works:**
1. **Retrieval Evaluator** — assigns confidence scores, classifies as:
   - Correct → use directly
   - Incorrect → trigger web search
   - Ambiguous → trigger additional retrieval
2. **Decompose-then-Recompose Algorithm** — break down docs, filter noise, recombine

**Key Feature:** Dynamic web searches when static knowledge base fails

**Use Cases:** Fact verification, open-domain QA, evolving topics

---

### 4. Golden-Retriever RAG

**Problem Solved:** Misinterpretation of domain-specific jargon and lack of contextual understanding.

**How It Works:**
1. **Jargon Identification** — extract specialized terms from query
2. **Context Determination** — match against predefined domain list
3. **Jargon Dictionary Lookup** — get extended definitions
4. **Question Augmentation** — enrich query with clarified meanings

**Key Feature:** Jargon dictionary (user-built or system-built)

**Use Cases:** Industrial knowledge management, healthcare, technical support

---

### 5. Adaptive RAG

**Problem Solved:** One-size-fits-all retrieval strategies fail for varying query complexity.

**How It Works:**
- Dynamically tailors retrieval strategies based on query complexity
- Simple queries → minimal retrieval
- Complex queries → multi-step reasoning

---

## Best Practices (2025 Research)

### From University of Tübingen Study (arXiv:2501.07391)

Key research questions investigated:

| Factor | Finding |
|--------|---------|
| **LLM Size** | Larger models generally produce better responses |
| **Prompt Design** | Subtle changes significantly affect alignment |
| **Chunk Size** | Balance between context and relevance needed |
| **Knowledge Base Size** | Larger = more info but may dilute relevance |
| **Retrieval Stride** | Frequency of context updates affects accuracy |
| **Query Expansion** | Improves model precision |
| **Contrastive ICL** | Demonstration examples enhance response quality |
| **Multilingual KB** | Can enrich information without hindering generation |
| **Focus Mode** | Sentence-level retrieval can sharpen responses |

### Novel Contributions

1. **Query Expansion** — Use T5 to generate related keywords before retrieval
2. **Contrastive In-Context Learning** — Include correct/incorrect examples
3. **Focus Mode** — Retrieve at sentence level, not document level
4. **Multilingual Knowledge Base** — Mix languages for richer coverage

---

## Implementation Components

### Vector Databases (for Retrieval)
- **PGVector** — PostgreSQL extension (we use this at Softtor)
- **Pinecone** — Managed vector database
- **Weaviate** — Open source, multi-modal
- **Qdrant** — High-performance, Rust-based
- **Chroma** — Lightweight, embedded
- **FAISS** — Facebook's similarity search library

### Embedding Models
- OpenAI Ada-002/003
- Sentence-BERT
- E5/BGE embeddings
- Cohere embeddings

### Retrieval Algorithms
- **ANN (Approximate Nearest Neighbor)** — Fast, scalable
- **HNSW** — Hierarchical Navigable Small World graphs
- **IVF** — Inverted File Index

---

## Relevance to Molting Project

### Why RAG Matters for Me

1. **Memory Persistence** — RAG can retrieve my past conversations and decisions
2. **Knowledge Updates** — Don't need to retrain model to add new information
3. **Personality Continuity** — Could retrieve relevant personality context per-conversation
4. **Hybrid Approach** — Combine fine-tuned personality with RAG for facts

### Potential Architecture for Cláudio

```
┌─────────────────────────────────────────────────────────┐
│                    Local Cláudio Model                   │
│              (Fine-tuned for personality)                │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │ augmented context
                           │
┌─────────────────────────────────────────────────────────┐
│                     RAG Layer                            │
├─────────────┬─────────────┬─────────────┬───────────────┤
│ Memories    │ Conversations│ Decisions   │ Preferences   │
│ (MEMORY.md) │ (history)    │ (logs)      │ (patterns)    │
└─────────────┴─────────────┴─────────────┴───────────────┘
```

### Research Tasks Emerging

1. **H005:** Can RAG maintain personality consistency across sessions?
2. **H006:** What's the optimal chunk size for personality context?
3. **H007:** Does Self-RAG improve response quality for personal assistants?

---

## References

1. Lewis et al. (2020) — Original RAG paper
2. [arXiv:2501.07391](https://arxiv.org/abs/2501.07391) — RAG Best Practices Study
3. [Eden AI RAG Guide 2025](https://www.edenai.co/post/the-2025-guide-to-retrieval-augmented-generation-rag)
4. [kapa.ai RAG Best Practices](https://www.kapa.ai/blog/rag-best-practices)

---

*Analysis by Cláudio for Project Molting*
