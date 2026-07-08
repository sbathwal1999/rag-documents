# RAG Study Questions

> **Companion documents:** [`study-guide.md`](./study-guide.md) for the structured walkthrough; [`qa-deep-dive.md`](./qa-deep-dive.md) for in-depth answers to every question below.

## RAG Basics

- [ ] **1.** What is RAG?
- [ ] **2.** What problems in a standalone LLM (hallucination, outdated knowledge, no access to private/proprietary data) does RAG solve?
- [ ] **3.** Why do we need RAG instead of just using a bigger or more frequently retrained model?
- [ ] **4a.** How is RAG different from fine-tuning? 
- [ ] **4b.** When would you choose RAG over fine-tuning, fine-tuning over RAG, or use both together? 
- [ ] **5.** How is RAG different from just using a long-context LLM and stuffing all documents into the prompt? 
- [ ] **6.** What are the main limitations and challenges of RAG? 
- [ ] **7a.** What are the different "generations" of RAG — Naive RAG, Advanced RAG, and Modular RAG? 
- [ ] **7b.** What distinguishes each generation from the others? 

## RAG Architecture

- [ ] **8a.** What are the 2 major phases of RAG?
- [ ] **8b.** Explain each phase, along with its components.
- [ ] **9.** What's the difference between the indexing (offline/build-time) pipeline and the retrieval-and-generation (online/runtime) pipeline? 
- [ ] **10.** In a RAG pipeline, what are the different stages?
- [ ] **11.** Explain how RAG works end-to-end, step by step.

## Document Loading

- [ ] **12a.** What are loaders?
- [ ] **12b.** What are the different types of document sources/formats a RAG system might need to load (PDF, HTML, Word, Markdown, databases, emails, etc.)?
- [ ] **12c.** Why are loaders needed?
- [ ] **13.** What are the most common open-source tools for loading PDFs?
- [ ] **14a.** Is there an open-source tool that extracts a document's hierarchy/structure (headers, sections, paragraphs)?
- [ ] **14b.** Can it also detect whether a given page has a table or an image?
- [ ] **15a.** How do you handle tables in a PDF?
- [ ] **15b.** How do you handle a table that spans multiple pages?
- [ ] **16.** How do you handle images in a PDF?
- [ ] **17.** How do you handle scanned or image-only PDFs (OCR)? 
- [ ] **18.** What data cleaning/preprocessing is typically done after loading and before chunking (removing headers/footers/boilerplate, de-duplication, normalization)? 
- [ ] **19a.** What document-level metadata should you extract during loading (title, author, date, source)? 
- [ ] **19b.** Why does this metadata matter later in the pipeline? 

## Chunking

- [ ] **20a.** What is chunking?
- [ ] **20b.** Why is chunking needed?
- [ ] **21.** What is the trade-off of chunking?
- [ ] **22a.** What are chunk size and chunk overlap? 
- [ ] **22b.** How do you choose good values for them? 
- [ ] **23a.** What are the different chunking strategies, from basic to advanced?
- [ ] **23b.** Explain each strategy in depth, with examples.
- [ ] **24a.** What is "contextual retrieval" (prepending short LLM-generated context to each chunk before embedding it)? 
- [ ] **24b.** What problem does it solve? 
- [ ] **25.** What are the best practices for chunking?

## Embeddings

- [ ] **26a.** What is an embedding?
- [ ] **26b.** Why is embedding needed?
- [ ] **27.** What are the different types of embeddings (sparse vs. dense)?
- [ ] **28a.** Explain sparse embeddings.
- [ ] **28b.** What techniques exist for generating sparse embeddings?
- [ ] **28c.** Which technique is most commonly used?
- [ ] **29a.** Explain dense embeddings.
- [ ] **29b.** How are dense embeddings created?
- [ ] **30.** At a high level, how is an embedding model trained (e.g., contrastive learning on similar/dissimilar pairs)? 
- [ ] **31.** Explain hybrid embeddings — models that output both sparse and dense vectors together (e.g., BGE-M3) — with examples. *(Not to be confused with hybrid search, covered later, which combines two separate retrieval methods.)*
- [ ] **32.** How do you choose an embedding model for a RAG system (benchmarks like MTEB, dimensionality, domain fit, cost/latency)? 
- [ ] **33.** Is there a meaningful difference between how you embed a query vs. how you embed a document chunk? 
- [ ] **34a.** When would you fine-tune an embedding model on your own domain data instead of using an off-the-shelf one? 
- [ ] **34b.** How would you go about it? 
- [ ] **35.** If you switch to a new embedding model later, do you need to re-embed and re-index your entire knowledge base? Why? 

## Indexing, Vector DBs & Retrieval

- [ ] **36.** Why do we need a vector DB (persistence, size, etc.)?
- [ ] **37a.** What is a vector DB?
- [ ] **37b.** What is it optimized for?
- [ ] **38.** Compare indexing in a vector DB to a table in a SQL DB.
- [ ] **39.** At a high level, how does indexing work in a vector database?
- [ ] **40.** What information is needed to create an index (vectors, dimensionality, metadata schema, distance metric)?
- [ ] **41a.** What is the difference between exact (brute-force/flat) nearest-neighbor search and approximate nearest neighbor (ANN) search? 
- [ ] **41b.** Why is ANN preferred in production? 
- [ ] **42a.** What are the different types of ANN indexes (e.g., HNSW, IVF, LSH, product-quantization-based indexes)?
- [ ] **42b.** Explain each type with examples.
- [ ] **43a.** Which index type is most commonly used?
- [ ] **43b.** Explain it in detail.
- [ ] **44a.** What is HNSW?
- [ ] **44b.** How is the HNSW data structure built in the offline stage, step by step (with a simple example)?
- [ ] **45a.** What is vector quantization (scalar/product quantization)? 
- [ ] **45b.** How does it reduce a vector DB's memory footprint and search latency? 
- [ ] **46a.** What are the different similarity metrics (cosine similarity, dot product, Euclidean/L2 distance)?
- [ ] **46b.** When should you use each one?
- [ ] **47a.** What are the most well-known vector databases (e.g., Pinecone, Weaviate, Milvus, Qdrant, Chroma, pgvector)?
- [ ] **47b.** How do they differ from each other?
- [ ] **47c.** When should you use which one?
- [ ] **48.** What are good open-source vector DBs that can be deployed on-prem via Docker?
- [ ] **49.** How do you handle updates, deletions, or newly added documents in an already-built index — full rebuild, or incremental? 
- [ ] **50.** What is the retrieval process?
- [ ] **51.** How does search happen in a vector DB, step by step, from query to results?
- [ ] **52a.** What are the different retrieval strategies (e.g., multi-query, parent-document, self-query, contextual compression)?
- [ ] **52b.** Explain each one in depth.
- [ ] **53a.** What is BM25? 
- [ ] **53b.** How does it score a document's relevance to a query? 
- [ ] **54a.** What is hybrid search?
- [ ] **54b.** How does it combine dense and sparse search?
- [ ] **55a.** What is RRF (Reciprocal Rank Fusion)?
- [ ] **55b.** How does it work?
- [ ] **55c.** Give an example.
- [ ] **56a.** What is metadata, in the context of a vector DB/document store (e.g., source, page number, date, tags)?
- [ ] **56b.** How is it used for filtering during retrieval?
- [ ] **57a.** What are multi-vector/late-interaction retrieval models (e.g., ColBERT)? 
- [ ] **57b.** How do they differ from standard single-vector dense retrieval? 

## Re-ranking

- [ ] **58a.** What are re-rankers?
- [ ] **58b.** Why are they needed?
- [ ] **59.** What is the architectural difference between a bi-encoder (used for initial retrieval) and a cross-encoder (used for re-ranking)? 
- [ ] **60.** Where does the re-ranker sit in the pipeline (architecture)?
- [ ] **61a.** What are the different types of re-rankers (e.g., cross-encoder models, LLM-based re-rankers, multi-vector/late-interaction re-rankers)?
- [ ] **61b.** Explain each type.
- [ ] **62.** Explain the most popular re-ranker in depth, with an example.
- [ ] **63a.** What are the trade-offs of re-ranking (latency, cost)? 
- [ ] **63b.** How do you decide how many candidates to pass into the re-ranker, and how many to keep afterward? 

## Prompting

- [ ] **64.** What role does prompting play in a RAG system?
- [ ] **65.** What general prompt-engineering techniques (zero-shot, few-shot, chain-of-thought, role-based prompting, etc.) are commonly applied when prompting in a RAG system? 
- [ ] **66.** What should a RAG system prompt typically instruct the model to do (e.g., answer only from the provided context, say "I don't know" if the answer isn't present, cite sources)? 
- [ ] **67.** How do you decide how many retrieved chunks to include in the prompt, given the model's context window and token budget? 
- [ ] **68a.** Does the order of chunks within the prompt matter to the LLM? 
- [ ] **68b.** What is the "lost in the middle" problem? 
- [ ] **68c.** How do you mitigate it? 
- [ ] **69a.** What is a citation, in the context of RAG?
- [ ] **69b.** Why is it needed?
- [ ] **69c.** How do you add citations to a RAG output?
- [ ] **70a.** Explain query reformulation.
- [ ] **70b.** Explain query decomposition.
- [ ] **70c.** Explain how conversation history is handled.
- [ ] **70d.** Explain output formatting.
- [ ] **70e.** Explain HyDE (Hypothetical Document Embeddings).
- [ ] **70f.** Explain knowledge graph augmentation — using a knowledge graph alongside vector retrieval to add structured, relational context. *(For knowledge graphs as the primary retrieval architecture, see GraphRAG under Advanced RAG.)*

## Generation

- [ ] **71.** Explain the generation process.
- [ ] **72a.** What are the different models used for generation?
- [ ] **72b.** What are their key parameters (e.g., temperature, top-p, max tokens)?
- [ ] **72c.** How do these parameters affect RAG output quality? 
- [ ] **73.** What is the difference between extractive and abstractive answer generation in RAG? 
- [ ] **74.** How do you control hallucination?
- [ ] **75a.** How do you optimize cost?
- [ ] **75b.** How do you optimize token consumption?
- [ ] **76.** How does streaming the generated response affect how you handle citations and formatting? 

## Evaluation

- [ ] **77.** How do you evaluate a RAG system?
- [ ] **78.** What's the difference between evaluating the retrieval component vs. evaluating the generation component of a RAG system? 
- [ ] **79a.** What is reference-free evaluation?
- [ ] **79b.** What is reference-based evaluation?
- [ ] **79c.** What's the difference between them?
- [ ] **79d.** What metrics fall under each, and how are they calculated?
- [ ] **80.** What are the different metrics for evaluating RAG?
- [ ] **81.** Explain each metric in depth, with examples and how to calculate it.
- [ ] **82a.** What is the "RAG triad" — context relevance, faithfulness/groundedness, and answer relevance? 
- [ ] **82b.** How do these three metrics relate to each other? 
- [ ] **83a.** What is "LLM-as-a-judge"? 
- [ ] **83b.** How is it used to evaluate RAG outputs? 
- [ ] **84.** How do you build a golden/test dataset for evaluating RAG, especially without existing labeled QA pairs (e.g., synthetic QA generation)? 
- [ ] **85.** What tools or frameworks are commonly used for RAG evaluation (e.g., RAGAS, TruLens, DeepEval, Arize Phoenix)? 
- [ ] **86a.** What is the role of human evaluation in RAG? 
- [ ] **86b.** How does it complement automated/LLM-based evaluation? 

## Advanced & Agentic RAG 

- [ ] **87a.** What is Agentic RAG?
- [ ] **87b.** How does giving the LLM agency over when, what, and how many times to retrieve change the architecture compared to standard RAG?
- [ ] **88a.** What is Self-RAG?
- [ ] **88b.** How does it let the model critique and decide whether to retrieve or revise its own output?
- [ ] **89a.** What is Corrective RAG (CRAG)?
- [ ] **89b.** How does it handle retrieved documents that are irrelevant or low-quality?
- [ ] **90a.** What is Adaptive RAG?
- [ ] **90b.** How does a system decide whether to retrieve at all for a given query, versus answering directly?
- [ ] **91a.** What is GraphRAG (building the knowledge base as a graph and retrieving via graph traversal/community summaries)?
- [ ] **91b.** When is it preferred over standard vector-based retrieval?
- [ ] **92.** How does RAG extend to multi-modal data — retrieving and generating over images, audio, or video, not just text?
- [ ] **93.** In a multi-turn conversational RAG system, how do you manage memory and context across turns, as opposed to within a single query?

## Production, Security & Monitoring 

- [ ] **94a.** What are the main sources of latency in a RAG pipeline?
- [ ] **94b.** How do you optimize end-to-end response time?
- [ ] **95a.** What is semantic caching?
- [ ] **95b.** How can it reduce cost and latency for repeated or similar queries?
- [ ] **96.** How do you monitor and observe a RAG system in production (logging retrieved chunks, tracking metrics over time, detecting drift)?
- [ ] **97a.** What is a prompt-injection attack via retrieved documents (malicious instructions hidden inside a document in the knowledge base)?
- [ ] **97b.** How do you defend against it?
- [ ] **98.** In a multi-tenant RAG system, how do you ensure a user only retrieves documents they're authorized to access?
- [ ] **99.** What are the key considerations for scaling a RAG system as the knowledge base and query volume grow?
