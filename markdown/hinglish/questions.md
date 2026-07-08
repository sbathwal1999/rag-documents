# RAG Study Questions (Hinglish)

> **Companion documents:** [`study-guide.md`](./study-guide.md) — structured Hinglish walkthrough; [`qa-deep-dive.md`](./qa-deep-dive.md) — har question ka detailed answer.

## RAG Basics

- [ ] **1.** RAG kya hai?
- [ ] **2.** Standalone LLM ke jo problems hain (hallucination, purani knowledge, private/proprietary data tak access na hona) — inn sabko RAG kaise solve karta hai?
- [ ] **3.** RAG ki jagah hum ek bigger ya frequently retrained model kyun nahi use karte?
- [ ] **4a.** RAG aur fine-tuning me kya difference hai?
- [ ] **4b.** Kab RAG choose karoge, kab fine-tuning, aur kab dono ek saath use karoge?
- [ ] **5.** RAG aur long-context LLM me kya farak hai — jaha hum saare documents prompt me hi daal dete hain?
- [ ] **6.** RAG ki main limitations aur challenges kya hain?
- [ ] **7a.** RAG ki alag-alag "generations" kya hain — Naive RAG, Advanced RAG, aur Modular RAG?
- [ ] **7b.** Har generation ko doosri se kya cheez distinguish karti hai?

## RAG Architecture

- [ ] **8a.** RAG ke 2 major phases kya hain?
- [ ] **8b.** Har phase ko uske components ke saath explain karo.
- [ ] **9.** Indexing (offline/build-time) pipeline aur retrieval-and-generation (online/runtime) pipeline me kya difference hai?
- [ ] **10.** RAG pipeline me alag-alag stages kya hain?
- [ ] **11.** End-to-end RAG kaise kaam karta hai, step-by-step samjhao.

## Document Loading

- [ ] **12a.** Loaders kya hote hain?
- [ ] **12b.** RAG system ko kis-kis type ke document sources/formats load karne pad sakte hain (PDF, HTML, Word, Markdown, databases, emails, etc.)?
- [ ] **12c.** Loaders ki zarurat kyun hoti hai?
- [ ] **13.** PDFs load karne ke most common open-source tools kaunse hain?
- [ ] **14a.** Kya koi open-source tool hai jo document ki hierarchy/structure (headers, sections, paragraphs) extract kar sake?
- [ ] **14b.** Kya wo detect bhi kar sakta hai ki kisi page pe table ya image hai?
- [ ] **15a.** PDF me tables ko kaise handle karte ho?
- [ ] **15b.** Multi-page table ko kaise handle karte ho?
- [ ] **16.** PDF me images ko kaise handle karte ho?
- [ ] **17.** Scanned ya image-only PDFs (OCR) ko kaise handle karte ho?
- [ ] **18.** Loading ke baad aur chunking se pehle typically kaunsa data cleaning/preprocessing hota hai (headers/footers/boilerplate hataana, de-duplication, normalization)?
- [ ] **19a.** Loading ke time pe kaunsa document-level metadata extract karna chahiye (title, author, date, source)?
- [ ] **19b.** Ye metadata pipeline me aage kyun important hoga?

## Chunking

- [ ] **20a.** Chunking kya hai?
- [ ] **20b.** Chunking kyun zaruri hai?
- [ ] **21.** Chunking ka trade-off kya hai?
- [ ] **22a.** Chunk size aur chunk overlap kya hote hain?
- [ ] **22b.** Inke liye achhi values kaise choose karoge?
- [ ] **23a.** Basic se advanced tak alag-alag chunking strategies kaunsi hain?
- [ ] **23b.** Har strategy ko examples ke saath depth me explain karo.
- [ ] **24a.** "Contextual retrieval" kya hota hai (chunk ko embed karne se pehle LLM se short context prepend karna)?
- [ ] **24b.** Ye kaunsi problem solve karta hai?
- [ ] **25.** Chunking ke best practices kya hain?

## Embeddings

- [ ] **26a.** Embedding kya hai?
- [ ] **26b.** Embedding ki zarurat kyun hai?
- [ ] **27.** Embeddings ke alag-alag types (sparse vs. dense) kaunse hain?
- [ ] **28a.** Sparse embeddings explain karo.
- [ ] **28b.** Sparse embeddings generate karne ke liye kaunsi techniques hain?
- [ ] **28c.** Sabse commonly kaunsi technique use hoti hai?
- [ ] **29a.** Dense embeddings explain karo.
- [ ] **29b.** Dense embeddings kaise create hote hain?
- [ ] **30.** High level pe, embedding model kaise train hota hai (jaise contrastive learning on similar/dissimilar pairs)?
- [ ] **31.** Hybrid embeddings — jo models sparse aur dense dono vectors ek saath output karte hain (jaise BGE-M3) — examples ke saath explain karo. *(Ye hybrid search se alag hai — wo baad me aayega, jo do alag retrieval methods combine karta hai.)*
- [ ] **32.** RAG system ke liye embedding model kaise choose karoge (MTEB jaise benchmarks, dimensionality, domain fit, cost/latency)?
- [ ] **33.** Query embed karne aur document chunk embed karne me koi meaningful difference hai kya?
- [ ] **34a.** Off-the-shelf embedding model chhod ke apne domain data pe kab fine-tune karoge?
- [ ] **34b.** Kaise karoge?
- [ ] **35.** Baad me naya embedding model switch karo toh kya poori knowledge base re-embed aur re-index karni padegi? Kyun?

## Indexing, Vector DBs & Retrieval

- [ ] **36.** Vector DB ki zarurat kyun hoti hai (persistence, size, etc.)?
- [ ] **37a.** Vector DB kya hai?
- [ ] **37b.** Wo kis cheez ke liye optimized hai?
- [ ] **38.** Vector DB me indexing ko SQL DB ki table se compare karo.
- [ ] **39.** High level pe, vector database me indexing kaise kaam karti hai?
- [ ] **40.** Index create karne ke liye kaunsi information chahiye (vectors, dimensionality, metadata schema, distance metric)?
- [ ] **41a.** Exact (brute-force/flat) nearest-neighbor search aur approximate nearest neighbor (ANN) search me kya difference hai?
- [ ] **41b.** Production me ANN kyun preferred hai?
- [ ] **42a.** ANN indexes ke alag-alag types kaunse hain (jaise HNSW, IVF, LSH, product-quantization-based indexes)?
- [ ] **42b.** Har type ko example ke saath explain karo.
- [ ] **43a.** Sabse commonly kaunsa index type use hota hai?
- [ ] **43b.** Detail me explain karo.
- [ ] **44a.** HNSW kya hai?
- [ ] **44b.** Offline stage me HNSW data structure step-by-step kaise build hota hai (simple example ke saath)?
- [ ] **45a.** Vector quantization (scalar/product quantization) kya hai?
- [ ] **45b.** Ye vector DB ka memory footprint aur search latency kaise reduce karta hai?
- [ ] **46a.** Alag-alag similarity metrics kaunsi hain (cosine similarity, dot product, Euclidean/L2 distance)?
- [ ] **46b.** Kaunsi kab use karni chahiye?
- [ ] **47a.** Sabse well-known vector databases kaunse hain (jaise Pinecone, Weaviate, Milvus, Qdrant, Chroma, pgvector)?
- [ ] **47b.** Ek doosre se kaise different hain?
- [ ] **47c.** Kab kaunsi use karni chahiye?
- [ ] **48.** Kaunse achhe open-source vector DBs hain jo Docker ke through on-prem deploy ho sakte hain?
- [ ] **49.** Already-built index me updates, deletions ya newly added documents kaise handle karte ho — full rebuild ya incremental?
- [ ] **50.** Retrieval process kya hai?
- [ ] **51.** Vector DB me search step-by-step kaise hota hai, query se leke results tak?
- [ ] **52a.** Alag-alag retrieval strategies kaunsi hain (jaise multi-query, parent-document, self-query, contextual compression)?
- [ ] **52b.** Har ek ko depth me explain karo.
- [ ] **53a.** BM25 kya hai?
- [ ] **53b.** Ye ek document ki query se relevance kaise score karta hai?
- [ ] **54a.** Hybrid search kya hai?
- [ ] **54b.** Dense aur sparse search ko kaise combine karta hai?
- [ ] **55a.** RRF (Reciprocal Rank Fusion) kya hai?
- [ ] **55b.** Kaise kaam karta hai?
- [ ] **55c.** Example do.
- [ ] **56a.** Vector DB/document store ke context me metadata kya hota hai (jaise source, page number, date, tags)?
- [ ] **56b.** Retrieval ke dauran filtering ke liye kaise use hota hai?
- [ ] **57a.** Multi-vector/late-interaction retrieval models (jaise ColBERT) kya hote hain?
- [ ] **57b.** Standard single-vector dense retrieval se kaise different hain?

## Re-ranking

- [ ] **58a.** Re-rankers kya hote hain?
- [ ] **58b.** Inki zarurat kyun hai?
- [ ] **59.** Bi-encoder (initial retrieval ke liye) aur cross-encoder (re-ranking ke liye) me architectural difference kya hai?
- [ ] **60.** Pipeline (architecture) me re-ranker kahan sit karta hai?
- [ ] **61a.** Alag-alag re-rankers ke types kaunse hain (jaise cross-encoder models, LLM-based re-rankers, multi-vector/late-interaction re-rankers)?
- [ ] **61b.** Har type ko explain karo.
- [ ] **62.** Sabse popular re-ranker ko depth me example ke saath samjhao.
- [ ] **63a.** Re-ranking ke trade-offs kya hain (latency, cost)?
- [ ] **63b.** Kitne candidates re-ranker me daalne hain aur kitne rakhne hain — kaise decide karoge?

## Prompting

- [ ] **64.** RAG system me prompting ka role kya hai?
- [ ] **65.** RAG system me prompting me commonly kaunsi general prompt-engineering techniques (zero-shot, few-shot, chain-of-thought, role-based prompting, etc.) apply hoti hain?
- [ ] **66.** RAG system prompt ko typically model ko kya karne ke liye instruct karna chahiye (jaise sirf provided context se answer, "I don't know" bolna agar answer na ho, sources cite karna)?
- [ ] **67.** Model ke context window aur token budget ko dekhte huye kitne retrieved chunks prompt me include karoge — kaise decide karoge?
- [ ] **68a.** Prompt me chunks ka order LLM ke liye matter karta hai kya?
- [ ] **68b.** "Lost in the middle" problem kya hai?
- [ ] **68c.** Isse kaise mitigate karoge?
- [ ] **69a.** RAG ke context me citation kya hai?
- [ ] **69b.** Kyun zaruri hai?
- [ ] **69c.** RAG output me citations kaise add karoge?
- [ ] **70a.** Query reformulation explain karo.
- [ ] **70b.** Query decomposition explain karo.
- [ ] **70c.** Conversation history kaise handle hoti hai — explain karo.
- [ ] **70d.** Output formatting explain karo.
- [ ] **70e.** HyDE (Hypothetical Document Embeddings) explain karo.
- [ ] **70f.** Knowledge graph augmentation explain karo — vector retrieval ke saath-saath knowledge graph use karke structured, relational context add karna. *(Knowledge graphs jab primary retrieval architecture ban jaayein — wo GraphRAG hai, Advanced RAG me covered.)*

## Generation

- [ ] **71.** Generation process ko explain karo.
- [ ] **72a.** Generation ke liye kaunse-kaunse models use hote hain?
- [ ] **72b.** Inke key parameters kya hain (jaise temperature, top-p, max tokens)?
- [ ] **72c.** Ye parameters RAG output ki quality kaise affect karte hain?
- [ ] **73.** RAG me extractive aur abstractive answer generation me kya difference hai?
- [ ] **74.** Hallucination ko kaise control karte ho?
- [ ] **75a.** Cost kaise optimize karte ho?
- [ ] **75b.** Token consumption kaise optimize karte ho?
- [ ] **76.** Response streaming karne se citations aur formatting ko kaise handle karna badalta hai?

## Evaluation

- [ ] **77.** RAG system evaluate kaise karte ho?
- [ ] **78.** Retrieval component evaluate karne aur generation component evaluate karne me kya difference hai?
- [ ] **79a.** Reference-free evaluation kya hai?
- [ ] **79b.** Reference-based evaluation kya hai?
- [ ] **79c.** Dono me kya difference hai?
- [ ] **79d.** Har category me kaunse metrics aate hain aur kaise calculate hote hain?
- [ ] **80.** RAG evaluate karne ke alag-alag metrics kaunse hain?
- [ ] **81.** Har metric ko examples aur calculation ke saath depth me explain karo.
- [ ] **82a.** "RAG triad" kya hai — context relevance, faithfulness/groundedness, aur answer relevance?
- [ ] **82b.** Ye teeno metrics ek doosre se kaise relate karti hain?
- [ ] **83a.** "LLM-as-a-judge" kya hai?
- [ ] **83b.** RAG outputs evaluate karne me ye kaise use hota hai?
- [ ] **84.** RAG evaluate karne ke liye golden/test dataset kaise build karte ho — especially jab labeled QA pairs exist hi na karein (jaise synthetic QA generation)?
- [ ] **85.** RAG evaluation ke liye commonly kaunse tools/frameworks use hote hain (jaise RAGAS, TruLens, DeepEval, Arize Phoenix)?
- [ ] **86a.** RAG me human evaluation ka role kya hai?
- [ ] **86b.** Automated/LLM-based evaluation ko ye kaise complement karta hai?

## Advanced & Agentic RAG

- [ ] **87a.** Agentic RAG kya hai?
- [ ] **87b.** LLM ko decide karne dena — kab, kya, aur kitni baar retrieve karna — standard RAG ke muqable architecture kaise badalta hai?
- [ ] **88a.** Self-RAG kya hai?
- [ ] **88b.** Ye model ko kaise apne output ko critique karne aur retrieve/revise karne ka decision lene deta hai?
- [ ] **89a.** Corrective RAG (CRAG) kya hai?
- [ ] **89b.** Ye irrelevant ya low-quality retrieved documents ko kaise handle karta hai?
- [ ] **90a.** Adaptive RAG kya hai?
- [ ] **90b.** System kaise decide karta hai ki kisi query ke liye retrieve karna hi hai ya directly answer dena hai?
- [ ] **91a.** GraphRAG kya hai (knowledge base ko graph ki tarah build karna aur graph traversal/community summaries se retrieve karna)?
- [ ] **91b.** Standard vector-based retrieval ke muqable ye kab preferred hai?
- [ ] **92.** RAG multi-modal data tak kaise extend hoti hai — images, audio, ya video pe retrieve aur generate karna, sirf text nahi?
- [ ] **93.** Multi-turn conversational RAG system me turns ke across memory aur context kaise manage karte ho, single query ke andar ke muqable?

## Production, Security & Monitoring

- [ ] **94a.** RAG pipeline me latency ki main sources kya hain?
- [ ] **94b.** End-to-end response time kaise optimize karoge?
- [ ] **95a.** Semantic caching kya hai?
- [ ] **95b.** Repeated ya similar queries ke liye ye cost aur latency kaise reduce kar sakti hai?
- [ ] **96.** Production me RAG system ko monitor aur observe kaise karte ho (retrieved chunks log karna, metrics time ke saath track karna, drift detect karna)?
- [ ] **97a.** Retrieved documents ke through prompt-injection attack kya hai (knowledge base ke document ke andar chhipe malicious instructions)?
- [ ] **97b.** Isse defend kaise karoge?
- [ ] **98.** Multi-tenant RAG system me kaise ensure karoge ki user sirf apne authorized documents hi retrieve kare?
- [ ] **99.** Knowledge base aur query volume grow karne ke saath RAG system scale karne ke key considerations kya hain?
