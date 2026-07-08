# RAG — Retrieval-Augmented Generation Learning Kit

A complete, beginner-to-production study kit for **Retrieval-Augmented Generation (RAG)**. Available in both **English** and **Hinglish**, with each language providing the same three companion resources in both **Markdown** and **PDF** formats:

1. **Study Guide** — A structured, end-to-end walkthrough of every stage of a RAG pipeline (load → chunk → embed → index → retrieve → re-rank → prompt → generate → evaluate → productionize).
2. **Q&A Deep Dive** — Detailed, example-rich answers to 99 interview-style questions.
3. **Questions List** — A flat checklist of all 99 questions for self-testing.

---

# Repository Structure

```text
RAG/
├── README.md
├── markdown/
│   ├── english/
│   │   ├── study-guide.md
│   │   ├── qa-deep-dive.md
│   │   └── questions.md
│   └── hinglish/
│       ├── study-guide.md
│       ├── qa-deep-dive.md
│       └── questions.md
└── pdf/
    ├── english/
    │   ├── study-guide.pdf
    │   ├── qa-deep-dive.pdf
    │   └── questions.pdf
    └── hinglish/
        ├── study-guide.pdf
        ├── qa-deep-dive.pdf
        └── questions.pdf
```

---

# Available Formats

Each language includes the complete learning kit in two formats:

| Format | Best For |
| --- | --- |
| **Markdown** | Reading in GitHub, Obsidian, VS Code, Cursor, Zed, or any Markdown editor |
| **PDF** | Offline reading, printing, tablets, and mobile devices |

---

# How to Use This Kit

| If you want to… | Start here |
| --- | --- |
| Learn RAG from scratch, end-to-end | `markdown/english/study-guide.md` |
| Drill for interviews or self-test | `markdown/english/questions.md` → `markdown/english/qa-deep-dive.md` |
| Prefer Hinglish explanations | `markdown/hinglish/study-guide.md` |
| Read offline or print | Corresponding files under the `pdf/` directory |
| Look up a single concept quickly | Glossary at the end of either study guide |

---

# Recommended Learning Path

1. Read the **Study Guide** once from start to finish to build a solid mental model.
2. Attempt every question in **Questions List** without referring to the answers.
3. Verify your answers using the **Q&A Deep Dive**.
4. Revisit the relevant Study Guide sections for any weak topics.
5. Repeat until you can confidently answer all questions.

---

# What's Covered

- **Foundations** — Why RAG exists, hallucinations, RAG vs fine-tuning vs long-context prompting.
- **Architecture** — Offline indexing pipeline and online retrieval-generation pipeline.
- **Document Loading & Preprocessing** — PDFs, tables, images, OCR, metadata, cleaning.
- **Chunking** — Fixed-size, semantic, recursive, parent-child, propositional, contextual retrieval, and more.
- **Embeddings** — Sparse vs dense embeddings, contrastive learning, MTEB, asymmetric retrieval, fine-tuning.
- **Vector Databases & Indexing** — HNSW, IVF, PQ/SQ quantization, similarity metrics, and database selection.
- **Retrieval** — BM25, hybrid search, RRF, HyDE, MMR, self-query retrieval, parent-document retrieval.
- **Re-ranking** — Bi-encoders, cross-encoders, retrieval funnels, thresholding.
- **Prompting & Context Augmentation** — Grounding, citations, query rewriting, lost-in-the-middle mitigation.
- **Generation** — Decoding strategies, streaming, extractive vs abstractive generation, hallucination reduction.
- **Evaluation** — Recall@K, MRR, the RAG triad, LLM-as-a-judge, golden datasets.
- **Advanced RAG** — Agentic RAG, Self-RAG, CRAG, Adaptive RAG, GraphRAG, multimodal RAG, conversational memory.
- **Production & Scaling** — Semantic caching, observability, latency optimization, security, prompt injection defense, multi-tenant systems.

---

# License & Attribution

This repository is intended as personal study material.

Feel free to fork, adapt, extend, and share it for learning purposes.
