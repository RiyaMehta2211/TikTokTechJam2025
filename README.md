# ComplianceGuard RAG System

**ComplianceGuard** is a prototype system that augments Large Language Models (LLMs) with **domain-specific regulatory knowledge** to detect product features that may require **geo-specific compliance logic**.

---

## The Problem: LLMs Lack Context

When evaluating product features for geo-specific compliance, off-the-shelf LLMs face several critical challenges:

- **Misinterpretation of Jargon**  
  Internal feature codenames, ambiguous abbreviations, and domain-specific terms are often misunderstood.

- **AI Hallucination**  
  Leads to exclusion of important features, misclassification, and ultimately, inaccurate or unreliable compliance flags.

- **Token Limits**  
  Critical context is truncated, degrading the quality and reliability of the model's reasoning.

---

## Why It Matters

Without the right context, even the most powerful LLMs can **overlook regulatory obligations**, creating risks of:

- **Legal exposure**  
- **Financial penalties**  
- **Loss of trust in compliance processes**


# Features

- **Automated Compliance Detection**  
  Identifies product features that may require geo-specific legal compliance.

- **Semantic Search**  
  Leverages cosine similarity with embeddings for accurate retrieval of relevant context.

- **Regulation Knowledge Base**  
  Pre-loaded with key regulations (e.g., DSA, California Kids Act, etc.).

- **Audit-Ready Outputs**  
  Generates transparent reasoning with clear references to regulations.

- **RAG Architecture**  
  Combines retrieval with LLM reasoning to provide accurate, context-aware compliance assessments.

---

# Advantages (vs. Normal Prompt Engineering & Vanilla LLMs)

- **Overcomes AI Hallucination**  
  Grounds the LLM in factual, company-specific regulatory knowledge by retrieving top-k relevant tokens from a comprehensive vector database.

- **Jargon-Aware**  
  Understands internal feature codenames and abbreviations, mapping them correctly to regulatory concepts.

- **Bypasses Token Limits**  
  Supplies only the most relevant regulatory context, without needing to load entire documents into a single prompt.

---

# Dependencies & Libraries

- **[chromadb](https://www.trychroma.com/)** – Vector database for storing and retrieving embeddings  
- **[Ollama](https://ollama.ai/)** (via HuggingFace import) – Local LLM inference  
- **[sentence-transformers](https://www.sbert.net/)** – Embedding generation for semantic search  
- **[pydantic](https://docs.pydantic.dev/)** – Data validation and structured modeling  
- **[langchain](https://www.langchain.com/)** – Framework for building RAG pipelines  
- **[numpy](https://numpy.org/)** – Core numerical computing library  
- **[transformers](https://huggingface.co/transformers/)** – HuggingFace library for pretrained models  
- **[torch](https://pytorch.org/)** – Deep learning framework powering model inference  

**Note:** To use `transformers` and `sentence-transformers`, you will need a [Hugging Face account](https://huggingface.co/join) and a personal access token.

# Technology Stack

- **Vector Database:** [ChromaDB](https://www.trychroma.com/)  
  Persistent and lightweight vector database for storing embeddings.

- **Embedding Model:** [SentenceTransformers](https://www.sbert.net/)  
  Using `all-MiniLM-L6-v2` for high-quality semantic embeddings.

- **Data Modeling:** [Pydantic](https://docs.pydantic.dev/)  
  Provides validated and structured data models.

- **Language:** Python **3.10+**

- **LLM:** [Ollama](https://ollama.ai/)  
  Local LLM inference to reduce API costs and enhance data privacy.
