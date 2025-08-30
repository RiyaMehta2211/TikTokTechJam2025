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
- **[peft](https://huggingface.co/docs/peft/en/index)** - Parameter-Efficient Fine-Tuning (for LoRA)

**Note:** To use `transformers` and `sentence-transformers`, you will need a [Hugging Face account](https://huggingface.co/join) and a personal access token.

# Technology Stack

- **Vector Database:** [ChromaDB](https://www.trychroma.com/)  
  Persistent and lightweight vector database for storing embeddings.

- **Embedding Model:** [SentenceTransformers](https://www.sbert.net/)  
  Using `all-MiniLM-L6-v2` for high-quality semantic embeddings.

- **Data Modeling:** [Pydantic](https://docs.pydantic.dev/)  
  Provides validated and structured data models.

- **Language:** Python **3.10+**

- **LLM:** [Meta LLaMA **3.2 1B Instruct**](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
  HuggingFace model used for compliance reasoning and response generation.

## Setup

You can get started with ComplianceGuard in **two ways**:

---

### **Option 1: Local Jupyter Notebook**

1. **Download the repository** locally using Git:
    ```bash
    git clone https://github.com/RiyaMehta2211/TikTokTechJam2025
    cd TikTokTechJam2025
    ```

2. **Open the Jupyter Notebook:** Launch your Jupyter environment and open `TikTok TechJam 2025 submission.ipynb`.

3. **Install Dependencies:** All required Python libraries are handled within the notebook using `pip` commands in the first few cells.

4. **Run Cells Sequentially:** Execute each cell in order from top to bottom to:
    - Import necessary libraries
    - Define the `ChunkModel` and other classes
    - Initialize the embedding model and ChromaDB client
    - Parse the `compliance_knowledge_base.json` file and populate the database

5. **Execute a Query:** Run the final cell containing the `main()` function to test the system.

---

### **Option 2: Google Colab**

1. Click this link to open the notebook in Colab:  
   [Open in Google Colab](https://colab.research.google.com/drive/10f9sF0w0gpOVuNCqyQF7nfmOHgHqwFCc?usp=sharing)

2. **Install Dependencies:** Run the first few cells to install required Python libraries.

3. **Run the Notebook:** Execute cells sequentially to initialize the system, populate the database, and test queries as described in Option 1.

---


## Expected Output

After execution, you should see output similar to:
<img width="2880" height="452" alt="image" src="https://github.com/user-attachments/assets/cf53f992-3582-4b7c-97fb-b5e569e979dc" />
