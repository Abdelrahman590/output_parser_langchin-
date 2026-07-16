# 📄 PDF Question Answering System with LangChain, FAISS, and Mistral-7B

An end-to-end Retrieval-Augmented Generation (RAG) application that enables users to ask natural language questions about the content of a PDF document. The system retrieves the most relevant text passages using semantic search (FAISS + Sentence Transformers) and generates context-aware answers using the Mistral-7B Large Language Model.

---

## 📌 Project Overview

This notebook demonstrates how to build a complete RAG pipeline capable of answering questions from PDF documents without requiring model fine-tuning.

Instead of sending the entire PDF to the language model, the document is divided into smaller chunks, converted into embeddings, indexed in a vector database, and only the most relevant chunks are retrieved for answering each question.

---

## 🚀 Features

- Load PDF documents
- Extract document text
- Split documents into semantic chunks
- Generate vector embeddings
- Store embeddings in a FAISS vector database
- Retrieve relevant context using semantic similarity
- Generate answers with Mistral-7B
- Interactive command-line Q&A interface
- Fully offline after downloading the required models

---

# 🏗️ System Architecture

```
                PDF Document
                     │
                     ▼
             PDF Text Extraction
                     │
                     ▼
            Document Chunking
                     │
                     ▼
       Sentence Transformer Embeddings
                     │
                     ▼
             FAISS Vector Database
                     │
         User Question (Query)
                     │
                     ▼
          Query Embedding Generation
                     │
                     ▼
      Semantic Similarity Search (FAISS)
                     │
                     ▼
          Top-k Relevant Chunks
                     │
                     ▼
             Prompt Construction
                     │
                     ▼
             Mistral-7B Inference
                     │
                     ▼
              Final Generated Answer
```

---

# 📚 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| LangChain | RAG Pipeline |
| PyPDFLoader | PDF Loading |
| CharacterTextSplitter | Document Chunking |
| Sentence Transformers | Text Embeddings |
| HuggingFace Embeddings | Embedding Wrapper |
| FAISS | Vector Database |
| Hugging Face Transformers | LLM Inference |
| Mistral-7B-Instruct | Answer Generation |
| PyTorch | Deep Learning Backend |

---

# 📂 Project Workflow

## 1. Load PDF

The notebook loads a PDF document using LangChain's `PyPDFLoader`.

```python
loader = PyPDFLoader(pdf_path)
documents = loader.load()
```

---

## 2. Split Document

Large documents are divided into overlapping chunks.

```python
CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
```

Benefits:

- Better retrieval
- Faster inference
- Reduced hallucinations

---

## 3. Create Embeddings

Each chunk is converted into a dense numerical representation using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings capture semantic meaning rather than simple keyword matching.

---

## 4. Build Vector Database

The embeddings are stored inside a FAISS vector database.

```python
vectordb = FAISS.from_documents(
    chunks,
    embedding
)
```

FAISS enables efficient similarity search over thousands of document chunks.

---

## 5. Retrieve Relevant Chunks

When the user asks a question:

```
Who is Cristiano Ronaldo?
```

the query is embedded using the same embedding model.

FAISS retrieves the most semantically relevant document chunks.

---

## 6. Prompt Construction

The retrieved context is inserted into a prompt.

Example:

```
Context:
{retrieved_chunks}

Question:
{user_question}

Answer:
```

The language model only answers using the retrieved context.

---

## 7. Generate Answer

The prompt is passed to Mistral-7B.

The model generates a natural language answer grounded in the retrieved document.

---

# 📖 Retrieval-Augmented Generation (RAG)

The notebook follows the Retrieval-Augmented Generation paradigm.

Instead of relying entirely on the language model's internal knowledge, external knowledge is retrieved from the PDF before generation.

Advantages include:

- Reduced hallucinations
- Better factual accuracy
- Lower computational cost
- No fine-tuning required
- Works with custom documents

---

# 📁 Project Structure

```
project/
│
├── notebook.ipynb
├── PDF document
├── FAISS Vector Store
├── Embedding Model
├── Mistral Model
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/pdf-rag-system.git
```

Install dependencies

```bash
pip install langchain
pip install faiss-cpu
pip install sentence-transformers
pip install transformers
pip install accelerate
pip install pypdf
pip install torch
```

---

# ▶️ Usage

Run the notebook.

Then ask questions such as:

```
Who is Cristiano Ronaldo?

Where was he born?

Which clubs did he play for?

How many Ballon d'Or awards has he won?

What are his achievements?
```

Example:

```
Ask a question:

> How many goals has Ronaldo scored?

Answer:

Cristiano Ronaldo has scored more than ...
```

---

# 📊 Components

| Component | Model |
|-----------|-------|
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| LLM | Mistral-7B-Instruct |
| Framework | LangChain |
| Language | Python |

---

# 🔄 Pipeline Summary

```
PDF
 │
 ▼
Load Document
 │
 ▼
Split into Chunks
 │
 ▼
Generate Embeddings
 │
 ▼
Store in FAISS
 │
 ▼
User Question
 │
 ▼
Embedding Query
 │
 ▼
Retrieve Top-k Chunks
 │
 ▼
Build Prompt
 │
 ▼
Mistral-7B
 │
 ▼
Answer
```

---

# 💡 Advantages

- Semantic Search
- Fast Retrieval
- Scalable
- Offline Inference
- No Fine-Tuning
- Supports Large PDFs
- Modular Design
- Easy to Extend

---

# ⚠️ Limitations

- Answer quality depends on retrieval quality.
- Very large prompts may exceed the model's context window.
- Poor chunking strategy can reduce retrieval accuracy.
- Retrieved context should be limited (Top-k) to avoid excessive prompt length.

---

# 🔮 Future Improvements

- Streamlit Web Interface
- Multi-PDF Support
- Conversational Memory
- Hybrid Search (BM25 + FAISS)
- Metadata Filtering
- Reranking Models
- Source Citation
- GPU Optimization
- Quantized Models
- Qdrant or ChromaDB Integration
- LangChain Retrieval Chains
- Evaluation Metrics
- Multi-language Support

---

# 📚 References

- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- Mistral-7B
- PyTorch

---

# 👨‍💻 Author

**Abdelrahman Mahmoud Mosaad**

Artificial Intelligence Engineer

Specializing in:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Natural Language Processing (NLP)
- AI Applications

---

# ⭐ Acknowledgements

This project demonstrates how modern Retrieval-Augmented Generation (RAG) systems combine vector databases, embedding models, and large language models to build intelligent document question-answering applications with improved accuracy and reduced hallucinations.
