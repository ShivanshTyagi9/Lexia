# 📚 Hybrid RAG Pipeline (Qdrant + BM25 + Reranker)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)

A production-ready **Retrieval-Augmented Generation (RAG)** system optimized for **notes, research papers, books, and tables**. This implementation combines the best of semantic search, keyword matching, and intelligent reranking to deliver highly relevant answers with precise citations.

## Updates:

* **29th August,2025**
  * Modularized LLM call
  * Added support for Local LLMs via Ollama
  * Modified config
  * Support for Sentence-Transformer embeddings model

## ✨ Key Features

* **🔄 Hybrid Retrieval Architecture**
  * Dense semantic search via **Qdrant** vector database
  * Sparse keyword search via **Whoosh BM25**
  * Intelligent fusion with **Reciprocal Rank Fusion (RRF)**

* **📊 Smart Document Processing**
  * Multi-format support (PDF, DOCX, TXT, MD)
  * Automatic table detection and separation
  * Independent chunking strategies for text vs. tables

* **🎯 Precision Enhancement**
  * **CrossEncoder reranking** for higher accuracy
  * Diversity filtering to reduce redundancy
  * Context-aware chunk selection

* **💬 LLM Integration**
  * Contextual answer synthesis
  * Inline citations with page references `[chunk_id p.<page>]`
  * Configurable response formats

* **📈 Production Features**
  * Comprehensive feedback system
  * Document management tools
  * Evaluation harness with standard metrics
  * Admin controls for collection management

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "📱 Client Interface"
        UQ[Ask Question]
        UU[Upload Files]
        UI[Trigger Ingestion]
        UL[Browse Documents]
        UF[Submit Feedback]
    end

    subgraph "🔌 API Layer"
        UP[POST /upload]
        ING[POST /ingest]
        QRY[POST /query]
        ANS[POST /answer]
        DOCS[GET /documents]
        FB[POST /feedback]
        DEL[GET /delete]
    end

    subgraph "⚡ Document Processing"
        RD[Multi-format Reader]
        SEP[Text/Table Separator]
        CT[Text Chunker]
        CTB[Table Chunker]
        EMB[Embedding Generator]
        WIX[Whoosh BM25 Index]
        QDR[Qdrant Vector Store]
    end

    subgraph "🔍 Hybrid Retrieval"
        EQ[Query Encoder]
        DS[Dense Search]
        SS[Sparse Search]
        FUS[Rank Fusion]
        RER[CrossEncoder Rerank]
        DIV[Diversity Filter]
    end

    subgraph "🤖 Answer Generation"
        CTX[Context Builder]
        LLM[Language Model]
        CIT[Citation Formatter]
    end

    UU --> UP --> RD --> SEP
    SEP --> CT --> EMB --> QDR
    SEP --> CTB --> EMB
    CT --> WIX
    CTB --> WIX
    UI --> ING --> RD

    UQ --> ANS --> EQ
    EQ --> DS --> FUS
    EQ --> SS --> FUS
    FUS --> RER --> DIV --> CTX --> LLM --> CIT

    UL --> DOCS
    UF --> FB
    DEL --> QDR
    DEL --> WIX
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (for Qdrant)
- 4GB+ RAM recommended

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ShivanshTyagi9/Lexia.git
cd Lexia

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant Vector Database

```bash
# Using Docker (recommended)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Or using Docker Compose
docker-compose up -d qdrant
```

### 3. Launch the API Server

```bash
# Development mode with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`

### 4. Upload Your First Document

```bash
# Using curl
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"

# Trigger ingestion
curl -X POST "http://localhost:8000/ingest"
```

### 5. Ask Your First Question

```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings in this document?"}'
```

## 📚 API Reference

### Document Management

#### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

file: <PDF|DOCX|TXT|MD file>
```

#### Trigger Ingestion
```http
POST /ingest
```
Processes all files in the `data/pdfs` directory and indexes them.

#### List Documents
```http
GET /documents
```
Returns all indexed documents with metadata.

### Retrieval & Search

#### Query Documents
```http
POST /query
Content-Type: application/json

{
  "question": "What does table 2 show?",
  "k": 8,
  "mode": "table"  // optional: "text", "table", or "both"
}
```

#### Generate Answer
```http
POST /answer
Content-Type: application/json

{
  "query": "Summarize the key findings"
}
```

**Response:**
```json
{
  "answer": "Based on the documents, the key findings include...",
  "citations": [
    {
      "chunk_id": "doc1_chunk_3",
      "doc_title": "Research Paper.pdf",
      "pages": [2, 3]
    }
  ],
  "chunks": [
    {
      "chunk_id": "doc1_chunk_3",
      "content": "...",
      "score": 0.85
    }
  ]
}
```


### Administration

#### Reset Collections
```http
GET /delete?mode=collection&recreate=true&wipe_whoosh=true
```

Parameters:
- `mode`: "collection" (reset vector store)
- `recreate`: true/false (recreate empty collection)
- `wipe_whoosh`: true/false (clear BM25 index)

```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=hybrid_rag

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LLM_MODEL=gpt-3.5-turbo

# API Keys (if using external services)
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_hf_key_here

# Performance Tuning
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=20
RERANK_TOP_K=10
```

### Advanced Configuration

Edit `config/settings.py` for advanced options:

```python
class Settings:
    # Chunking strategy
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval parameters
    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3
    RRF_K = 60
    
    # Reranking
    ENABLE_RERANKING = True
    RERANK_TOP_K = 10
    
    # Answer generation
    MAX_CONTEXT_LENGTH = 4000
    TEMPERATURE = 0.1
```

## 🛣️ Roadmap

### Upcominf features
- [ ] Feedback mechanism
- [ ] support for different LLMs (Local LLMs and other LLM providers)
- [ ] Database Optimization 
- [ ] Batch processing for large document sets
- [ ] multi-query retrieval
- [ ] Agentic RAG pipeline
- [ ] creating essential tools
- [ ] streaming chat completion
- [ ] Federated search across multiple sources

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Qdrant](https://qdrant.tech/) for the excellent vector database
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [Whoosh](https://whoosh.readthedocs.io/) for BM25 search capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the robust API framework
---

**⭐ If this project helped you, please consider giving it a star!**