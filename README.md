# College Chatbot (AI Powered)

An intelligent chatbot designed for college website that helps students quickly find information like placement rules, faculty details, and general college queries.

---

## Features

-  Supports multiple PDFs (rules, placements, faculty data)
-  Can fetch answers from college website
-  AI-based Question Answering (no predefined questions)
-  Semantic search using FAISS (Vector Database)
-  FastAPI backend for real-time responses
-  Simple UI for interaction
-  Provides precise answers instead of long paragraphs

---

##  How It Works

1. Load documents (PDFs + Website content)
2. Split text into smaller chunks
3. Convert text into embeddings using HuggingFace
4. Store embeddings in FAISS vector database
5. Retrieve relevant chunks and answer using QA model

---

##  Tech Stack

- Backend: FastAPI
- Frontend: HTML, CSS, JavaScript
- AI Models: HuggingFace Transformers
- Embeddings: Sentence Transformers
- Vector DB: FAISS

---

##  Run Project

```bash
pip install fastapi uvicorn langchain langchain-community langchain-core
pip install sentence-transformers transformers torch faiss-cpu
python -m uvicorn main:app --port 8001 --reload
```
