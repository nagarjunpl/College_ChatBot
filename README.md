# 🎓 College AI Chatbot
 
An intelligent AI-powered chatbot for PES College of Engineering (PESCE) that helps students instantly find information about admissions, courses, placements, facilities, and more.
 
---
 
## ✨ Features
 
- 🤖 AI-based Question Answering — no predefined questions needed
- 🔍 Semantic search using FAISS vector database
- 🌐 Fetches data from the college website automatically
- ⚡ Fast responses powered by Groq (LLaMA 3.1)
- 📱 Fully responsive UI — works on mobile & desktop
- 💬 Chat history sidebar with quick links
- 🔒 No HuggingFace API dependency — runs embeddings locally
---

## 🛠️ Tech Stack
 
| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| Frontend | HTML, CSS, Vanilla JS |
| LLM | Groq API (LLaMA 3.1 8B) |
| Embeddings | FastEmbed (BAAI/bge-small-en-v1.5) |
| Vector DB | FAISS |
| Hosting | Render |
 
---
 
## 📁 Project Structure
 
```
college-chatbot/
├── main.py          # FastAPI backend
├── build_db.py      # Script to build FAISS index from website
├── faiss_index/     # Generated vector database (after running build_db.py)
├── requirements.txt # Python dependencies
├── Procfile         # Render deployment config
├── .python-version  # Python version pin
└── README.md
```
 
---
 
## 🧠 How It Works
 
```
User Question
     ↓
FastEmbed converts question → vector
     ↓
FAISS searches for similar chunks from website data
     ↓
Top 5 relevant chunks sent as context to Groq LLM
     ↓
LLaMA 3.1 generates a clear, concise answer
     ↓
Answer returned to user
```
