from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
from groq import Groq
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

# ✅ ONLY LOAD FAISS (no preprocessing)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ FAISS loaded")

# Groq setup
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

@app.get("/")
def home():
    return {"message": "PESCE Chatbot API is running"}

@app.post("/chat")
def chat(query: Query):
    try:
        results = db.similarity_search(query.question, k=5)

        context = "\n\n---\n\n".join([doc.page_content for doc in results])[:4000]

        if not context.strip():
            return {"answer": "I don't know based on available data."}

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI assistant for PES College of Engineering.
                    Answer clearly and completely using the given context."""
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query.question}"
                }
            ]
        )

        return {"answer": response.choices[0].message.content}

    except Exception as e:
        print("ERROR:", e)
        return {"answer": "Sorry, something went wrong."}