from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
import requests
import os

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

# ✅ Lightweight HF API embeddings — no torch, no sentence-transformers
class HFAPIEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def embed_documents(self, texts):
        response = requests.post(self.url, headers=self.headers, json={"inputs": texts})
        return response.json()

    def embed_query(self, text):
        response = requests.post(self.url, headers=self.headers, json={"inputs": text})
        result = response.json()
        # HF API returns list of lists for single string — flatten if needed
        if isinstance(result[0], list):
            return result[0]
        return result

db = None
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.on_event("startup")
async def startup_event():
    global db
    embeddings = HFAPIEmbeddings(api_key=os.getenv("HF_TOKEN"))
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS loaded")

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
                {"role": "system", "content": "You are an AI assistant for PES College of Engineering. Answer clearly using the given context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.question}"}
            ]
        )
        return {"answer": response.choices[0].message.content}

    except Exception as e:
        print("ERROR:", e)
        return {"answer": "Sorry, something went wrong."}