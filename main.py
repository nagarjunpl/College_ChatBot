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
        self.url = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _call_api(self, inputs):
        import time
        for attempt in range(5):  # retry up to 5 times
            response = requests.post(self.url, headers=self.headers, json={"inputs": inputs, "options": {"wait_for_model": True}})
            print(f"HF API status: {response.status_code}, body: {response.text[:200]}")
            
            if response.status_code == 200 and response.text.strip():
                return response.json()
            elif response.status_code == 503:
                # Model is loading — wait and retry
                wait = response.json().get("estimated_time", 10)
                print(f"Model loading, waiting {wait}s...")
                time.sleep(wait)
            else:
                time.sleep(2)
        
        raise Exception(f"HF API failed after retries: {response.status_code} {response.text}")

    def embed_documents(self, texts):
        result = self._call_api(texts)
        return result

    def embed_query(self, text):
        result = self._call_api(text)
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
        import traceback
        print("ERROR:", traceback.format_exc())  # prints full error in Render logs
        return {"answer": f"Error: {str(e)}"}