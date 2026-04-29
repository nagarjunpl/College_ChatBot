from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from groq import Groq
from dotenv import load_dotenv
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

db = None
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.on_event("startup")
async def startup_event():
    global db
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
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
        print("ERROR:", traceback.format_exc())
        return {"answer": f"Error: {str(e)}"}