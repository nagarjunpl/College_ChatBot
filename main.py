from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
# pip install groq
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import requests
from bs4 import BeautifulSoup

def load_website(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.extract()
    return soup.get_text(separator="\n", strip=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── THIS WAS MISSING ──
class Query(BaseModel):
    question: str

# ── Load PDFs ──
all_docs = []
folder_path = "data"

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, file))
        pages = loader.load()
        for doc in pages:
            doc.metadata["source"] = file
        all_docs.extend(pages)

print(f"Loaded {len(all_docs)} PDF pages")

# ── Load website ──
try:
    website_text = load_website("https://pesce.ac.in/")
    all_docs.append(Document(page_content=website_text, metadata={"source": "pesce.ac.in"}))
    print("Website loaded")
except Exception as e:
    print(f"Website load failed: {e}")

# ── Chunk & embed ──
splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=80)
docs = splitter.split_documents(all_docs)
print(f"Split into {len(docs)} chunks")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
print("Vector DB ready")

# ── Groq setup ──
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)

@app.get("/")
def home():
    return {"message": "PESCE Chatbot API is running"}

@app.post("/chat")
def chat(query: Query):
    try:
        results = db.similarity_search(query.question, k=5)

        # Always define context
        if results:
            context = "\n\n---\n\n".join([doc.page_content for doc in results])
            context = context[:4000]
        else:
            context = ""

        # If no context, return early
        if not context.strip():
            return {"answer": "I don't know based on available data."}

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                "role": "system",
                "content": """You are an AI assistant for PES College of Engineering.

                Answer clearly and completely using the given context.
                - Provide full answers, not short or partial.
                - If multiple points exist, explain them properly.
                - If answer is not in context, say: "I don't know based on available data."
                """
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
        return {"answer": "Sorry, something went wrong. Please try again."}