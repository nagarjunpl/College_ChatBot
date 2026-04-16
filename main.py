from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
import google.generativeai as genai

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

# ── Gemini setup ──
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.0-flash")

@app.post("/chat")
def chat(query: Query):
    results = db.similarity_search(query.question, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    prompt = f"""You are a helpful assistant for PESCE (P.E.S. College of Engineering), Mandya.
Answer using ONLY the context below. Be clear and concise.
If the answer is not in the context, say: "I don't have that information. Please contact the college office directly."

Context:
{context}

Question: {query.question}"""

    response = gemini.generate_content(prompt)
    return {"answer": response.text}