from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import requests
from bs4 import BeautifulSoup

def load_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts/styles
    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text(separator="\n")
    return text

#  Create app
app = FastAPI()

#  Enable CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  STEP 1: Load ALL PDFs from folder
all_docs = []
folder_path = "data"

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, file))
        documents = loader.load()

        # Add source info (optional)
        for doc in documents:
            doc.metadata["source"] = file

        all_docs.extend(documents)

print(f" Loaded {len(all_docs)} pages from PDFs")

# Add your college website
website_url = "https://pesce.ac.in/"

website_text = load_website(website_url)

website_doc = Document(
    page_content=website_text,
    metadata={"source": "website"}
)

# Add to existing docs
all_docs.append(website_doc)

# STEP 2: Split text
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = text_splitter.split_documents(all_docs)

print(f" Split into {len(docs)} chunks")

#  STEP 3: Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

print(" Vector DB created")

#  STEP 4: Load QA Model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

print(" QA Model loaded")

#  Request format
class Query(BaseModel):
    question: str

#  Answer function
def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
    return answer

#  API endpoint
@app.post("/chat")
def chat(query: Query):
    #  Search relevant chunks
    results = db.similarity_search(query.question, k=3)

    #  Combine context
    context = " ".join([doc.page_content for doc in results])[:1000]

    #  Get answer
    answer = get_answer(query.question, context)

    return {"answer": answer}