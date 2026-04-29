import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["USER_AGENT"] = "Mozilla/5.0"

urls = [
    "https://pes-data-for-chatbot.vercel.app/",
    "https://pes-data-for-chatbot.vercel.app/about.html",
    "https://pes-data-for-chatbot.vercel.app/programs.html",
    "https://pes-data-for-chatbot.vercel.app/academics.html",
    "https://pes-data-for-chatbot.vercel.app/admissions.html",
    "https://pes-data-for-chatbot.vercel.app/research.html",
    "https://pes-data-for-chatbot.vercel.app/placements.html",
]

loader = WebBaseLoader(urls)
loader.requests_kwargs = {"verify": False}
docs = loader.load()
print(f"✅ Loaded {len(docs)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
documents = splitter.split_documents(docs)
print(f"✅ Split into {len(documents)} chunks")

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss_index")
print("✅ FAISS index saved!")