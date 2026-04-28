import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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

# Local build uses sentence-transformers (fine on your PC)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss_index")
print("✅ FAISS index saved!")