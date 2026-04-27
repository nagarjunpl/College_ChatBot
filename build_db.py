import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Fix user agent
os.environ["USER_AGENT"] = "Mozilla/5.0"

urls = [
    "https://pes-data-for-chatbot.vercel.app/"
]

loader = WebBaseLoader(urls)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
documents = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss_index")

print("Clean DB created successfully!")