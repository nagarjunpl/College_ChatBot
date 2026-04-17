import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

all_docs = []

folder_path = "data"

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, file))
        pages = loader.load()
        all_docs.extend(pages)

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)

# 🔥 Save DB
db.save_local("faiss_index")

print("Vector DB saved!")