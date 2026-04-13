import re
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# 📄 Load PDF
loader = PyPDFLoader("data/nagarathna.pdf")
documents = loader.load()

# ✂️ Split text
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = text_splitter.split_documents(documents)

# 🔢 Embeddings (FREE)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🧠 Create Vector DB
db = FAISS.from_documents(docs, embeddings)

# 🤖 Load QA Model (NO pipeline issues)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# 🎯 Function to extract answer
def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
    return answer

# 🔁 Chat loop
while True:
    query = input("\nAsk your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("👋 Exiting chatbot...")
        break

    # 🔍 Retrieve relevant chunks
    results = db.similarity_search(query, k=3)

    # 🧩 Combine context (limit size to avoid errors)
    context = " ".join([doc.page_content for doc in results])[:1000]

    # 🎯 Get answer
    answer = get_answer(query, context)

    print("✅ Answer:", answer)