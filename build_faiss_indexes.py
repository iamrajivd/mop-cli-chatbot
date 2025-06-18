
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os

def build_index(file_path, output_dir, model="all-minilm", max_chunks=50):
    print(f"📄 Building index from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)[:max_chunks]
    docs = [Document(page_content=chunk) for chunk in chunks]

    embedding = OllamaEmbeddings(model=model)
    index = FAISS.from_documents(docs, embedding)
    index.save_local(output_dir)
    print(f"✅ Saved index to: {output_dir}")

# Create vector store directory if it doesn't exist
os.makedirs("vector_store", exist_ok=True)

# Build indexes for CMM and CMG
build_index("DN09131852CMM22.2ISS.1_V1_CLI Reference Guide - CMM22.2.txt", "vector_store/cmm_index")
build_index("CMG.txt", "vector_store/cmg_index")
