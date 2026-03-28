import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from config import settings

class DocumentIngestion:
    def __init__(self):
        print(f"Initializing Ollama Embeddings with model: {settings.OLLAMA_EMBED_MODEL}")
        self.embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

    def load_documents(self, doc_path: str):
        print(f"Loading documents from {doc_path}...")
        if not os.path.exists(doc_path):
            print(f"Warning: Path {doc_path} does not exist.")
            return []
            
        loader = DirectoryLoader(doc_path, glob=".pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents

    def process_and_store(self, doc_path: str):
        # 1. Load Data
        docs = self.load_documents(doc_path)
        if not docs:
            print("No documents found.")
            return

        # 2. Split Text
        print("Splitting text...")
        chunks = self.text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks.")

        # 3. Create Chroma DB
        print(f"Storing in ChromaDB at {settings.CHROMA_PERSIST_DIR}...")
        
        # We create the DB. If it exists, we use it (persist_directory handles this).
        # For fresh ingestion, we use from_documents. 
        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        
        print("Vector store created and persisted successfully.")

if __name__ == "__main__":
    # Ensure documents directory exists
    os.makedirs("documents", exist_ok=True)
    
    # Create a dummy text file if none exists for testing
    dummy_path = "documents/demo.txt"
    if not os.listdir("documents"):
        print("No documents found in 'documents' folder. Creating a dummy file for testing...")
        with open(dummy_path, "w") as f:
            f.write("LangChain is a framework for developing applications powered by language models. RAG stands for Retrieval-Augmented Generation.")
    
    ingester = DocumentIngestion()
    ingester.process_and_store("documents")