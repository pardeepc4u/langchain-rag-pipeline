import os
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from core.embeddings import get_embeddings
from config.settings import settings

def get_vectorstore(persist_dir: str = settings.CHROMA_PERSIST_DIR):
    os.makedirs(persist_dir, exist_ok=True)
    client = PersistentClient(path=persist_dir)
    
    vectorstore = Chroma(
        client=client,
        embedding_function=get_embeddings(),
    )
    return vectorstore

def create_vectorstore_from_chunks(chunks, persist_dir: str = settings.CHROMA_PERSIST_DIR):
    # First, clean the persist_dir (for fresh ingestion) or skip if you want to append
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir
    ).persist()
    print(f"[VectorStore] Created and saved to '{persist_dir}'")

