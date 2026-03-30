from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import settings

def load_documents(data_dir: str = settings.DATA_DIR):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    print(f"[Ingestion] Loaded {len(documents)} documents.")
    return documents

def split_documents(documents: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"[Ingestion] Split into {len(chunks)} chunks.")
    return chunks

