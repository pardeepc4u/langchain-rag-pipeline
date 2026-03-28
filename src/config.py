import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # Ollama Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3:8b")
    
    # Vector Store Settings
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    # Document Processing Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

settings = Settings()