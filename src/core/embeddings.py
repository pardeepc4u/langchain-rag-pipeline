from langchain_community.embeddings import OllamaEmbeddings
from config.settings import settings

def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL
    )