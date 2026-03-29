from langchain_community.chat_models import ChatOllama
from config.settings import settings

def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.7,
        num_predict=256,
    )

