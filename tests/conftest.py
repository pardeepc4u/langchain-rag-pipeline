import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

# Mock Ollama Embeddings
@pytest.fixture
def mock_embeddings():
    with patch("src.core.embeddings.OllamaEmbeddings") as MockEmbeddings:
        instance = Mock()
        # Embeddings must return list of floats
        instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        instance.embed_query.return_value = [0.1, 0.2]
        MockEmbeddings.return_value = instance
        yield instance

# Mock QA Chain
@pytest.fixture
def mock_qa_chain():
    with patch("src.rag.chain.get_qa_chain") as MockChain:
        instance = Mock()
        instance.invoke.return_value = {
            "result": "LangChain is a framework for LLM apps.",
            "source_documents": [
                Document(
                    page_content="LangChain helps build apps.",
                    metadata={"source": "demo.pdf", "page": 1}
                )
            ]
        }
        MockChain.return_value = instance
        yield instance

        