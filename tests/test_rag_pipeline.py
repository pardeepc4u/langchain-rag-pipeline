from src.ingestion.loader import split_documents
from src.core.embeddings import get_embeddings

def test_split_documents(mock_embeddings):
    docs = [mock_embeddings, mock_embeddings]  # Dummy docs
    # Actually, we need Langchain Documents
    from langchain_core.documents import Document
    docs = [
        Document(page_content="Hello " * 500), 
        Document(page_content="World " * 500)
    ]
    
    chunks = split_documents(docs)
    
    assert len(chunks) > 2  # Should be split due to chunk_size=1000
    assert len(chunks[0].page_content) <= 1000

 