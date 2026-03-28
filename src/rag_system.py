import os
import sys
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import settings

class RAGSystem:
    def __init__(self):
        print(f"Connecting to Ollama at {settings.OLLAMA_BASE_URL}")
        
        # 1. Load Embeddings
        self.embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        # 2. Load Chroma DB
        print("Loading Vector Database...")
        if not os.path.exists(settings.CHROMA_PERSIST_DIR):
            raise FileNotFoundError("Vector store not found. Please run 'python ingestion.py' first.")
            
        self.vectorstore = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings
        )
        
        # 3. Initialize Ollama LLM
        self.llm = ChatOllama(
            model=settings.OLLAMA_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.7,
        )

        # Custom Prompt Template
        template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
        Always provide a citation pointing to the specific document source used.

        {context}

        Question: {question}
        Answer:
        """
        
        self.PROMPT = ChatPromptTemplate.from_template(template)

        # Create the Retrieval Chain
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.PROMPT
            | self.llm
            | StrOutputParser()
        )

    def ask_question(self, query: str):
        try:
            docs = self.retriever.invoke(query)
            answer = self.qa_chain.invoke(query)
            return answer, docs
        except Exception as e:
            return f"Error: {str(e)}", []

if __name__ == "__main__":
    rag = RAGSystem()
    
    print("\n--- Ollama RAG System Initialized ---")
    print(f"Model: {settings.OLLAMA_LLM_MODEL}")
    print("Type 'exit' to quit.")
    
    while True:
        user_query = input("\nAsk a question: ")
        if user_query.lower() in ["exit", "quit"]:
            break
            
        answer, sources = rag.ask_question(user_query)
        
        print("\n" + "="*50)
        print("ANSWER:")
        print("-" * 50)
        print(answer)
        
        if sources:
            print("\n SOURCES:")
            print("-" * 50)
            for i, doc in enumerate(sources):
                # Chroma metadata is simpler than PyPDF
                source_info = doc.metadata.get("source", "Unknown")
                page_info = doc.metadata.get("page", "N/A")
                print(f"[Source {i+1}] {source_info} (Page: {page_info})")
                print(f"Content: {doc.page_content[:150]}...")
        print("="*50)