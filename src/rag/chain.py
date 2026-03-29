from langchain_community.chains.retrieval_qa import RetrievalQA
from langchain_core.prompts import PromptTemplate
from core.llm import get_llm
from storage.vectorstore import get_vectorstore
from config.settings import settings

# Custom prompt for citation
PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following context to answer the question. 
If you don't know the answer, just state "I don't know." DO NOT fabricate.
Always cite the source document and page number.

Context: {context}

Question: {question}

Answer:
"""

def get_qa_chain():
    llm = get_llm()
    vectorstore = get_vectorstore()

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K_RESULTS}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

