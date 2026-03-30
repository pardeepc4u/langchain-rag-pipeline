from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from src.core.llm import get_llm
from src.storage.vectorstore import get_vectorstore
from src.config.settings import settings

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K_RESULTS})

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = RunnableParallel(
        {
            "result": (
                {
                    "context": itemgetter("question") | retriever | format_docs,
                    "question": itemgetter("question"),
                }
                | prompt
                | llm
                | StrOutputParser()
            ),
            "source_documents": itemgetter("question") | retriever,
        }
    )
    return chain

