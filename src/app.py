import logging
from langchain.chains import RetrievalAugmentedGenerationChain
from langchain.prompts import BasePromptTemplate
from src.data import DocumentDataset
from src.embed import Embedder
from src.model import RAGModel

def main():
    # Initialize the dataset and embedder
    dataset = DocumentDataset('data/docs.csv')
    embedder = Embedder()

    # Initialize the RAG model
    rag_model = RAGModel()

    # Define a prompt template for retrieval-augmented generation
    class QuestionPrompt(BasePromptTemplate):
        def __init__(self, question):
            super().__init__(
                input_variables=['question'],
                output_variable='answer',
                variable_type='text'
            )
            self.question = question

        def to_markdown(self):
            return f'What is the answer to {self.question}?\n'

    # Define a chain for retrieval-augmented generation
    chain = RetrievalAugmentedGenerationChain(
        embedder=embedder,
        model=rag_model.model,
        database=dataset.get_document_embeddings(embedder),
        query_function=lambda x: QuestionPrompt(x).to_markdown(),
        max_queries=10
    )

    # Run the pipeline
    logging.info('Starting RAG pipeline...')
    prompt = 'What is the capital of France?'
    output = chain.run(prompt=prompt)
    logging.info(f'Answer: {output}')

if __name__ == '__main__':
    main()