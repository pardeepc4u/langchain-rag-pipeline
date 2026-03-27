import pandas as pd

class DocumentDataset:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def get_documents(self):
        return self.data['text'].tolist()

    def get_document_embeddings(self, model):
        embeddings = []
        for doc in self.get_documents():
            embedding = model.encode(doc)
            embeddings.append(embedding)
        return embeddings