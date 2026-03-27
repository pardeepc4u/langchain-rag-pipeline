import faiss

class Embedder:
    def __init__(self):
        self.index = faiss.IndexFlatL2(128)

    def encode(self, text):
        # Use a pre-trained model to get the embedding
        embedding = openai.Embedding.create(text).vector
        return embedding