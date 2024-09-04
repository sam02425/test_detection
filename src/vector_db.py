import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, product_database):
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.product_database = product_database
        self.index = self._create_index()

    def _create_index(self):
        embeddings = self.model.encode(self.product_database)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def search(self, query, k=5):
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        return [(self.product_database[i], 1 / (1 + d)) for d, i in zip(distances[0], indices[0])]