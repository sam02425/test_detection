
# Content of src/unified_product_matcher.py
import sqlite3
import logging
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class UnifiedProductMatcher:
    def __init__(self, db_path, use_vector_db=True, similarity_threshold=0.7):
        self.use_vector_db = use_vector_db
        self.similarity_threshold = similarity_threshold
        self.products = self.load_products(db_path)

        if use_vector_db:
            self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            self.index = self.create_index()

    def load_products(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT brand, flavor, size FROM products")
        products = [{'brand': row[0], 'flavor': row[1], 'size': row[2]} for row in cursor.fetchall()]
        conn.close()
        return products

    def create_index(self):
        product_texts = [f"{p['brand']} {p['flavor']} {p['size']}" for p in self.products]
        embeddings = self.model.encode(product_texts)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def match_product(self, text):
        if self.use_vector_db:
            return self.vector_match(text)
        else:
            return self.fuzzy_match(text)

    def vector_match(self, text):
        query_vector = self.model.encode([text])
        distances, indices = self.index.search(query_vector.astype('float32'), 5)  # Get top 5 matches

        best_match = None
        best_score = 0
        for dist, idx in zip(distances[0], indices[0]):
            product = self.products[idx]
            product_text = f"{product['brand']} {product['flavor']} {product['size']}"
            score = fuzz.partial_ratio(text.lower(), product_text.lower())
            if score > best_score:
                best_score = score
                best_match = product

        confidence = best_score / 100  # Convert to 0-1 range
        return best_match, confidence

    def fuzzy_match(self, text):
        best_match = None
        best_score = 0
        for product in self.products:
            product_text = f"{product['brand']} {product['flavor']} {product['size']}"
            score = fuzz.partial_ratio(text.lower(), product_text.lower())
            if score > best_score:
                best_score = score
                best_match = product

        confidence = best_score / 100  # Convert to 0-1 range
        return best_match, confidence

    def close(self):
        # No need to close anything for this implementation
        logging.info("UnifiedProductMatcher resources released")

    def __del__(self):
        self.close()
