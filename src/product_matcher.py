
# Content of src/product_matcher.py
import sqlite3
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fuzzywuzzy import fuzz, process

class ProductMatcher:
    def __init__(self, db_path, use_vector_db=False):
        self.db_path = db_path
        self.use_vector_db = use_vector_db
        self.conn = None
        self.cursor = None
        self.products = []
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.products = self.load_products()
            if use_vector_db:
                self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                self.index = self._create_index()
            logging.info(f"Loaded {len(self.products)} products from database")
        except Exception as e:
            logging.error(f"Error initializing ProductMatcher: {e}")
            raise

    def load_products(self):
        try:
            self.cursor.execute("SELECT brand, flavor, size FROM products")
            return [{'brand': row[0], 'flavor': row[1], 'size': row[2]} for row in self.cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"SQLite error when loading products: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error when loading products: {e}")
            return []

    def _create_index(self):
        product_texts = [f"{p['brand']} {p['flavor']} {p['size']}" for p in self.products]
        embeddings = self.model.encode(product_texts)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def match_product(self, text):
        best_match, score = process.extractOne(text, [f"{p['brand']} {p['flavor']} {p['size']}" for p in self.products])
        confidence = score / 100.0
        if confidence > 0.7:  # Adjust this threshold as needed
            return best_match, confidence
        return None, 0.0

    def _vector_match(self, text):
        query_vector = self.model.encode([text])
        distances, indices = self.index.search(query_vector.astype('float32'), 1)
        best_match = self.products[indices[0][0]]
        confidence = 1 / (1 + distances[0][0])
        return best_match, confidence

    def _sql_match(self, text):
        text = text.lower()
        best_match = None
        best_score = 0
        for product in self.products:
            product_text = f"{product['brand']} {product['flavor']} {product['size']}".lower()
            score = self._similarity_score(text, product_text)
            if score > best_score:
                best_score = score
                best_match = product
        return best_match, best_score

    def _similarity_score(self, text1, text2):
        set1 = set(text1.split())
        set2 = set(text2.split())
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def close(self):
        if self.conn:
            self.conn.close()
