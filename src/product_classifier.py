import sqlite3
import os
import re
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class ProductClassifier:
    def __init__(self, db_path, use_vector_db=False):
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'product_database.db')
        if not os.path.exists(self.db_path):
            logging.error(f"Database file not found: {self.db_path}")
            self.conn = None
        else:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

        self.use_vector_db = use_vector_db
        if use_vector_db:
            self.vector_db = self.initialize_vector_db()

    def initialize_vector_db(self):
        products = self.load_products()
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        embeddings = model.encode([f"{p['brand']} {p['flavor']}" for p in products])

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        return {
            'index': index,
            'products': products,
            'model': model
        }

    def load_products(self):
        self.cursor.execute("""
            SELECT b.name as brand, f.name as flavor
            FROM brands b
            JOIN flavors f ON f.brand_id = b.id
        """)
        return [{'brand': row[0], 'flavor': row[1]} for row in self.cursor.fetchall()]

    def classify(self, ocr_text):
        if not self.conn:
            return "Database not initialized", 0

        ocr_text = ocr_text.lower()

        if self.use_vector_db:
            return self.vector_db_classify(ocr_text)
        else:
            return self.sql_classify(ocr_text)

    def vector_db_classify(self, ocr_text):
        query_vector = self.vector_db['model'].encode([ocr_text])
        distances, indices = self.vector_db['index'].search(query_vector.astype('float32'), 1)

        best_match = self.vector_db['products'][indices[0][0]]
        confidence = 1 / (1 + distances[0][0])

        return f"{best_match['brand']} {best_match['flavor']}", confidence

    def sql_classify(self, ocr_text):
        # Search for matching brands
        self.cursor.execute("""
            SELECT b.name, COUNT(*) as match_count
            FROM brands b
            JOIN brand_keywords bk ON b.id = bk.brand_id
            WHERE LOWER(?) LIKE '%' || LOWER(bk.keyword) || '%'
            GROUP BY b.id
            ORDER BY match_count DESC
            LIMIT 1
        """, (ocr_text,))
        brand_result = self.cursor.fetchone()

        if not brand_result:
            return "Unknown Product", 0

        brand, brand_match_count = brand_result

        # Search for matching flavors
        self.cursor.execute("""
            SELECT f.name, COUNT(*) as match_count
            FROM flavors f
            JOIN flavor_keywords fk ON f.id = fk.flavor_id
            JOIN brands b ON f.brand_id = b.id
            WHERE b.name = ? AND LOWER(?) LIKE '%' || LOWER(fk.keyword) || '%'
            GROUP BY f.id
            ORDER BY match_count DESC
            LIMIT 1
        """, (brand, ocr_text))
        flavor_result = self.cursor.fetchone()

        if flavor_result:
            flavor, flavor_match_count = flavor_result
            total_match_count = brand_match_count + flavor_match_count
            confidence = min(total_match_count / 10, 1.0)  # Normalize confidence
            return f"{brand} {flavor}", confidence
        else:
            confidence = min(brand_match_count / 5, 1.0)  # Normalize confidence
            return brand, confidence

    def extract_additional_info(self, ocr_text):
        info = {}

        # Extract volume
        volume_match = re.search(r'\d+(\.\d+)?\s*(ml|l)', ocr_text, re.IGNORECASE)
        if volume_match:
            info['volume'] = volume_match.group()

        # Extract alcohol percentage
        abv_match = re.search(r'\d+(\.\d+)?%', ocr_text)
        if abv_match:
            info['abv'] = abv_match.group()

        return info

    def __del__(self):
        if self.conn:
            self.conn.close()