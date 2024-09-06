# src/unified_product_matcher.py

import sqlite3
from fuzzywuzzy import fuzz

class UnifiedProductMatcher:
    def __init__(self, db_path, use_vector_db=False):
        self.db_path = db_path
        self.use_vector_db = use_vector_db
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.products = self.load_products()

    def load_products(self):
        self.cursor.execute("SELECT brand, flavor, size FROM products")
        return [{'brand': row[0], 'flavor': row[1], 'size': row[2]} for row in self.cursor.fetchall()]

    def match_product(self, text):
        best_match = None
        best_score = 0

        for product in self.products:
            product_text = f"{product['brand']} {product['flavor']} {product['size']}"
            score = fuzz.partial_ratio(text.lower(), product_text.lower())

            if score > best_score:
                best_score = score
                best_match = product

        if best_score >= 80:  # Adjust this threshold as needed
            confidence = best_score / 100
            return best_match, confidence
        else:
            return None, 0

    def close(self):
        self.conn.close()