import re
import sqlite3
from fuzzywuzzy import fuzz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class ImprovedProductMatcher:
    def __init__(self, db_path='products.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.load_products()
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.index = self._create_index()

    def load_products(self):
        self.cursor.execute("SELECT brand, flavor, size FROM products")
        self.products = self.cursor.fetchall()

    def _create_index(self):
        product_texts = [f"{brand} {flavor} {size}".strip() for brand, flavor, size in self.products]
        embeddings = self.model.encode(product_texts)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def extract_size(self, text):
        size_pattern = r'\b(\d+(\.\d+)?)\s*(ML|L|OZ)\b'
        match = re.search(size_pattern, text, re.IGNORECASE)
        return match.group() if match else None

    def match_product(self, ocr_text):
        ocr_text = ocr_text.upper()
        size = self.extract_size(ocr_text)

        # Vector search
        query_vector = self.model.encode([ocr_text])
        distances, indices = self.index.search(query_vector.astype('float32'), 5)  # Get top 5 matches

        best_match = None
        best_score = 0

        for d, i in zip(distances[0], indices[0]):
            product = self.products[i]
            brand, flavor, db_size = product

            # Calculate separate scores for brand, flavor, and size
            brand_score = fuzz.partial_ratio(ocr_text, brand.upper())
            flavor_score = fuzz.partial_ratio(ocr_text, flavor.upper()) if flavor else 100
            size_score = 100 if size and size.upper() == db_size.upper() else 0

            # Combine vector similarity with fuzzy matching
            vector_score = 1 / (1 + d)  # Convert distance to similarity score
            total_score = (vector_score * 0.4) + (brand_score * 0.3) + (flavor_score * 0.2) + (size_score * 0.1)

            if total_score > best_score:
                best_score = total_score
                best_match = product

        if best_match:
            confidence = best_score / 100
            return {
                'brand': best_match[0],
                'flavor': best_match[1],
                'size': best_match[2]
            }, confidence
        else:
            return None, 0

    def close(self):
        self.conn.close()

class ProductMatcher:
    def __init__(self, product_database):
        self.product_database = product_database
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.index = self._create_index()

    def _create_index(self):
        embeddings = self.model.encode(self.product_database)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def match_product(self, ocr_text):
        ocr_text = ocr_text.upper()
        query_vector = self.model.encode([ocr_text])
        distances, indices = self.index.search(query_vector.astype('float32'), 1)  # Get top match

        best_match = self.product_database[indices[0][0]]
        confidence = 1 / (1 + distances[0][0])  # Convert distance to similarity score

        return best_match, confidence

def get_final_product(detections, threshold=0.5):
    if not detections:
        return "No product detected", 0.0

    product_scores = {}
    for detection in detections:
        product = detection['class']
        conf = detection['conf']
        if product not in product_scores:
            product_scores[product] = []
        product_scores[product].append(conf)

    if not product_scores:
        return "No consistent product detected", 0.0

    best_product = max(product_scores, key=lambda x: sum(product_scores[x]) / len(product_scores[x]))
    avg_confidence = sum(product_scores[best_product]) / len(product_scores[best_product])

    if avg_confidence > threshold:
        return best_product, avg_confidence
    else:
        return "Low confidence detection", avg_confidence

# Example usage
if __name__ == "__main__":
    matcher = ImprovedProductMatcher()

    # Test cases
    test_cases = [
        "JIM BEAM PEACH 375",
        "JACK DANIELS TENESSEE FIRE 50ML",
        "ABSOLUT CITRON 375ML",
        "WILD TURKEY 101 375 ML",
        "MAKERS MARK 50 ML",
        "SMIRNOF VODKA 375ML",  # Misspelled brand
        "OLD FORSTER 100 PROOF 375ML",  # Misspelled brand
    ]

    for case in test_cases:
        matched_product, confidence = matcher.match_product(case)
        if matched_product:
            print(f"Input: {case}")
            print(f"Matched Product: {matched_product['brand']} {matched_product['flavor']} {matched_product['size']}")
            print(f"Confidence: {confidence:.2f}")
        else:
            print(f"No match found for: {case}")
        print()

    matcher.close()