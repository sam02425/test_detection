#src/product_classifier.py
import sqlite3
import os
import re
import logging

class ProductClassifier:
    def __init__(self):
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'product_database.db')
        if not os.path.exists(db_path):
            logging.error(f"Database file not found: {db_path}")
            self.conn = None
            return
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def classify(self, ocr_text):
        if not self.conn:
            return "Database not initialized", 0

        ocr_text = ocr_text.lower()

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