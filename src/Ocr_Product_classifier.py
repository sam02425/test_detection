import os
import re
import logging
import shutil
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import cv2
import numpy as np
import sqlite3
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
import hashlib
import json

@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]

class ProductClassifier:
    def __init__(self, db_path: str, use_vector_db: bool = True, cache_dir: str = "cache"):
        self.db_path = os.path.abspath(db_path)
        self.cache_dir = cache_dir

        # Clear the cache directory
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        logging.info(f"Attempting to connect to database at: {self.db_path}")
        logging.info(f"Current working directory: {os.getcwd()}")

        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logging.info(f"Successfully connected to database at: {self.db_path}")
        except sqlite3.Error as e:
            logging.error(f"Database connection error: {e}")
            logging.error(f"Failed to connect to database at: {self.db_path}")
            raise

        self.use_vector_db = use_vector_db
        if use_vector_db:
            self.vector_db = self.initialize_vector_db()

        logging.info(f"ProductClassifier initialized with Vector DB: {use_vector_db}")

    def initialize_vector_db(self):
        try:
            products = self.load_products()
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            embeddings = model.encode([f"{p['brand']} {p['flavor']} {p['size']}" for p in products])

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))

            return {
                'index': index,
                'products': products,
                'model': model
            }
        except Exception as e:
            logging.error(f"Error initializing vector DB: {e}")
            raise

    def load_products(self):
        try:
            self.cursor.execute("""
                SELECT p.id, b.name as brand, f.name as flavor, s.name as size
                FROM products p
                JOIN brands b ON p.brand_id = b.id
                JOIN flavors f ON p.flavor_id = f.id
                JOIN sizes s ON p.size_id = s.id
            """)
            return [{'id': row[0], 'brand': row[1], 'flavor': row[2], 'size': row[3]} for row in self.cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Error loading products from database: {e}")
            raise

    def perform_ocr(self, image: np.ndarray) -> Dict[str, List[OCRResult]]:
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_image)
            logging.info(f"Tesseract OCR result: {text}")
            h, w = image.shape[:2]
            return {"tesseract": [OCRResult(text=text.strip(), confidence=1.0, bounding_box=(0, 0, w, h))]}
        except Exception as e:
            logging.error(f"Tesseract OCR failed: {e}")
            return {"tesseract": []}

    def process_image(self, image_path: str) -> Dict[str, any]:
        try:
            image_hash = self.compute_image_hash(image_path)
            cached_result = self.get_cached_result(image_hash)
            if cached_result:
                logging.info(f"Using cached result for image: {image_path}")
                return cached_result

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            logging.info(f"Image shape: {image.shape}")
            logging.info(f"Image type: {image.dtype}")

            ocr_results = self.perform_ocr(image)

            product, confidence = self.classify(ocr_results)

            combined_text = " ".join([result.text for results in ocr_results.values() for result in results])
            additional_info = self.extract_additional_info(combined_text)

            result = {
                "product": product,
                "confidence": confidence,
                "ocr_results": {method: [{"text": r.text, "confidence": r.confidence, "bounding_box": r.bounding_box} for r in results] for method, results in ocr_results.items()},
                "additional_info": additional_info
            }

            self.cache_result(image_hash, result)
            return result
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return {
                "product": "Unknown",
                "confidence": 0.0,
                "ocr_results": {},
                "additional_info": {},
                "error": str(e)
            }

    def classify(self, ocr_results: Dict[str, List[OCRResult]]) -> Tuple[str, float]:
        combined_text = " ".join([result.text for results in ocr_results.values() for result in results])

        if self.use_vector_db:
            return self.vector_db_classify(combined_text)
        else:
            return self.sql_classify(combined_text)

    def vector_db_classify(self, ocr_text: str) -> Tuple[str, float]:
        try:
            query_vector = self.vector_db['model'].encode([ocr_text])
            distances, indices = self.vector_db['index'].search(query_vector.astype('float32'), 1)

            best_match = self.vector_db['products'][indices[0][0]]
            confidence = 1 / (1 + distances[0][0])

            return f"{best_match['brand']} {best_match['flavor']} {best_match['size']}", confidence
        except Exception as e:
            logging.error(f"Error in vector DB classification: {e}")
            raise

    def sql_classify(self, ocr_text: str) -> Tuple[str, float]:
        try:
            self.cursor.execute("""
                SELECT b.name, f.name, s.name, COUNT(*) as match_count
                FROM products p
                JOIN brands b ON p.brand_id = b.id
                JOIN flavors f ON p.flavor_id = f.id
                JOIN sizes s ON p.size_id = s.id
                LEFT JOIN brand_keywords bk ON b.id = bk.brand_id
                LEFT JOIN flavor_keywords fk ON f.id = fk.flavor_id
                WHERE LOWER(?) LIKE '%' || LOWER(COALESCE(bk.keyword, '')) || '%'
                   OR LOWER(?) LIKE '%' || LOWER(COALESCE(fk.keyword, '')) || '%'
                GROUP BY p.id
                ORDER BY match_count DESC
                LIMIT 1
            """, (ocr_text, ocr_text))
            result = self.cursor.fetchone()

            if not result:
                return "Unknown Product", 0

            brand, flavor, size, match_count = result
            confidence = min(match_count / 5, 1.0)
            return f"{brand} {flavor} {size}", confidence
        except sqlite3.Error as e:
            logging.error(f"SQL error in classification: {e}")
            raise

    def extract_additional_info(self, ocr_text: str) -> Dict[str, str]:
        info = {}

        volume_match = re.search(r'\d+(\.\d+)?\s*(ml|l)', ocr_text, re.IGNORECASE)
        if volume_match:
            info['volume'] = volume_match.group()

        abv_match = re.search(r'\d+(\.\d+)?%', ocr_text)
        if abv_match:
            info['abv'] = abv_match.group()

        return info

    def compute_image_hash(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def get_cached_result(self, image_hash: str) -> Optional[Dict[str, any]]:
        cache_file = os.path.join(self.cache_dir, f"{image_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Corrupted cache file: {cache_file}. Removing it.")
                os.remove(cache_file)
            except Exception as e:
                logging.warning(f"Error reading cache file: {cache_file}. Error: {e}")
        return None

    def cache_result(self, image_hash: str, result: Dict[str, any]):
        cache_file = os.path.join(self.cache_dir, f"{image_hash}.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f, default=str)
        except Exception as e:
            logging.warning(f"Error writing cache file: {cache_file}. Error: {e}")

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()