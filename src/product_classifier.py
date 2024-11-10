# Content of src/product_classifier.py
import os
import re
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import cv2
import numpy as np
import torch
import sqlite3
from transformers import AutoProcessor, VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
from safetensors.torch import load_file
from paddleocr import PaddleOCR
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
import hashlib
import json

@dataclass(frozen=True)
class ModelConfig:
    ORIGINAL_MODEL_NAME: str = 'microsoft/trocr-small-printed'
    FINE_TUNED_MODEL_PATH: str = 'test_detection2/models/model.safetensors'

@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]

class ProductClassifier:
    def __init__(self, db_path: str, trocr_model_path: str, use_fine_tuned: bool = True, use_vector_db: bool = True, use_gpu: bool = True, cache_dir: str = "cache"):
        self.db_path = os.path.abspath(db_path)
        self.cache_dir = cache_dir
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

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.initialize_ocr_models(trocr_model_path, use_fine_tuned, use_gpu)

        logging.info(f"ProductClassifier initialized with Vector DB: {use_vector_db}, "
                     f"Device: {self.device}")

    def initialize_ocr_models(self, trocr_model_path: str, use_fine_tuned: bool, use_gpu: bool):
        try:
            if use_fine_tuned:
                model_path = ModelConfig.FINE_TUNED_MODEL_PATH
            else:
                model_path = ModelConfig.ORIGINAL_MODEL_NAME

            self.trocr_processor = AutoProcessor.from_pretrained(model_path)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_path)

            if use_fine_tuned and os.path.isfile(trocr_model_path):
                state_dict = load_file(trocr_model_path)
                self.trocr_model.load_state_dict(state_dict, strict=False)
                logging.info(f"Loaded fine-tuned TrOCR weights from SafeTensors file: {trocr_model_path}")
            elif use_fine_tuned:
                logging.warning(f"Fine-tuned TrOCR model file not found: {trocr_model_path}")
                logging.info("Falling back to original TrOCR model")

            self.trocr_model.to(self.device)

            # Initialize PaddleOCR
            self.ppocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)

        except Exception as e:
            logging.error(f"Error initializing OCR models: {e}")
            raise

    def perform_ocr(self, image: np.ndarray) -> Dict[str, List[OCRResult]]:
        results = {}

        # TrOCR
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.trocr_model.generate(pixel_values)
            trocr_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            h, w = image.shape[:2]
            results["trocr"] = [OCRResult(text=trocr_text, confidence=1.0, bounding_box=(0, 0, w, h))]
        except Exception as e:
            logging.error(f"TrOCR failed: {e}")
            results["trocr"] = []

        # PaddleOCR
        try:
            paddle_result = self.ppocr.ocr(image, cls=True)
            results["ppocr"] = []
            if paddle_result:
                for line in paddle_result:
                    for word_info in line:
                        bbox, (text, confidence) = word_info
                        x1, y1 = map(int, bbox[0])
                        x2, y2 = map(int, bbox[2])
                        results["ppocr"].append(OCRResult(text=text, confidence=confidence, bounding_box=(x1, y1, x2, y2)))
        except Exception as e:
            logging.error(f"PaddleOCR failed: {e}")
            results["ppocr"] = []

        # Fallback to Tesseract if both TrOCR and PaddleOCR fail
        if not results["trocr"] and not results["ppocr"]:
            try:
                tesseract_text = pytesseract.image_to_string(image)
                h, w = image.shape[:2]
                results["tesseract"] = [OCRResult(text=tesseract_text, confidence=1.0, bounding_box=(0, 0, w, h))]
            except Exception as e:
                logging.error(f"Tesseract OCR failed: {e}")
                results["tesseract"] = []

        return results


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

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            kernel = np.ones((3,3), np.uint8)
            morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(morph)
            return image
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            raise

    def classify(self, ocr_results: Dict[str, List[OCRResult]]) -> Tuple[str, float]:
        try:
            combined_text = " ".join([result.text for results in ocr_results.values() for result in results])
            combined_text = combined_text.lower()

            if self.use_vector_db:
                return self.vector_db_classify(combined_text)
            else:
                return self.sql_classify(combined_text)
        except Exception as e:
            logging.error(f"Error classifying product: {e}")
            raise

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

            ocr_results = self.perform_ocr(image)

            # Classify using available OCR results
            final_product, final_confidence = "Unknown", 0.0
            for ocr_method, results in ocr_results.items():
                if results:
                    product, confidence = self.classify({ocr_method: results})
                    if confidence > final_confidence:
                        final_product, final_confidence = product, confidence

            combined_text = " ".join([result.text for results in ocr_results.values() for result in results])
            additional_info = self.extract_additional_info(combined_text)

            result = {
                "product": final_product,
                "confidence": final_confidence,
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
                json.dump(result, f, default=str)  # Use default=str to handle non-serializable objects
        except Exception as e:
            logging.warning(f"Error writing cache file: {cache_file}. Error: {e}")


    def update_database(self, brand: str, flavor: str, size: str, keywords: List[str]):
        try:
            # Insert or update brand
            self.cursor.execute("INSERT OR IGNORE INTO brands (name) VALUES (?)", (brand,))
            self.cursor.execute("SELECT id FROM brands WHERE name = ?", (brand,))
            brand_id = self.cursor.fetchone()[0]

            # Insert or update flavor
            self.cursor.execute("INSERT OR IGNORE INTO flavors (name) VALUES (?)", (flavor,))
            self.cursor.execute("SELECT id FROM flavors WHERE name = ?", (flavor,))
            flavor_id = self.cursor.fetchone()[0]

            # Insert or update size
            self.cursor.execute("INSERT OR IGNORE INTO sizes (name) VALUES (?)", (size,))
            self.cursor.execute("SELECT id FROM sizes WHERE name = ?", (size,))
            size_id = self.cursor.fetchone()[0]

            # Insert or update product
            self.cursor.execute("""
                INSERT OR IGNORE INTO products (brand_id, flavor_id, size_id, full_name)
                VALUES (?, ?, ?, ?)
            """, (brand_id, flavor_id, size_id, f"{brand} {flavor} {size}"))

            # Update keywords
            for keyword in keywords:
                self.cursor.execute("INSERT OR IGNORE INTO brand_keywords (brand_id, keyword) VALUES (?, ?)", (brand_id, keyword))
                self.cursor.execute("INSERT OR IGNORE INTO flavor_keywords (flavor_id, keyword) VALUES (?, ?)", (flavor_id, keyword))

            self.conn.commit()

            if self.use_vector_db:
                self.vector_db = self.initialize_vector_db()  # Reinitialize vector DB with updated products

            logging.info(f"Database updated with new product: {brand} {flavor} {size}")
        except sqlite3.Error as e:
            logging.error(f"Error updating database: {e}")
            self.conn.rollback()
            raise

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

# @cli.command()
# @click.option('--image', required=True, help='Path to the image file')
# @click.option('--db', required=True, help='Path to the product database')
# @click.option('--model', required=True, help='Path to the TrOCR model')
# def classify_image(image, db, model):
#     classifier = ProductClassifier(db, model)
#     result = classifier.process_image(image)
#     click.echo(f"Product: {result['product']}")
#     click.echo(f"Confidence: {result['confidence']:.2f}")
#     click.echo(f"Additional Info: {result['additional_info']}")

# @cli.command()
# @click.option('--brand', required=True, help='Brand name')
# @click.option('--flavor', required=True, help='Flavor name')
# @click.option('--size', required=True, help='Size')
# @click.option('--keywords', required=True, help='Keywords (comma-separated)')
# @click.option('--db', required=True, help='Path to the product database')
# @click.option('--model', required=True, help='Path to the TrOCR model')
# def update_db(brand, flavor, size, keywords, db, model):
#     classifier = ProductClassifier(db, model)
#     classifier.update_database(brand, flavor, size, keywords.split(','))
#     click.echo("Database updated successfully")

# # FastAPI interface
# app = FastAPI()

# @app.post("/classify")
# async def classify_image_api(file: UploadFile = File(...)):
#     temp_path = f"/tmp/{file.filename}"
#     with open(temp_path, "wb") as buffer:
#         buffer.write(await file.read())

#     classifier = ProductClassifier("path/to/db", "path/to/model")
#     result = classifier.process_image(temp_path)

#     os.remove(temp_path)
#     return result

# @app.post("/update_db")
# async def update_db_api(brand: str, flavor: str, size: str, keywords: str):
#     classifier = ProductClassifier("path/to/db", "path/to/model")
#     classifier.update_database(brand, flavor, size, keywords.split(','))
#     return {"message": "Database updated successfully"}

# if __name__ == "__main__":
#     # Set up logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     # Run CLI
#     cli()

#     # Uncomment the following line to run the FastAPI server
#     # uvicorn.run(app, host="0.0.0.0", port=8000)