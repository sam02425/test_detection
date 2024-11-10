import click
import os
import logging
import yaml
from product_classifier import ProductClassifier, OCRResult, ModelConfig
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from paddleocr import PaddleOCR
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedProductClassifier(ProductClassifier):
    def __init__(self, db_path: str, config: dict):
        super().__init__(db_path, config.get('trocr_model_path', ModelConfig.ORIGINAL_MODEL_NAME), use_fine_tuned=False)
        self.yolo_model = YOLO(config['yolo_model'])
        self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.ppocr = PaddleOCR(use_angle_cls=True, lang='en')

    def detect_objects(self, image: np.ndarray):
        results = self.yolo_model(image)
        return results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()

    def perform_trocr(self, image: np.ndarray) -> dict:
        try:
            pixel_values = self.trocr_processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            chars = list(text)

            return {
                'text': text,
                'chars': chars
            }
        except Exception as e:
            logging.error(f"TrOCR failed: {e}")
            return {'text': '', 'chars': []}

    def perform_ppocr(self, image: np.ndarray) -> list:
        try:
            result = self.ppocr.ocr(image, cls=True)
            if result is None:
                return []

            processed_results = []
            for line in result:
                if isinstance(line, list) and len(line) == 2:
                    bbox, (text, confidence) = line
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1 = map(int, bbox[0])
                        x2, y2 = map(int, bbox[2])
                        processed_results.append({
                            'text': text,
                            'confidence': confidence,
                            'bounding_box': (x1, y1, x2, y2),
                            'chars': list(text)
                        })
            return processed_results
        except Exception as e:
            logging.error(f"PaddleOCR failed: {e}")
            return []

    def process_image(self, image_path: str) -> dict:
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image_hash = self.compute_image_hash(image_path)
            cached_result = self.get_cached_result(image_hash)
            if cached_result:
                logging.info(f"Using cached result for image: {image_path}")
                return cached_result

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, classes = self.detect_objects(image_rgb)

            ocr_results = {'trocr': [], 'ppocr': []}
            yolo_detections = []

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_image = image_rgb[y1:y2, x1:x2]

                # Perform OCR on cropped image
                trocr_result = self.perform_trocr(cropped_image)
                ocr_results['trocr'].append({
                    'text': trocr_result['text'],
                    'chars': trocr_result['chars'],
                    'bounding_box': (x1, y1, x2, y2)
                })

                ppocr_results = self.perform_ppocr(cropped_image)
                for ppocr_result in ppocr_results:
                    ppocr_result['bounding_box'] = (
                        x1 + ppocr_result['bounding_box'][0],
                        y1 + ppocr_result['bounding_box'][1],
                        x1 + ppocr_result['bounding_box'][2],
                        y1 + ppocr_result['bounding_box'][3]
                    )
                ocr_results['ppocr'].extend(ppocr_results)

                # Get YOLO class name
                class_name = self.yolo_model.names[int(cls)]
                yolo_detections.append(class_name)

            # Fusion method
            final_product, final_confidence = self.fuse_results(yolo_detections, ocr_results)

            combined_text = " ".join([r['text'] for results in ocr_results.values() for r in results] + yolo_detections)
            additional_info = self.extract_additional_info(combined_text)

            result = {
                "product": final_product,
                "confidence": final_confidence,
                "ocr_results": ocr_results,
                "yolo_detections": yolo_detections,
                "additional_info": additional_info
            }

            self.cache_result(image_hash, result)
            return result
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}", exc_info=True)
            return {
                "product": "Unknown",
                "confidence": 0.0,
                "ocr_results": {},
                "yolo_detections": [],
                "additional_info": {},
                "error": str(e)
            }

    def process_all_images(classifier, image_dir):
        image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        if not image_paths:
            logging.warning(f"No .jpg images found in directory: {image_dir}")
            return []

        results = []
        for image_path in image_paths:
            logging.info(f"Processing image: {image_path}")
            result = classifier.process_image(image_path)
            results.append((image_path, result))
        return results

    def fuse_results(self, yolo_detections, ocr_results):
            candidates = {}

            # Add YOLO detections with high weight
            for detection in yolo_detections:
                candidates[detection] = {'score': 0.8, 'source': 'YOLO'}

            # Process OCR results
            ocr_text = " ".join([r['text'] for results in ocr_results.values() for r in results])
            ocr_words = ocr_text.lower().split()

            # Compare OCR words with known products
            known_products = self.get_known_products()
            for product in known_products:
                product_words = product.lower().split()
                matches = sum(word in ocr_words for word in product_words)
                if matches > 0:
                    score = matches / len(product_words) * 0.5  # Lower weight for OCR
                    if product in candidates:
                        candidates[product]['score'] += score
                    else:
                        candidates[product] = {'score': score, 'source': 'OCR'}

            # Find the best match
            if candidates:
                best_match = max(candidates.items(), key=lambda x: x[1]['score'])
                return best_match[0], best_match[1]['score']
            else:
                return "Unknown", 0.0

    def get_known_products(self):
            # This method should return a list of known product names from your database
            # For example:
            return [
                "Cheetos-Crunchy-Flamin-Hot",
                "Cheetos-crunchy-XXTRA-Flamin-Hot",
                "DORITOS-Spicy-Nacho",
                "Chips-Ahoy-KingSize",

            ]

def print_result(image_path, result):
    click.echo(f"\n{'='*50}")
    click.echo(f"Results for image: {os.path.basename(image_path)}")
    click.echo(f"{'='*50}")

    if "error" in result:
        click.echo(f"An error occurred during classification: {result['error']}")
        return

    click.echo(f"Final Result:")
    click.echo(f"Product: {result['product']}")
    click.echo(f"Confidence: {result['confidence']:.2f}")

    click.echo("\nOCR Results:")
    for method, ocr_results in result['ocr_results'].items():
        click.echo(f"\n{method.upper()} Results:")
        for ocr_result in ocr_results:
            click.echo(f"  Text: {ocr_result.get('text', 'N/A')}")
            if 'confidence' in ocr_result:
                click.echo(f"  Confidence: {ocr_result['confidence']:.2f}")
            if 'chars' in ocr_result:
                click.echo(f"  Detected characters: {', '.join(ocr_result['chars'])}")
            if 'bounding_box' in ocr_result:
                click.echo(f"  Bounding Box: {ocr_result['bounding_box']}")
            click.echo("  ---")

    click.echo("\nYOLO Detections:")
    for detection in result['yolo_detections']:
        click.echo(f"Detected: {detection}")

    if result['additional_info']:
        click.echo(f"\nAdditional Info: {result['additional_info']}")
    else:
        click.echo(f"\nNo additional info extracted.")

@click.group()
def cli():
    pass

@cli.command()
@click.option('--image', required=True, help='Path to the image file or directory')
@click.option('--db', required=True, help='Path to the product database')
@click.option('--config', required=True, help='Path to the configuration file')
def classify_image(image, db, config):
    try:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)

        classifier = EnhancedProductClassifier(db, config_data)

        if os.path.isdir(image):
            image_paths = glob.glob(os.path.join(image, "*.jpg"))
            if not image_paths:
                click.echo(f"No .jpg images found in directory: {image}")
                return

            for image_path in image_paths:
                result = classifier.process_image(image_path)
                print_result(image_path, result)
        elif os.path.isfile(image):
            result = classifier.process_image(image)
            print_result(image, result)
        else:
            click.echo(f"Invalid path: {image}. Please provide a valid image file or directory.")

    except Exception as e:
        logging.error(f"Error during classification: {e}", exc_info=True)
        click.echo(f"An unexpected error occurred during classification: {e}")
        click.echo("Please check the logs for more details.")

@cli.command()
@click.option('--brand', required=True, help='Brand name')
@click.option('--flavor', required=True, help='Flavor name')
@click.option('--size', required=True, help='Size')
@click.option('--keywords', required=True, help='Keywords (comma-separated)')
@click.option('--db', required=True, help='Path to the product database')
@click.option('--config', required=True, help='Path to the configuration file')
def update_db(brand, flavor, size, keywords, db, config):
    try:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)

        classifier = EnhancedProductClassifier(db, config_data)
        classifier.update_database(brand, flavor, size, keywords.split(','))
        click.echo("Database updated successfully")
    except Exception as e:
        logging.error(f"Error updating database: {e}", exc_info=True)
        click.echo(f"An error occurred while updating the database: {e}")
        click.echo("Please check the logs for more details.")

if __name__ == "__main__":
    cli()