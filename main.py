import cv2
import logging
import argparse
import yaml
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from ultralytics import YOLO
from transformers import BertTokenizer, BertForMaskedLM
from src.ocr_processor import OCRProcessor
from src.unified_product_matcher import UnifiedProductMatcher
from src.performance_monitor import performance_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProductDetectionSystem:
    def __init__(self, config):
        self.config = config
        self.detector = YOLO(config['yolo_model']) if config.get('use_yolo') else None
        self.ocr_processor = OCRProcessor() if config.get('use_ocr') else None
        self.product_matcher = UnifiedProductMatcher(config['db_path'], use_vector_db=config.get('use_vector_db', False))

        if config.get('use_bert'):
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def process_frame(self, frame):
        detections = []

        if self.detector:
            yolo_results = self.detector(frame, conf=self.config['conf_threshold'])
            for r in yolo_results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = self.detector.names[cls]
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name,
                        'conf': conf,
                        'detection_type': 'YOLO'
                    })

        if self.ocr_processor:
            ocr_results = self.ocr_processor.perform_ocr(frame)
            for bbox, text, conf in ocr_results:
                if conf > self.config['ocr_threshold']:
                    if self.config.get('use_bert'):
                        text = self.bert_correct(text)
                    matched_product, confidence = self.product_matcher.match_product(text)
                    if matched_product:
                        detections.append({
                            'bbox': bbox,
                            'class': f"{matched_product['brand']} {matched_product['flavor']} {matched_product['size']}",
                            'conf': confidence,
                            'detection_type': 'OCR'
                        })

        processed_frame = self.draw_detections(frame, detections)
        return processed_frame, detections

    def preprocess_for_ocr(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        return dilated

    def bert_correct(self, text):
        tokens = self.bert_tokenizer.tokenize(text)
        corrected_tokens = []
        for token in tokens:
            if token == '[UNK]':
                context = ' '.join(corrected_tokens + ['[MASK]'] + tokens[len(corrected_tokens)+1:])
                inputs = self.bert_tokenizer(context, return_tensors='pt')
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                predicted_token_id = outputs.logits[0, len(corrected_tokens)].argmax()
                predicted_token = self.bert_tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
                corrected_tokens.append(predicted_token)
            else:
                corrected_tokens.append(token)
        return self.bert_tokenizer.convert_tokens_to_string(corrected_tokens)

    def draw_detections(self, frame, detections):
        for detection in detections:
            bbox = detection['bbox']
            if detection['detection_type'] == 'YOLO':
                x1, y1, x2, y2 = map(int, bbox)
            elif detection['detection_type'] == 'OCR':
                # OCR bboxes are typically in the format [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                x1, y1 = map(int, bbox[0])
                x2, y2 = map(int, bbox[2])
            else:
                logging.warning(f"Unknown detection type: {detection['detection_type']}")
                continue

            color = (0, 255, 0) if detection['detection_type'] == 'YOLO' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{detection['class']} ({detection['conf']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame

@performance_logger
def process_video(input_path, output_path, config, max_frames=None):
    system = ProductDetectionSystem(config)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {input_path}")
        return []

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_count >= max_frames):
            break

        frame_count += 1
        logging.info(f"Processing frame {frame_count}")

        processed_frame, frame_detections = system.process_frame(frame)
        all_detections.extend(frame_detections)
        out.write(processed_frame)

        cv2.imshow('Processed Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info(f"Video processing complete. Output saved to {output_path}")
    return all_detections

def evaluate_detections(detections, ground_truth):
    true_positives = sum(1 for d in detections if d['class'] in ground_truth)
    false_positives = len(detections) - true_positives
    false_negatives = len(ground_truth) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_detections': len(detections),
        'average_confidence': sum(d['conf'] for d in detections) / len(detections) if detections else 0
    }

def main():
    parser = argparse.ArgumentParser(description="Product Detection System")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output directory")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ground_truth = config['ground_truth']
    results = {}

    for approach in ['YOLOv8+OCR+BERT+VectorDB', 'YOLOv8+OCR+BERT+DB', 'OCR+BERT', 'YOLOv8']:
        logging.info(f"Running {approach} approach")

        approach_config = config.copy()
        approach_config['use_yolo'] = 'YOLO' in approach
        approach_config['use_ocr'] = 'OCR' in approach
        approach_config['use_bert'] = 'BERT' in approach
        approach_config['use_vector_db'] = 'VectorDB' in approach

        output_video_path = os.path.join(output_dir, f"{approach}_processed_video_{timestamp}.mp4")

        detections = process_video(input_path, output_video_path, approach_config)

        if detections:
            evaluation = evaluate_detections(detections, ground_truth)
        else:
            evaluation = {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': len(ground_truth),
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'total_detections': 0,
                'average_confidence': 0
            }

        results[approach] = {
            'evaluation': evaluation,
            'output_video': output_video_path
        }

    results_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {results_path}")

    print("\nComparative Results:")
    for approach, data in results.items():
        print(f"\n{approach}:")
        for metric, value in data['evaluation'].items():
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()