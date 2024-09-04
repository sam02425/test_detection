import cv2
import logging
import numpy as np
from ultralytics import YOLO
from src.ocr_processor import OCRProcessor
from src.product_matcher_vector_db import ImprovedProductMatcher

class ProductDetectionSystem:
    def __init__(self, model_path='yolo8n-l.pt', conf_threshold=0.25, ocr_threshold=0.3):
        self.detector = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.ocr_processor = OCRProcessor()
        self.ocr_threshold = ocr_threshold
        self.product_matcher = ImprovedProductMatcher()

    def process_frame(self, frame):
        logging.info("Processing frame")

        # YOLO detection
        yolo_results = self.detector(frame, conf=self.conf_threshold)
        yolo_detections = []
        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                cls = int(box.cls)
                class_name = self.detector.names[cls]
                matched_product, confidence = self.product_matcher.match_product(class_name)
                if matched_product:
                    yolo_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': f"{matched_product['brand']} {matched_product['flavor']} {matched_product['size']}",
                        'conf': confidence,
                        'detection_type': 'YOLO'
                    })

        # OCR detection
        ocr_results = self.ocr_processor.perform_ocr(frame)
        ocr_detections = []
        for bbox, text, conf in ocr_results:
            if conf > self.ocr_threshold:
                matched_product, confidence = self.product_matcher.match_product(text)
                if matched_product:
                    x1 = int(min(point[0] for point in bbox))
                    y1 = int(min(point[1] for point in bbox))
                    x2 = int(max(point[0] for point in bbox))
                    y2 = int(max(point[1] for point in bbox))
                    ocr_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': f"{matched_product['brand']} {matched_product['flavor']} {matched_product['size']}",
                        'conf': confidence,
                        'detection_type': 'OCR'
                    })

        all_detections = yolo_detections + ocr_detections

        # Draw bounding boxes and labels
        for detection in all_detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection['class']} ({detection['conf']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame, all_detections

    def get_final_product(self, detections, threshold=0.5):
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