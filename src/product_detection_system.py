import cv2
import logging
import numpy as np
from ultralytics import YOLO
from src.ocr_processor import OCRProcessor
from src.product_matcher_vector_db import ImprovedProductMatcher

class ProductDetectionSystem:
    def __init__(self, config):
        self.config = config
        self.detector = YOLO(config['yolo_model']) if config['yolo_model'] else None
        self.ocr_processor = OCRProcessor() if config['use_ocr'] else None
        self.product_matcher = ImprovedProductMatcher(config['db_path'], use_vector_db=config['use_vector_db'])

        logging.info("ProductDetectionSystem initialized with configuration: %s", config)

    def process_frame(self, frame):
        logging.info("Processing frame")
        try:
            all_detections = []

            if self.detector:
                yolo_detections = self._perform_yolo_detection(frame)
                all_detections.extend(yolo_detections)

            if self.ocr_processor:
                ocr_detections = self._perform_ocr_detection(frame)
                all_detections.extend(ocr_detections)

            frame_with_boxes = self._draw_detections(frame, all_detections)
            return frame_with_boxes, all_detections

        except Exception as e:
            logging.error("Error processing frame: %s", str(e), exc_info=True)
            return frame, []

    def _perform_yolo_detection(self, frame):
        yolo_results = self.detector(frame, conf=self.config['conf_threshold'])
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
        return yolo_detections

    def _perform_ocr_detection(self, frame):
        ocr_results = self.ocr_processor.perform_ocr(frame)
        ocr_detections = []
        for bbox, text, conf in ocr_results:
            if conf > self.config['ocr_threshold']:
                matched_product, confidence = self.product_matcher.match_product(text)
                if matched_product:
                    x1, y1 = map(int, min(bbox, key=lambda p: p[0] + p[1]))
                    x2, y2 = map(int, max(bbox, key=lambda p: p[0] + p[1]))
                    ocr_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': f"{matched_product['brand']} {matched_product['flavor']} {matched_product['size']}",
                        'conf': confidence,
                        'detection_type': 'OCR'
                    })
        return ocr_detections

    def _draw_detections(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            color = (0, 255, 0) if detection['detection_type'] == 'YOLO' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{detection['class']} ({detection['conf']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

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