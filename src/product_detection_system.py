# #/Users/saumil/Desktop/yolov8+ocr/src/product_detection_system.py
# import cv2
# import logging
# import os
# from .object_detector import ObjectDetector
# from .ocr_processor import OCRProcessor
# from .product_classifier import ProductClassifier

# class ProductDetectionSystem:
#     def __init__(self, model_path='yolo8n-l.pt', conf_threshold=0.1, ocr_threshold=0.3):
#         self.detector = ObjectDetector(model_path, conf_threshold)
#         self.ocr_processor = OCRProcessor()
#         self.product_classifier = ProductClassifier()
#         self.ocr_threshold = ocr_threshold

#     def process_frame(self, frame):
#         logging.info("Processing frame")
#         try:
#             detections = self.detector.detect(frame)
#             logging.info(f"Detected objects: {detections}")

#             results = []
#             if not detections:
#                 logging.info("No relevant objects detected, performing OCR on entire frame")
#                 ocr_result = self.ocr_processor.perform_ocr(frame)
#                 logging.info(f"OCR Result for entire frame: {ocr_result}")

#                 if ocr_result:
#                     ocr_text = " ".join([text for _, text, _ in ocr_result])
#                     product_class = self.product_classifier.classify(ocr_text)
#                     additional_info = self.product_classifier.extract_additional_info(ocr_text)

#                     results.append({
#                         'bbox': (0, 0, frame.shape[1], frame.shape[0]),
#                         'class': 'full_frame',
#                         'conf': 1.0,
#                         'ocr_text': ocr_text,
#                         'product_class': product_class,
#                         'additional_info': additional_info
#                     })
#             else:
#                 for bbox, conf, class_name in detections:
#                     x1, y1, x2, y2 = map(int, bbox)
#                     roi = frame[y1:y2, x1:x2]
#                     ocr_result = self.ocr_processor.perform_ocr(roi)
#                     logging.info(f"OCR Result for {class_name}: {ocr_result}")

#                     if ocr_result:
#                         ocr_text = " ".join([text for _, text, _ in ocr_result])
#                         product_class = self.product_classifier.classify(ocr_text)
#                         additional_info = self.product_classifier.extract_additional_info(ocr_text)
#                     else:
#                         ocr_text = ""
#                         product_class = "Unknown Product"
#                         additional_info = {}

#                     results.append({
#                         'bbox': (x1, y1, x2, y2),
#                         'class': class_name,
#                         'conf': conf,
#                         'ocr_text': ocr_text,
#                         'product_class': product_class,
#                         'additional_info': additional_info
#                     })

#             # Draw bounding boxes and labels
#             for result in results:
#                 x1, y1, x2, y2 = result['bbox']
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 label = f"{result['product_class']} ({result['conf']:.2f})"
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                 if result['additional_info']:
#                     info_text = ', '.join([f"{k}: {v}" for k, v in result['additional_info'].items()])
#                     cv2.putText(frame, info_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#             return frame, results
#         except Exception as e:
#             logging.error(f"Frame processing failed: {str(e)}")
#             return frame, []

#     def process_static_image(self, image_path):
#         logging.info(f"Processing image: {image_path}")
#         if not os.path.isfile(image_path):
#             logging.error(f"File not found: {image_path}")
#             return

#         try:
#             frame = cv2.imread(image_path)
#             if frame is None:
#                 raise FileNotFoundError(f"Could not read the image: {image_path}")

#             logging.info(f"Image shape: {frame.shape}")
#             processed_frame, results = self.process_frame(frame)

#             if not results:
#                 logging.info("No products detected in the image.")
#             else:
#                 for result in results:
#                     logging.info(f"Detected: {result['product_class']} (Conf: {result['conf']:.2f}), "
#                                  f"OCR: {result['ocr_text']}, "
#                                  f"Additional Info: {result['additional_info']}, "
#                                  f"BBox: {result['bbox']}")

#             cv2.imshow('Processed Image', processed_frame)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#         except Exception as e:
#             logging.error(f"Error processing static image: {e}")

#     def run_webcam(self):
#         logging.info("Starting webcam")
#         try:
#             cap = cv2.VideoCapture(0)
#             if not cap.isOpened():
#                 raise IOError("Cannot open webcam")

#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     logging.error("Failed to grab frame")
#                     break

#                 processed_frame, results = self.process_frame(frame)
#                 cv2.imshow('Product Detection', processed_frame)

#                 for result in results:
#                     logging.info(f"Detected: {result['product_class']} (Conf: {result['conf']:.2f}), "
#                                  f"OCR: {result['ocr_text']}, "
#                                  f"Additional Info: {result['additional_info']}, "
#                                  f"BBox: {result['bbox']}")

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#             cap.release()
#             cv2.destroyAllWindows()
#         except Exception as e:
#             logging.error(f"Webcam processing failed: {e}")

import cv2
import logging
import numpy as np
from .object_detector import ObjectDetector
from .ocr_processor import OCRProcessor
from .product_classifier import ProductClassifier


class ProductDetectionSystem:
    def __init__(self, model_path='yolo8n-l.pt', conf_threshold=0.1, ocr_threshold=0.3):
        self.detector = ObjectDetector(model_path, conf_threshold)
        self.ocr_processor = OCRProcessor()
        self.product_classifier = ProductClassifier()
        self.ocr_threshold = ocr_threshold

    def process_frame(self, frame):
        logging.info("Processing frame")
        try:
            yolo_detections = self.detector.detect(frame)
            logging.info(f"YOLO Detected objects: {yolo_detections}")

            ocr_result = self.ocr_processor.perform_ocr(frame)
            logging.info(f"OCR Result: {ocr_result}")

            ocr_text = " ".join([text for _, text, _ in ocr_result])
            product_class, confidence = self.product_classifier.classify(ocr_text)
            additional_info = self.product_classifier.extract_additional_info(ocr_text)

            results = []
            if yolo_detections:
                for bbox, conf, class_name in yolo_detections:
                    x1, y1, x2, y2 = map(int, bbox)
                    results.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': class_name,
                        'conf': conf,
                        'detection_type': 'YOLO'
                    })

            results.append({
                'bbox': (0, 0, frame.shape[1], frame.shape[0]),
                'class': product_class,
                'conf': confidence,
                'ocr_text': ocr_text,
                'additional_info': additional_info,
                'detection_type': 'OCR'
            })

            # Draw bounding boxes and labels
            for result in results:
                x1, y1, x2, y2 = result['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{result['class']} ({result['conf']:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return frame, results
        except Exception as e:
            logging.error(f"Frame processing failed: {str(e)}")
            return frame, []