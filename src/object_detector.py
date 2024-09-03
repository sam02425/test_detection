#src/object_detector.py
from ultralytics import YOLO
import logging

# class ObjectDetector:
#     def __init__(self, model_path='yolo8n-l.pt', conf_threshold=0.1):
#         try:
#             self.model = YOLO(model_path)
#             logging.info(f"YOLO model loaded: {self.model}")
#         except Exception as e:
#             logging.error(f"Failed to load YOLO model: {e}")
#             raise
#         self.conf_threshold = conf_threshold
#         self.relevant_classes = ['bottle', 'cup', 'can', 'bowl', 'box', 'carton', 'package', 'book']

#     def detect(self, frame):
#         try:
#             results = self.model(frame, conf=self.conf_threshold)
#             all_detections = []
#             relevant_detections = []
#             for r in results:
#                 for box in r.boxes:
#                     cls = int(box.cls)
#                     conf = float(box.conf)
#                     class_name = self.model.names[cls]
#                     all_detections.append((class_name, conf))
#                     if class_name in self.relevant_classes:
#                         relevant_detections.append((box.xyxy[0].tolist(), conf, class_name))
#             logging.info(f"All detected objects: {all_detections}")
#             logging.info(f"Relevant detections: {relevant_detections}")
#             return relevant_detections
#         except Exception as e:
#             logging.error(f"Error in object detection: {e}")
#             return []
class ObjectDetector:
    def __init__(self, model_path='yolo8n-l.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                cls = int(box.cls)
                class_name = self.model.names[cls]
                detections.append({
                    'bbox': (x1, y1, x2, y2),  # Ensure this is a tuple, not a list
                    'class': class_name,
                    'conf': conf,
                    'detection_type': 'YOLO'
                })
        logging.info(f"YOLO Detected objects: {detections}")
        return detections

# 4. Update the process_frame function in ProductDetectionSystem
def process_frame(self, frame):
    yolo_detections = self.detector.detect(frame)
    ocr_results = self.ocr_processor.perform_ocr(frame)

    results = yolo_detections + [
        {
            'bbox': (0, 0, frame.shape[1], frame.shape[0]),
            'class': ocr_text,
            'conf': conf,
            'detection_type': 'OCR'
        } for (bbox, ocr_text, conf) in ocr_results
    ]

    return frame, results