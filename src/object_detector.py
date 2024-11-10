
# Content of src/object_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import logging

class ObjectDetector:
    def __init__(self, config):
        self.model_path = config.get('yolo_model')
        self.conf_threshold = config.get('conf_threshold', 0.1)  # Lowered threshold
        logging.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        logging.info(f"YOLO model loaded successfully. Model type: {type(self.model)}")
        logging.info(f"YOLO model classes: {self.model.names}")

    def detect(self, frame):
        # Debug: print image shape and dtype
        logging.debug(f"Input image shape: {frame.shape}, dtype: {frame.dtype}")

        # Ensure the image is in the correct format (BGR)
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Debug: print pixel value range
        logging.debug(f"Pixel value range: min={frame.min()}, max={frame.max()}")

        results = self.model(frame, conf=self.conf_threshold)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = self.model.names[cls]
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'conf': conf,
                    'detection_type': 'YOLO'
                })
        logging.debug(f"YOLO detections: {detections}")
        return detections
