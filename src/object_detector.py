from ultralytics import YOLO
import logging

class ObjectDetector:
    def __init__(self, model_path='yolo8n-l.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        logging.info(f"Loaded YOLO model from {model_path}")

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
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'conf': conf,
                    'detection_type': 'YOLO'
                })
        return detections