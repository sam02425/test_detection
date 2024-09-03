import cv2
import torch
import numpy as np
from ultralytics import YOLO
import easyocr
from scipy import special

def test_libraries():
    print("Testing OpenCV:", cv2.__version__)
    print("Testing PyTorch:", torch.__version__)
    print("Testing NumPy:", np.__version__)
    print("Testing Ultralytics YOLO")
    model = YOLO("yolov8n.pt")
    print("Testing EasyOCR")
    reader = easyocr.Reader(['en'])
    print("Testing SciPy:", special.__version__)
    print("All libraries imported successfully!")

if __name__ == "__main__":
    test_libraries()