# test_detection
# Improved Product Detection System

## Overview

This system is designed to detect and identify products in video streams or webcam feeds using a combination of object detection and optical character recognition (OCR) technologies. It's particularly useful for inventory management, retail analytics, and automated product identification tasks.

## How It Works

1. **Input**: The system takes either a video file or a live webcam feed as input.

2. **Frame Processing**: Each frame of the video/webcam feed is processed individually.

3. **Object Detection**: YOLO (You Only Look Once) model is used to detect products in the frame.

4. **OCR Processing**: Optical Character Recognition is performed on the frame to extract text.

5. **Detection Combination**: Results from YOLO and OCR are combined.

6. **Product Matching**: Detected objects and text are matched against a product database.

7. **Confidence Calculation**: A confidence score is calculated based on the consistency and quality of detections.

8. **Final Output**: The system outputs the most likely product detected along with a confidence score.

## Models Involved

1. **YOLO (You Only Look Once)**
   - Purpose: Object detection
   - Version: YOLOv8
   - Input: Video frame
   - Output: Bounding boxes, class labels, and confidence scores for detected objects

2. **OCR (Optical Character Recognition)**
   - Purpose: Text extraction from images
   - Library: EasyOCR
   - Input: Video frame
   - Output: Detected text with bounding boxes and confidence scores

3. **BERT (Bidirectional Encoder Representations from Transformers)**
   - Purpose: Improve OCR results by predicting masked words
   - Model: bert-base-uncased
   - Input: OCR output with masked words
   - Output: Predicted words for masked tokens

4. **Product Matcher**
   - Purpose: Match detected objects and text to product database
   - Algorithm: Custom matching algorithm using fuzzy string matching
   - Input: Detected objects and text
   - Output: Matched product names and confidence scores

## Process Flow

1. Video Input → Frame Extraction
2. Frame → YOLO Model → Object Detections
3. Frame → OCR Model → Text Detections
4. Text Detections → BERT Model → Improved Text Detections
5. Object Detections + Improved Text Detections → Product Matcher
6. Product Matcher → Database Lookup
7. Database Lookup → Confidence Calculation
8. Confidence Calculation → Final Product Identification

## Setup and Usage

1. Install required dependencies (OpenCV, YOLO, EasyOCR, transformers, etc.)
2. Place your video files in the `data/` directory
3. Run the script:
python main.py
4. Choose between webcam input or video file processing when prompted

## Configuration

- Adjust confidence thresholds in the `ProductDetectionSystem` initialization
- Modify the `PRODUCT_DATABASE` list to match your inventory
- Fine-tune YOLO model on your specific products for better accuracy

## Future Improvements

- Implement real-time database updates
- Add support for multiple simultaneous product detections
- Integrate with inventory management systems
- Develop a user interface for easier operation


Graph 
<img src="https://github.com/user-attachments/assets/e615117b-c72f-48c4-88e7-d5bbc8c39f4e" alt="graph" width="400"/>
