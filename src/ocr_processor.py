import cv2
import numpy as np
import easyocr

class OCRProcessor:
    def __init__(self, lang=['en']):
        self.reader = easyocr.Reader(lang)

    def preprocess_image(self, image):
        # Check if the image is already grayscale
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        denoised = cv2.fastNlMeansDenoising(gray)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        return dilated

    def perform_ocr(self, image):
        preprocessed = self.preprocess_image(image)
        results = self.reader.readtext(preprocessed)
        return results