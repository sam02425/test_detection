import re
from collections import Counter
import sqlite3
from fuzzywuzzy import fuzz

class ImprovedProductMatcher:
    def __init__(self, db_path='products.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.load_products()

    def load_products(self):
        self.cursor.execute("SELECT brand, flavor, size FROM products")
        self.products = self.cursor.fetchall()

    def extract_size(self, text):
        size_pattern = r'\b(\d+(\.\d+)?)\s*(ML|L|OZ)\b'
        match = re.search(size_pattern, text, re.IGNORECASE)
        return match.group() if match else None

    def match_product(self, ocr_text):
        ocr_text = ocr_text.upper()
        size = self.extract_size(ocr_text)

        best_match = None
        best_score = 0

        for product in self.products:
            brand, flavor, db_size = product

            # Calculate separate scores for brand, flavor, and size
            brand_score = fuzz.partial_ratio(ocr_text, brand.upper())
            flavor_score = fuzz.partial_ratio(ocr_text, flavor.upper()) if flavor else 100
            size_score = 100 if size and size.upper() == db_size.upper() else 0

            # Weight the scores (you can adjust these weights)
            total_score = (brand_score * 0.5) + (flavor_score * 0.3) + (size_score * 0.2)

            if total_score > best_score:
                best_score = total_score
                best_match = product

        if best_match:
            confidence = best_score / 100
            return {
                'brand': best_match[0],
                'flavor': best_match[1],
                'size': best_match[2]
            }, confidence
        else:
            return None, 0

    def close(self):
        self.conn.close()

# Example usage
if __name__ == "__main__":
    matcher = ImprovedProductMatcher()

    # Test cases
    test_cases = [
        "JIM BEAM PEACH 375",
        "JACK DANIELS TENESSEE FIRE 50ML",
        "ABSOLUT CITRON 375ML",
        "WILD TURKEY 101 375 ML",
        "MAKERS MARK 50 ML",
        "SMIRNOF VODKA 375ML",  # Misspelled brand
        "OLD FORSTER 100 PROOF 375ML",  # Misspelled brand
    ]

    for case in test_cases:
        matched_product, confidence = matcher.match_product(case)
        if matched_product:
            print(f"Input: {case}")
            print(f"Matched Product: {matched_product['brand']} {matched_product['flavor']} {matched_product['size']}")
            print(f"Confidence: {confidence:.2f}")
        else:
            print(f"No match found for: {case}")
        print()

    matcher.close()

class ProductMatcher:
    def __init__(self, product_database):
        self.product_database = product_database

    def match_product(self, ocr_text):
        # Convert OCR text to uppercase for case-insensitive matching
        ocr_text = ocr_text.upper()

        # Extract numbers and words
        numbers = re.findall(r'\d+', ocr_text)
        words = re.findall(r'\b[A-Z]+\b', ocr_text)

        best_match = None
        best_score = 0

        for product in self.product_database:
            score = 0
            product_upper = product.upper()

            # Check for number matches
            for num in numbers:
                if num in product_upper:
                    score += 1

            # Check for word matches
            for word in words:
                if word in product_upper:
                    score += 2  # Give more weight to word matches

            if score > best_score:
                best_score = score
                best_match = product

        confidence = best_score / (len(numbers) + 2 * len(words)) if (len(numbers) + len(words)) > 0 else 0
        return best_match, confidence

def get_final_product(detections, threshold=0.5):
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
# In the main function:
def main():
    # ... (earlier code remains the same)

    yolo_product, yolo_confidence = get_final_product(yolo_detections)
    ocr_product, ocr_confidence = get_final_product(ocr_detections)

    print("\nYOLO Detection Result:")
    print(f"Product: {yolo_product}")
    print(f"Confidence: {yolo_confidence:.2f}")

    print("\nOCR Detection Result:")
    print(f"Product: {ocr_product}")
    print(f"Confidence: {ocr_confidence:.2f}")

    # Combined result (giving more weight to YOLO if confident)
    if yolo_confidence > 0.7:
        final_product = yolo_product
        final_confidence = yolo_confidence
    elif ocr_confidence > 0.5:
        final_product = ocr_product
        final_confidence = ocr_confidence
    else:
        final_product = "Unknown Product"
        final_confidence = max(yolo_confidence, ocr_confidence)

    print("\nFinal Combined Result:")
    print(f"Product: {final_product}")
    print(f"Confidence: {final_confidence:.2f}")

if __name__ == "__main__":
    main()