import re
import sqlite3
from fuzzywuzzy import fuzz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

class UnifiedProductMatcher:
    def __init__(self, db_path='products.db', use_vector_db=True):
        self.db_path = db_path
        self.use_vector_db = use_vector_db
        self.conn = None
        self.cursor = None
        self.products = []
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.products = self.load_products()
            self.use_vector_db = use_vector_db
            if use_vector_db:
                self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                self.index = self._create_index()
        except Exception as e:
            logging.error(f"Error initializing UnifiedProductMatcher: {e}")

    def load_products(self):
        try:
            logging.info(f"Attempting to load products from {self.db_path}")
            self.cursor.execute("SELECT brand, flavor, size FROM products")
            products = self.cursor.fetchall()
            logging.info(f"Loaded {len(products)} products from database")
            return [{'brand': row[0], 'flavor': row[1], 'size': row[2]} for row in products]
        except sqlite3.Error as e:
            logging.error(f"SQLite error when loading products: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error when loading products: {e}")
            return [
                {"brand": "1792", "flavor": "Bottled In Bond", "size": "750ML"},
                {"brand": "1792", "flavor": "Full Proof", "size": "750ML"},
                {"brand": "1792", "flavor": "Small Batch", "size": "750ML"},
                {"brand": "1792", "flavor": "Sweet Wheat", "size": "750ML"},
                {"brand": "1800", "flavor": "Coconut Tequila", "size": "750ML"},
                {"brand": "1800", "flavor": "Cristalino Anejo", "size": "750ML"},
                {"brand": "1800", "flavor": "Silver Blanco Tequila", "size": "750ML"},
                {"brand": "360", "flavor": "Blue Raspberry", "size": "50ML"},
                {"brand": "360", "flavor": "Pineapple", "size": "50ML"},
                {"brand": "99", "flavor": "Apple", "size": "100ML"},
                {"brand": "99", "flavor": "Apple", "size": "50ML"},
                {"brand": "99", "flavor": "Bananas", "size": "100ML"},
                {"brand": "99", "flavor": "Bananas", "size": "50ML"},
                {"brand": "99", "flavor": "Blue Raspberry", "size": "50ML"},
                {"brand": "99", "flavor": "Butterscotch", "size": "100ML"},
                {"brand": "99", "flavor": "Cherry Limeade", "size": "50ML"},
                {"brand": "99", "flavor": "Chocolate", "size": "50ML"},
                {"brand": "99", "flavor": "Fruit Punch", "size": "50ML"},
                {"brand": "99", "flavor": "Grapes", "size": "100ML"},
                {"brand": "99", "flavor": "Mystery Flavor", "size": "50ML"},
                {"brand": "99", "flavor": "Peach", "size": "50ML"},
                {"brand": "99", "flavor": "Peaches", "size": "100ML"},
                {"brand": "99", "flavor": "Peppermint", "size": "50ML"},
                {"brand": "99", "flavor": "Root Beer", "size": "50ML"},
                {"brand": "99", "flavor": "Sour Apple", "size": "50ML"},
                {"brand": "99", "flavor": "Sour Berry", "size": "50ML"},
                {"brand": "99", "flavor": "Sour Cherry", "size": "50ML"},
                {"brand": "99", "flavor": "Whipped", "size": "50ML"},
                {"brand": "Absolut", "flavor": "80", "size": "375ML"},
                {"brand": "Admiral Nelson's", "flavor": "Spiced Rum", "size": "1.75L"},
                {"brand": "Admiral Nelson's", "flavor": "Spiced Rum", "size": "750ML"},
                {"brand": "Avion", "flavor": "Silver Tequila", "size": "750ML"},
                {"brand": "Bacardi", "flavor": "Gold", "size": "375ML"},
                {"brand": "Bacardi", "flavor": "Superior", "size": "375ML"},
                {"brand": "Bacardi", "flavor": "Superior", "size": "50ML"},
                {"brand": "Baileys", "flavor": "Irish Cream", "size": "375ML"},
                {"brand": "Baileys", "flavor": "Irish Cream", "size": "50ML"},
                {"brand": "Bakers", "flavor": "BBN 13YR", "size": "750ML"},
                {"brand": "Banks", "flavor": "Island Blend 5 Rum", "size": "750ML"},
                {"brand": "Basil Hayden", "flavor": "10YR", "size": "750ML"},
                {"brand": "Basil Hayden", "flavor": "", "size": "375ML"},
                {"brand": "Basil Hayden", "flavor": "", "size": "750ML"},
                {"brand": "Belvedere", "flavor": "Vodka", "size": "375ML"},
                {"brand": "Benchmark", "flavor": "", "size": "50ML"},
                {"brand": "Benjamins", "flavor": "Pineapple", "size": "50ML"},
                {"brand": "Benjamins", "flavor": "Watermelon", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "7YR", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Apple Whiskey", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Black Cherry", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Blackberry Whiskey", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Candy Cane Whiskey", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Chocolate Whiskey", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Ginger Bread", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Mesquite Brown Sugar", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Peanut Butter", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Pumpkin Spice Whiskey", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Salted Caramel", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Smores Whiskey", "size": "50ML"},
                {"brand": "Bird Dog", "flavor": "Strawberry", "size": "50ML"},
                {"brand": "Blood Oath", "flavor": "Pact No 9", "size": "750ML"},
                {"brand": "Blood Oath", "flavor": "Pact No.7", "size": "750ML"},
                {"brand": "Bombay", "flavor": "Bramble Gin", "size": "50ML"},
                {"brand": "Bomberger's", "flavor": "Declaration Bourbon", "size": "750ML"},
                {"brand": "Booker's", "flavor": "Little Book 122.6PR", "size": ""},
                {"brand": "Booker's", "flavor": "Noe 2021-02", "size": "750ML"},
                {"brand": "Bozal", "flavor": "Ensemble Mezcal", "size": "750ML"},
                {"brand": "Brinley Gold", "flavor": "Shipwreck Coconut Rum", "size": "750ML"},
                {"brand": "Brinley Gold", "flavor": "Shipwreck Coffee Rum", "size": "750ML"},
                {"brand": "Brinley Gold", "flavor": "Shipwreck Spiced Rum", "size": "750ML"},
                {"brand": "Brinley Gold", "flavor": "Vanilla", "size": "750ML"},
                {"brand": "Bushmills", "flavor": "10YR", "size": "750ML"},
                {"brand": "Buzzballz", "flavor": "Choco Tease", "size": "200ML"},
                {"brand": "Buzzballz", "flavor": "Tequila 'Rita", "size": "200ML"},
                {"brand": "Captain Morgan", "flavor": "Pineapple", "size": "750ML"},
                {"brand": "Caravella", "flavor": "Limoncello", "size": "750ML"},
                {"brand": "Ciroc", "flavor": "Snap Frost", "size": "375ML"},
                {"brand": "Ciroc", "flavor": "Snap Frost", "size": "50ML"},
                {"brand": "Clyde Mays", "flavor": "92PR", "size": "750ML"},
                {"brand": "Cognac Park", "flavor": "Pineau des Charentes", "size": "750ML"},
                {"brand": "Crown Royal", "flavor": "Apple", "size": "50ML"},
                {"brand": "Crown Royal", "flavor": "Whisky & Cola", "size": "375ML"},
                {"brand": "Cutwater", "flavor": "Mango Margarita", "size": "355ML"},
                {"brand": "DeLeón", "flavor": "Blanco Tequila", "size": "50ML"},
                {"brand": "Deep Eddy", "flavor": "Lemon Vodka", "size": "50ML"},
                {"brand": "Dewar's", "flavor": "Blended Scotch Whisky", "size": "50ML"},
                {"brand": "Disaronno", "flavor": "Amaretto", "size": "50ML"},
                {"brand": "Domaine de Canton", "flavor": "French Ginger Liqueur", "size": "750ML"},
                {"brand": "Don Julio", "flavor": "Blanco Tequila", "size": "50ML"},
                {"brand": "Dos Hombres", "flavor": "Mezcal", "size": "50ML"},
                {"brand": "Dr. McGillicuddy's", "flavor": "Fireball", "size": "375ML"},
                {"brand": "Dr. McGillicuddy's", "flavor": "Root Beer", "size": "50ML"},
                {"brand": "E&J", "flavor": "Brandy", "size": "375ML"},
                {"brand": "Eagle Rare", "flavor": "10YR", "size": "750ML"},
                {"brand": "Elijah Craig", "flavor": "Small Batch", "size": "50ML"},
                {"brand": "Elijah Craig", "flavor": "Small Batch", "size": "750ML"},
                {"brand": "Evan Williams", "flavor": "Apple", "size": "50ML"},
                {"brand": "Evan Williams", "flavor": "Honey", "size": "50ML"},
                {"brand": "Fireball", "flavor": "", "size": "375ML"},
                {"brand": "Fireball", "flavor": "", "size": "50ML"},
                {"brand": "Firefly", "flavor": "Moonshine Apple Pie", "size": "375ML"},
                {"brand": "Firefly", "flavor": "Peach Moonshine", "size": "375ML"},
                {"brand": "Firefly", "flavor": "Sweet Tea Vodka", "size": "375ML"},
                {"brand": "Firefly", "flavor": "Sweet Tea Vodka", "size": "50ML"},
                {"brand": "Firefly", "flavor": "White Lightning Moonshine", "size": "375ML"},
                {"brand": "Garrison Brothers", "flavor": "Balmorhea", "size": "750ML"},
                {"brand": "Gentleman Jack", "flavor": "", "size": "375ML"},
                {"brand": "Gentleman Jack", "flavor": "", "size": "50ML"},
                {"brand": "Ghost Tequila", "flavor": "", "size": "50ML"},
                {"brand": "Glenlivet", "flavor": "12YR", "size": "50ML"},
                {"brand": "Glenlivet", "flavor": "12YR", "size": "750ML"},
                {"brand": "Grey Goose", "flavor": "", "size": "50ML"},
                {"brand": "Hennessy", "flavor": "VS", "size": "375ML"},
                {"brand": "Hennessy", "flavor": "VS", "size": "50ML"},
                {"brand": "High West", "flavor": "American Prairie", "size": "375ML"},
                {"brand": "Hiram Walker", "flavor": "Blue Curacao", "size": "750ML"},
                {"brand": "Hiram Walker", "flavor": "Orange Curacao", "size": "750ML"},
                {"brand": "Hiram Walker", "flavor": "Triple Sec", "size": "750ML"},
                {"brand": "Hornitos", "flavor": "Plata", "size": "50ML"},
                {"brand": "Horse Soldier", "flavor": "Barrel Strength", "size": "750ML"},
                {"brand": "Jameson", "flavor": "Irish Whiskey", "size": "50ML"},
                {"brand": "Jefferson's", "flavor": "Ocean Aged", "size": "50ML"},
                {"brand": "Jefferson's", "flavor": "Reserve", "size": "750ML"},
                {"brand": "Jack Daniels", "flavor": "Single Barrel", "size": "375ML"},
                {"brand": "Jack Daniels", "flavor": "Single Barrel", "size": "50ML"},
                {"brand": "Jack Daniels", "flavor": "Tennessee Honey", "size": "50ML"},
                {"brand": "Jim Beam", "flavor": "Apple", "size": "50ML"},
                {"brand": "Jim Beam", "flavor": "Honey", "size": "50ML"},
                {"brand": "Jim Beam", "flavor": "Peach", "size": "50ML"},
                {"brand": "Jim Beam", "flavor": "Red Stag", "size": "50ML"},
                {"brand": "Jose Cuervo", "flavor": "Margarita Mix", "size": "1.75L"},
                {"brand": "Jose Cuervo", "flavor": "Tequila Silver", "size": "50ML"},
                {"brand": "Jose Cuervo", "flavor": "Tequila Tradicional", "size": "50ML"},
                {"brand": "Kinky", "flavor": "Pink", "size": "50ML"},
                {"brand": "Kinky", "flavor": "Red", "size": "50ML"},
                {"brand": "Kraken", "flavor": "Black Spiced Rum", "size": "375ML"},
                {"brand": "Kraken", "flavor": "Black Spiced Rum", "size": "50ML"},
                {"brand": "Larceny", "flavor": "BBN", "size": "750ML"},
                {"brand": "Larceny", "flavor": "SBW", "size": "750ML"},
                {"brand": "Larceny", "flavor": "Small Batch", "size": "50ML"},
                {"brand": "Larceny", "flavor": "Small Batch", "size": "750ML"},
                {"brand": "Lemon Hart", "flavor": "151", "size": "750ML"},
                {"brand": "Licor 43", "flavor": "43", "size": "50ML"},
                {"brand": "Licor 43", "flavor": "Licor 43", "size": "375ML"},
                {"brand": "Licor 43", "flavor": "Orochata", "size": "750ML"},
                {"brand": "Licor 43", "flavor": "Rojo", "size": "750ML"},
                {"brand": "Malibu", "flavor": "Black", "size": "375ML"},
                {"brand": "Malibu", "flavor": "Black", "size": "375ML"},
                {"brand": "Malibu", "flavor": "Coconut", "size": "50ML"},
                {"brand": "Maker's Mark", "flavor": "", "size": "375ML"},
                {"brand": "Maker's Mark", "flavor": "", "size": "50ML"},
                {"brand": "McCormick", "flavor": "Vodka", "size": "1.75L"},
                {"brand": "McCormick", "flavor": "Vodka", "size": "375ML"},
                {"brand": "McCormick", "flavor": "Vodka", "size": "50ML"},
                {"brand": "Milagro", "flavor": "Tequila", "size": "50ML"},
                {"brand": "New Amsterdam", "flavor": "Gin", "size": "50ML"},
                {"brand": "New Amsterdam", "flavor": "Vodka", "size": "50ML"},
                {"brand": "Old Forester", "flavor": "100 Proof", "size": "375ML"},
                {"brand": "Old Forester", "flavor": "86 Proof", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Apple Pie Moonshine", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Blackberry Moonshine", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Hunch Punch Moonshine", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Mango Habanero Whiskey", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Moonshine Cherries", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Moonshine Pickles", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Peach Moonshine", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Pineapple Moonshine", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Salty Watermelon Whiskey", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "Strawberry Mango Margarita", "size": "375ML"},
                {"brand": "Ole Smoky", "flavor": "White Lightning Moonshine", "size": "375ML"},
                {"brand": "Patron", "flavor": "Silver Tequila", "size": "50ML"},
                {"brand": "Paul Masson", "flavor": "Brandy", "size": "50ML"},
                {"brand": "Pinnacle", "flavor": "Vodka", "size": "50ML"},
                {"brand": "Proper Twelve", "flavor": "Irish Whiskey", "size": "50ML"},
                {"brand": "Rebel Yell", "flavor": "Kentucky Straight Bourbon", "size": "50ML"},
                {"brand": "Remy Martin", "flavor": "VSOP", "size": "50ML"},
                {"brand": "Remy Martin", "flavor": "VSOP", "size": "375ML"},
                {"brand": "Sailor Jerry", "flavor": "Spiced Rum", "size": "50ML"},
                {"brand": "Sailor Jerry", "flavor": "Spiced Rum", "size": "750ML"},
                {"brand": "Seagram's", "flavor": "Extra Dry Gin", "size": "375ML"},
                {"brand": "Seagram's", "flavor": "Extra Dry Gin", "size": "50ML"},
                {"brand": "Seagram's", "flavor": "Lime Twisted Gin", "size": "375ML"},
                {"brand": "Seagram's", "flavor": "Peach Twisted Gin", "size": "375ML"},
                {"brand": "Seagram's", "flavor": "Pineapple Twisted Gin", "size": "375ML"},
                {"brand": "Seagram's", "flavor": "Watermelon Twisted Gin", "size": "375ML"},
                {"brand": "Skol", "flavor": "Vodka", "size": "1.75L"},
                {"brand": "Skol", "flavor": "Vodka", "size": "375ML"},
                {"brand": "Skol", "flavor": "Vodka", "size": "50ML"},
                {"brand": "Skyy", "flavor": "Vodka", "size": "50ML"},
                {"brand": "Smirnoff", "flavor": "Citrus Vodka", "size": "50ML"},
                {"brand": "Smirnoff", "flavor": "Citrus Vodka", "size": "750ML"},
                {"brand": "Smirnoff", "flavor": "Vodka", "size": "375ML"},
                {"brand": "Smirnoff", "flavor": "Vodka", "size": "50ML"},
                {"brand": "Southern Comfort", "flavor": "Peach", "size": "50ML"},
                {"brand": "Southern Comfort", "flavor": "Whiskey", "size": "50ML"},
                {"brand": "Southern Comfort", "flavor": "Whiskey", "size": "750ML"},
                {"brand": "Stoli", "flavor": "Vodka", "size": "50ML"},
                {"brand": "Stoli", "flavor": "Vodka", "size": "750ML"},
                {"brand": "Tito's", "flavor": "Vodka", "size": "50ML"},
                {"brand": "Tito's", "flavor": "Vodka", "size": "750ML"},
                {"brand": "Triple Sec", "flavor": "", "size": "50ML"},
                {"brand": "Triple Sec", "flavor": "", "size": "750ML"},
                {"brand": "Twisted Tea", "flavor": "Half & Half", "size": "24OZ"},
                {"brand": "Twisted Tea", "flavor": "Original", "size": "24OZ"},
                {"brand": "Twisted Tea", "flavor": "Peach", "size": "24OZ"},
                {"brand": "Twisted Tea", "flavor": "Raspberry", "size": "24OZ"},
                {"brand": "Twisted Tea", "flavor": "Sweet Tea", "size": "24OZ"},
                {"brand": "Twisted Tea", "flavor": "Watermelon", "size": "24OZ"},
                {"brand": "Wild Turkey", "flavor": "101", "size": "375ML"},
                {"brand": "Wild Turkey", "flavor": "101", "size": "50ML"},
                {"brand": "Wild Turkey", "flavor": "Bourbon", "size": "375ML"},
                {"brand": "Wild Turkey", "flavor": "Bourbon", "size": "50ML"},
                {"brand": "Wild Turkey", "flavor": "Rare Breed", "size": "750ML"},
                {"brand": "Wild Turkey", "flavor": "Rye", "size": "750ML"},
                {"brand": "Woodford Reserve", "flavor": "", "size": "375ML"},
                {"brand": "Woodford Reserve", "flavor": "", "size": "50ML"},
                {"brand": "Yukon Jack", "flavor": "Honey", "size": "375ML"},
                {"brand": "Yukon Jack", "flavor": "Snakebite", "size": "375ML"}
    ]

    def _create_index(self):
            product_texts = [f"{p['brand']} {p['flavor']} {p['size']}" for p in self.products]
            embeddings = self.model.encode(product_texts)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))
            return index

    def extract_size(self, text):
        size_pattern = r'\b(\d+(\.\d+)?)\s*(ML|L|OZ)\b'
        match = re.search(size_pattern, text, re.IGNORECASE)
        return match.group() if match else None

    def match_product(self, ocr_text):
        ocr_text = ocr_text.upper()
        size = self.extract_size(ocr_text)

        if self.use_vector_db:
            return self._vector_match(ocr_text, size)
        else:
            return self._fuzzy_match(ocr_text, size)

    def _vector_match(self, ocr_text, size):
        try:
            query_vector = self.model.encode([ocr_text])
            distances, indices = self.index.search(query_vector.astype('float32'), 1)

            best_match = self.products[indices[0][0]]
            confidence = 1 / (1 + distances[0][0])

            return best_match, confidence
        except Exception as e:
            logging.error(f"Error in vector matching: {e}")
            return None, 0

    def _fuzzy_match(self, ocr_text, size):
        best_match = None
        best_score = 0

        for product in self.products:
            brand, flavor, db_size = product

            brand_score = fuzz.partial_ratio(ocr_text, brand.upper())
            flavor_score = fuzz.partial_ratio(ocr_text, flavor.upper()) if flavor else 100
            size_score = 100 if size and size.upper() == db_size.upper() else 0

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