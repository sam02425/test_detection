import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_databases(sqlite_db_path, vector_db_path):
    # Product data
    products = [
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

    # Ensure the directory exists
    os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)

    try:
        # Initialize SQLite database
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS brands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS flavors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_id INTEGER,
            flavor_id INTEGER,
            size_id INTEGER,
            full_name TEXT UNIQUE NOT NULL,
            FOREIGN KEY (brand_id) REFERENCES brands (id),
            FOREIGN KEY (flavor_id) REFERENCES flavors (id),
            FOREIGN KEY (size_id) REFERENCES sizes (id)
        );
        """)

        # Prepare data for vector database
        product_names = []
        product_ids = []

        for product in products:
            brand = product['brand']
            flavor = product['flavor'] if product['flavor'] else 'Original'
            size = product['size']

            # Insert data into SQLite
            cursor.execute("INSERT OR IGNORE INTO brands (name) VALUES (?)", (brand,))
            cursor.execute("INSERT OR IGNORE INTO flavors (name) VALUES (?)", (flavor,))
            cursor.execute("INSERT OR IGNORE INTO sizes (name) VALUES (?)", (size,))

            cursor.execute("SELECT id FROM brands WHERE name = ?", (brand,))
            brand_id = cursor.fetchone()[0]
            cursor.execute("SELECT id FROM flavors WHERE name = ?", (flavor,))
            flavor_id = cursor.fetchone()[0]
            cursor.execute("SELECT id FROM sizes WHERE name = ?", (size,))
            size_id = cursor.fetchone()[0]

            full_name = f"{brand} {flavor} {size}".strip()
            cursor.execute("""
            INSERT OR IGNORE INTO products (brand_id, flavor_id, size_id, full_name)
            VALUES (?, ?, ?, ?)
            """, (brand_id, flavor_id, size_id, full_name))

            # Get the product ID for vector database
            cursor.execute("SELECT id FROM products WHERE full_name = ?", (full_name,))
            product_id = cursor.fetchone()[0]

            # Prepare data for vector database
            product_names.append(full_name)
            product_ids.append(product_id)

        conn.commit()
        conn.close()

        logging.info("SQLite database initialized successfully!")

        # Initialize vector database
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        embeddings = model.encode(product_names)

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        # Save FAISS index
        faiss.write_index(index, vector_db_path + ".index")

        # Save product IDs and names
        with open(vector_db_path + "_metadata.pkl", "wb") as f:
            pickle.dump({"product_ids": product_ids, "product_names": product_names}, f)

        logging.info("Vector database initialized successfully!")

    except sqlite3.Error as e:
        logging.error(f"SQLite error occurred: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    sqlite_db_path = os.path.join(current_dir, "liquor_database.db")
    vector_db_path = os.path.join(current_dir, "liquor_vector_db")

    logging.info(f"Initializing databases in: {current_dir}")
    logging.info(f"SQLite DB path: {sqlite_db_path}")
    logging.info(f"Vector DB path: {vector_db_path}")

    initialize_databases(sqlite_db_path, vector_db_path)