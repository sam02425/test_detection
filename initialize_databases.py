import sqlite3
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

def initialize_databases(db_path, vector_db_path):
    # Initialize SQLite database
    conn = sqlite3.connect(db_path)
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

    # Product class names
    class_names = [
        "BargsBlack20Oz", "BuenoShareSize", "CheetosCrunchy", "CheetosCrunchyFlaminHot",
        "CheetosCrunchyFlaminHotLimon", "CheetosCrunchyXXTRAFlaminHot",
        "CheetosCrunchyXXTRAFlaminHotLimon", "CheetosPuffs", "CherryCocaCola20Oz",
        "CherryVanillaCocaCola20Oz", "ChipsAhoy", "ChipsAhoyKingSize", "CocaCola16Oz",
        "CocaCola20Oz", "CocaCola350Ml", "CocaColaSpiced20Oz", "CocaColaZero16Oz",
        "Crunch", "Crush16Oz", "DietCocaCola20Oz", "DietCokeCan16Oz", "DoritosCoolRanch",
        "DoritosFlaminHotCoolRanch", "DoritosNachoCheese", "DoritosSpicyNacho",
        "DrPapPepper1L", "DrPapPepper20Oz", "DrPapPepperCan16Oz", "Fanta20Oz",
        "FantaCan16Oz", "FantaGrape20Oz", "FantaZero20Oz",
        "FunyunsOnionFlavoredRingsFlaminHot", "LaysBarbecue", "LaysClassic", "LaysLimon",
        "LennyLarrysBirthdayCake", "LennyLarrysChocolateChips",
        "LennyLarrysDoubleChocolateChips", "LennyLarrysPeanutButter",
        "LennyLarrysPeanutButterChocolateChips", "LennyLarrysSnickerdoodle",
        "MinuteMaidBlueRaspberry20Oz", "MinuteMaidFruitPunch", "MinuteMaidLemonade20Oz",
        "MinuteMaidPinkLemonade", "MountainDew16Oz", "MtnDew16Oz", "NerdsShareSize",
        "Oreo", "OreoDoubleStufKingSize", "OreoKingSize", "PaydayShareSize",
        "SkittlesShareSize", "SmartfoodWhiteCheddar", "SourPunchShareSize", "Sprite20Oz",
        "Sprite40Oz", "SpriteCan16Oz", "SpriteCherry20Oz", "SpriteTropicalMix20Oz",
        "SpriteZero20Oz", "SunChipsHarvestCheddar", "VanillaCocaCola20Oz",
        "WhatchamacallitKingSize", "ZeroCocaCola16Oz"
    ]

    # Prepare data for vector database
    product_names = []
    product_ids = []

    for product_name in class_names:
        # Extract brand, flavor, and size
        parts = re.findall('[A-Z][^A-Z]*', product_name)

        if len(parts) == 1:
            brand = parts[0]
            flavor = "Original"
            size = "Regular"
        elif len(parts) == 2:
            brand, flavor = parts
            size = "Regular"
        else:
            brand = parts[0]
            size = parts[-1] if any(char.isdigit() for char in parts[-1]) else "Regular"
            flavor = " ".join(parts[1:-1]) if size != "Regular" else " ".join(parts[1:])

        # Special cases
        if product_name.endswith("ShareSize"):
            size = "ShareSize"
            flavor = flavor.replace("Share Size", "").strip()
        if "KingSize" in product_name:
            size = "KingSize"
            flavor = flavor.replace("King Size", "").strip()
        if brand == "Dr":
            brand = "Dr Pepper"
            flavor = flavor.replace("Pap Pepper", "").strip()
        if brand == "Mtn":
            brand = "Mountain Dew"

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

        cursor.execute("""
        INSERT OR IGNORE INTO products (brand_id, flavor_id, size_id, full_name)
        VALUES (?, ?, ?, ?)
        """, (brand_id, flavor_id, size_id, product_name))

        # Get the product ID for vector database
        cursor.execute("SELECT id FROM products WHERE full_name = ?", (product_name,))
        product_id = cursor.fetchone()[0]

        # Prepare data for vector database
        product_names.append(product_name)
        product_ids.append(product_id)

    conn.commit()
    conn.close()

    print("SQLite database initialized successfully!")

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

    print("Vector database initialized successfully!")

if __name__ == "__main__":
    sqlite_db_path = "product_database.db"
    vector_db_path = "product_vector_db"
    initialize_databases(sqlite_db_path, vector_db_path)