import sqlite3
import os

def initialize_database(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        brand TEXT NOT NULL,
        flavor TEXT,
        size TEXT
    )
    ''')

    # Add some sample data
    sample_products = [
    ('1792', 'Bottled In Bond', '750ML'),
    ('1800', 'Coconut Tequila', '750ML'),
    ('99', 'Apple', '100ML'),
    ('Absolut', '80', '375ML'),
    ('Admiral Nelson\'s', 'Spiced Rum', '1.75L'),
    ('Avion', 'Silver Tequila', '750ML'),
    ('Bacardi', 'Superior', '50ML'),
    ('Baileys', 'Irish Cream', '375ML'),
    ('Basil Hayden', '10YR', '750ML'),
    ('Bird Dog', 'Black Cherry', '50ML'),
    ('Bombay', 'Bramble Gin', '50ML'),
    ('Captain Morgan', 'Pineapple', '750ML'),
    ('Crown Royal', 'Apple', '50ML'),
    ('Don Julio', 'Blanco Tequila', '50ML'),
    ('Eagle Rare', '10YR', '750ML'),
    ('Fireball', '', '375ML'),
    ('Gentleman Jack', '', '375ML'),
    ('Grey Goose', '', '50ML'),
    ('Hennessy', 'VS', '375ML'),
    ('Jack Daniels', 'Single Barrel', '375ML'),
    ('Jim Beam', 'Apple', '50ML'),
    ('Jose Cuervo', 'Tequila Silver', '50ML'),
    ('Maker\'s Mark', '', '375ML'),
    ('Ole Smoky', 'Apple Pie Moonshine', '375ML'),
    ('Patron', 'Silver Tequila', '50ML'),
    ('Smirnoff', 'Citrus Vodka', '750ML'),
    ('Southern Comfort', 'Whiskey', '750ML'),
    ('Tito\'s', 'Vodka', '750ML'),
    ('Wild Turkey', '101', '375ML'),
    ('Woodford Reserve', '', '375ML')
]

    cursor.executemany('INSERT INTO products (brand, flavor, size) VALUES (?, ?, ?)', sample_products)

    conn.commit()
    conn.close()

    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'product_database.db')
    initialize_database(db_path)