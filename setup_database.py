import sqlite3

def initialize_database():
    # SQL script as a multi-line string
    sql_script = """
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        brand TEXT,
        flavor TEXT,
        size TEXT
    );

    INSERT INTO products (brand, flavor, size) VALUES
        ('Absolut', 'Citron', '375ML'),
        ('Absolut', 'Lime', '50ML'),
        ('Absolut', 'Peppar', '50ML'),
        ('Bacardi', 'Gold', '375ML'),
        ('Bacardi', 'Superior', '50ML'),
        ('Captain Morgan', 'Spiced Rum', '50ML'),
        ('Crown Royal', 'Apple', '50ML'),
        ('Crown Royal', 'Vanilla', '375ML'),
        ('Fireball', 'Cinnamon Whisky', '375ML'),
        ('Fireball', 'Cinnamon Whisky', '50ML'),
        ('Jack Daniels', 'Old No. 7', '50ML'),
        ('Jack Daniels', 'Honey', '375ML'),
        ('Jack Daniels', 'Tennessee Fire', '50ML'),
        ('Jameson', 'Irish Whiskey', '50ML'),
        ('Jameson', 'Irish Whiskey', '375ML'),
        ('Jim Beam', 'Bourbon', '50ML'),
        ('Jim Beam', 'Apple', '375ML'),
        ('Jim Beam', 'Peach', '50ML'),
        ('Jose Cuervo', 'Especial Gold Tequila', '50ML'),
        ('Jose Cuervo', 'Especial Silver Tequila', '375ML'),
        ('Malibu', 'Coconut', '50ML'),
        ('Malibu', 'Black', '375ML'),
        ('Maker''s Mark', '', '375ML'),
        ('Maker''s Mark', '', '50ML'),
        ('McCormick', 'Vodka', '1.75L'),
        ('McCormick', 'Vodka', '375ML'),
        ('McCormick', 'Vodka', '50ML'),
        ('Milagro', 'Tequila', '50ML'),
        ('New Amsterdam', 'Gin', '50ML'),
        ('New Amsterdam', 'Vodka', '50ML'),
        ('Old Forester', '100 Proof', '375ML'),
        ('Old Forester', '86 Proof', '375ML'),
        ('Ole Smoky', 'Apple Pie Moonshine', '375ML'),
        ('Ole Smoky', 'Blackberry Moonshine', '375ML'),
        ('Ole Smoky', 'Hunch Punch Moonshine', '375ML'),
        ('Ole Smoky', 'Mango Habanero Whiskey', '375ML'),
        ('Ole Smoky', 'Moonshine Cherries', '375ML'),
        ('Ole Smoky', 'Moonshine Pickles', '375ML'),
        ('Ole Smoky', 'Peach Moonshine', '375ML'),
        ('Ole Smoky', 'Pineapple Moonshine', '375ML'),
        ('Ole Smoky', 'Salty Watermelon Whiskey', '375ML'),
        ('Ole Smoky', 'Strawberry Mango Margarita', '375ML'),
        ('Ole Smoky', 'White Lightning Moonshine', '375ML'),
        ('Patron', 'Silver Tequila', '50ML'),
        ('Paul Masson', 'Brandy', '50ML'),
        ('Pinnacle', 'Vodka', '50ML'),
        ('Proper Twelve', 'Irish Whiskey', '50ML'),
        ('Rebel Yell', 'Kentucky Straight Bourbon', '50ML'),
        ('Remy Martin', 'VSOP', '50ML'),
        ('Remy Martin', 'VSOP', '375ML'),
        ('Sailor Jerry', 'Spiced Rum', '50ML'),
        ('Sailor Jerry', 'Spiced Rum', '750ML'),
        ('Seagram''s', 'Extra Dry Gin', '375ML'),
        ('Seagram''s', 'Extra Dry Gin', '50ML'),
        ('Seagram''s', 'Lime Twisted Gin', '375ML'),
        ('Seagram''s', 'Peach Twisted Gin', '375ML'),
        ('Seagram''s', 'Pineapple Twisted Gin', '375ML'),
        ('Seagram''s', 'Watermelon Twisted Gin', '375ML'),
        ('Skol', 'Vodka', '1.75L'),
        ('Skol', 'Vodka', '375ML'),
        ('Skol', 'Vodka', '50ML'),
        ('Skyy', 'Vodka', '50ML'),
        ('Smirnoff', 'Citrus Vodka', '50ML'),
        ('Smirnoff', 'Citrus Vodka', '750ML'),
        ('Smirnoff', 'Vodka', '375ML'),
        ('Smirnoff', 'Vodka', '50ML'),
        ('Southern Comfort', 'Peach', '50ML'),
        ('Southern Comfort', 'Whiskey', '50ML'),
        ('Southern Comfort', 'Whiskey', '750ML'),
        ('Stoli', 'Vodka', '50ML'),
        ('Stoli', 'Vodka', '750ML'),
        ('Tito''s', 'Vodka', '50ML'),
        ('Tito''s', 'Vodka', '750ML'),
        ('Triple Sec', '', '50ML'),
        ('Triple Sec', '', '750ML'),
        ('Twisted Tea', 'Half & Half', '24OZ'),
        ('Twisted Tea', 'Original', '24OZ'),
        ('Twisted Tea', 'Peach', '24OZ'),
        ('Twisted Tea', 'Raspberry', '24OZ'),
        ('Twisted Tea', 'Sweet Tea', '24OZ'),
        ('Twisted Tea', 'Watermelon', '24OZ'),
        ('Wild Turkey', '101', '375ML'),
        ('Wild Turkey', '101', '50ML'),
        ('Wild Turkey', 'Bourbon', '375ML'),
        ('Wild Turkey', 'Bourbon', '50ML'),
        ('Wild Turkey', 'Rare Breed', '750ML'),
        ('Wild Turkey', 'Rye', '750ML'),
        ('Woodford Reserve', '', '375ML'),
        ('Woodford Reserve', '', '50ML'),
        ('Yukon Jack', 'Honey', '375ML'),
        ('Yukon Jack', 'Snakebite', '375ML');
    """

    # Connect to the database (this will create it if it doesn't exist)
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()

    # Execute the SQL script
    cursor.executescript(sql_script)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Database initialized successfully.")

if __name__ == "__main__":
    initialize_database()