import sqlite3
import os

def create_alcohol_product_database():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'alcohol_product_database.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS brands
                 (id INTEGER PRIMARY KEY, name TEXT UNIQUE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS brand_keywords
                 (id INTEGER PRIMARY KEY, brand_id INTEGER, keyword TEXT,
                  FOREIGN KEY (brand_id) REFERENCES brands(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS flavors
                 (id INTEGER PRIMARY KEY, brand_id INTEGER, name TEXT,
                  FOREIGN KEY (brand_id) REFERENCES brands(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS flavor_keywords
                 (id INTEGER PRIMARY KEY, flavor_id INTEGER, keyword TEXT,
                  FOREIGN KEY (flavor_id) REFERENCES flavors(id))''')

    # Insert sample data
    brands_data = [
        ("1792", ["1792"]),
        ("1800", ["1800"]),
        ("360", ["360"]),
        ("99", ["99"]),
        ("Absolut", ["absolut"]),
        ("Admiral Nelson's", ["admiral", "nelson", "nelsons"]),
        ("Avion", ["avion"]),
        ("Bacardi", ["bacardi"]),
        ("Bailey's", ["bailey", "baileys"]),
        ("Baker's", ["baker", "bakers"]),
        ("Basil Hayden", ["basil", "hayden"]),
        ("Belvedere", ["belvedere"]),
        ("Bird Dog", ["bird", "dog", "birddog"]),
        ("Buffalo Trace", ["buffalo", "trace"]),
        ("Bulleit", ["bulleit"]),
        ("Calumet Farm", ["calumet", "farm"]),
        ("Captain Morgan", ["captain", "morgan"]),
        ("Crown Royal", ["crown", "royal"]),
        ("DeKuyper", ["dekuyper"]),
        ("Don Julio", ["don", "julio"]),
        ("Early Times", ["early", "times"]),
        ("Elijah Craig", ["elijah", "craig"]),
        ("Evan Williams", ["evan", "williams"]),
        ("Fireball", ["fireball"]),
        ("Jack Daniel's", ["jack", "daniels"]),
        ("Jim Beam", ["jim", "beam"]),
        ("Johnnie Walker", ["johnnie", "walker"]),
        ("Jose Cuervo", ["jose", "cuervo"]),
        ("Maker's Mark", ["makers", "mark"]),
        ("Old Forester", ["old", "forester"]),
        ("Patron", ["patron"]),
        ("Smirnoff", ["smirnoff"]),
        ("Wild Turkey", ["wild", "turkey"]),
        ("Woodford Reserve", ["woodford", "reserve"])
    ]

    flavors_data = [
        ("1792", "Bottled in Bond", ["bottled", "bond"]),
        ("1792", "Full Proof", ["full", "proof"]),
        ("1792", "Small Batch", ["small", "batch"]),
        ("1792", "Sweet Wheat", ["sweet", "wheat"]),
        ("1800", "Coconut", ["coconut"]),
        ("1800", "Cristalino Anejo", ["cristalino", "anejo"]),
        ("1800", "Silver Blanco", ["silver", "blanco"]),
        ("360", "Blue Raspberry", ["blue", "raspberry"]),
        ("360", "Pineapple", ["pineapple"]),
        ("99", "Apple", ["apple"]),
        ("99", "Bananas", ["bananas"]),
        ("99", "Butterscotch", ["butterscotch"]),
        ("99", "Cinnamon", ["cinnamon"]),
        ("Absolut", "Original", ["original"]),
        ("Admiral Nelson's", "Spiced Rum", ["spiced", "rum"]),
        ("Avion", "Silver", ["silver"]),
        ("Bacardi", "Gold", ["gold"]),
        ("Bacardi", "Superior", ["superior"]),
        ("Bailey's", "Irish Cream", ["irish", "cream"]),
        ("Baker's", "Bourbon", ["bourbon"]),
        ("Basil Hayden", "Original", ["original"]),
        ("Belvedere", "Vodka", ["vodka"]),
        ("Bird Dog", "Apple", ["apple"]),
        ("Bird Dog", "Black Cherry", ["black", "cherry"]),
        ("Bird Dog", "Blackberry", ["blackberry"]),
        ("Buffalo Trace", "Bourbon", ["bourbon"]),
        ("Bulleit", "Bourbon", ["bourbon"]),
        ("Calumet Farm", "Bourbon", ["bourbon"]),
        ("Captain Morgan", "Spiced Rum", ["spiced", "rum"]),
        ("Crown Royal", "Original", ["original"]),
        ("Crown Royal", "Apple", ["apple"]),
        ("Crown Royal", "Peach", ["peach"]),
        ("DeKuyper", "Apple Pucker", ["apple", "pucker"]),
        ("DeKuyper", "Buttershots", ["buttershots"]),
        ("DeKuyper", "Hot Damn", ["hot", "damn"]),
        ("Don Julio", "Blanco", ["blanco"]),
        ("Don Julio", "Reposado", ["reposado"]),
        ("Early Times", "Original", ["original"]),
        ("Elijah Craig", "Small Batch", ["small", "batch"]),
        ("Evan Williams", "Original", ["original"]),
        ("Fireball", "Cinnamon", ["cinnamon"]),
        ("Jack Daniel's", "Old No. 7", ["old", "no", "7"]),
        ("Jack Daniel's", "Apple", ["apple"]),
        ("Jack Daniel's", "Fire", ["fire"]),
        ("Jack Daniel's", "Honey", ["honey"]),
        ("Jim Beam", "Original", ["original"]),
        ("Jim Beam", "Apple", ["apple"]),
        ("Jim Beam", "Honey", ["honey"]),
        ("Johnnie Walker", "Black Label", ["black", "label"]),
        ("Johnnie Walker", "Red Label", ["red", "label"]),
        ("Jose Cuervo", "Gold", ["gold"]),
        ("Jose Cuervo", "Silver", ["silver"]),
        ("Maker's Mark", "Original", ["original"]),
        ("Old Forester", "Original", ["original"]),
        ("Patron", "Silver", ["silver"]),
        ("Patron", "Reposado", ["reposado"]),
        ("Smirnoff", "Original", ["original"]),
        ("Smirnoff", "Blue Raspberry", ["blue", "raspberry"]),
        ("Smirnoff", "Raspberry", ["raspberry"]),
        ("Wild Turkey", "101", ["101"]),
        ("Wild Turkey", "American Honey", ["american", "honey"]),
        ("Woodford Reserve", "Original", ["original"]),
        ("Woodford Reserve", "Double Oaked", ["double", "oaked"])
    ]

    for brand, keywords in brands_data:
        c.execute("INSERT OR IGNORE INTO brands (name) VALUES (?)", (brand,))
        brand_id = c.lastrowid
        for keyword in keywords:
            c.execute("INSERT INTO brand_keywords (brand_id, keyword) VALUES (?, ?)", (brand_id, keyword))

    for brand, flavor, keywords in flavors_data:
        c.execute("SELECT id FROM brands WHERE name = ?", (brand,))
        brand_id = c.fetchone()[0]
        c.execute("INSERT INTO flavors (brand_id, name) VALUES (?, ?)", (brand_id, flavor))
        flavor_id = c.lastrowid
        for keyword in keywords:
            c.execute("INSERT INTO flavor_keywords (flavor_id, keyword) VALUES (?, ?)", (flavor_id, keyword))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_alcohol_product_database()

# import sqlite3
# import os

# def create_product_database():
#     db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'product_database.db')
#     conn = sqlite3.connect(db_path)
#     c = conn.cursor()

#     # Create tables
#     c.execute('''CREATE TABLE IF NOT EXISTS brands
#                  (id INTEGER PRIMARY KEY, name TEXT UNIQUE)''')
#     c.execute('''CREATE TABLE IF NOT EXISTS brand_keywords
#                  (id INTEGER PRIMARY KEY, brand_id INTEGER, keyword TEXT,
#                   FOREIGN KEY (brand_id) REFERENCES brands(id))''')
#     c.execute('''CREATE TABLE IF NOT EXISTS flavors
#                  (id INTEGER PRIMARY KEY, brand_id INTEGER, name TEXT,
#                   FOREIGN KEY (brand_id) REFERENCES brands(id))''')
#     c.execute('''CREATE TABLE IF NOT EXISTS flavor_keywords
#                  (id INTEGER PRIMARY KEY, flavor_id INTEGER, keyword TEXT,
#                   FOREIGN KEY (flavor_id) REFERENCES flavors(id))''')

#     # Insert sample data
#     brands_data = [
#     ("Bargs", ["bargs"]),
#     ("Bird Dog", ["bird", "dog", "birddog"]),
#     ("Bueno", ["bueno"]),
#     ("Cheetos", ["cheetos"]),
#     ("Coca-Cola", ["coca", "cola", "coke"]),
#     ("Chesters", ["chesters"]),
#     ("Chips Ahoy", ["chips", "ahoy", "chipsahoy"]),
#     ("Crunch", ["crunch"]),
#     ("Crush", ["crush"]),
#     ("Dekuyper", ["dekuyper"]),
#     ("Dr Pepper", ["dr", "pepper", "drpepper"]),
#     ("Early Times", ["early", "times", "earlytimes"]),
#     ("Fanta", ["fanta"]),
#     ("Frito-Lay", ["frito", "lay", "fritolay"]),
#     ("Fritos", ["fritos"]),
#     ("Funyuns", ["funyuns"]),
#     ("Grandma's", ["grandma", "grandmas"]),
#     ("Jim Beam", ["jim", "beam", "jimbeam"]),
#     ("Lay's", ["lay", "lays"]),
#     ("Lenny Larry's", ["lenny", "larry", "lennylarrys"]),
#     ("Minute Maid", ["minute", "maid", "minutemaid"]),
#     ("Mountain Dew", ["mountain", "dew", "mountaindew"]),
#     ("Munchies", ["munchies"]),
#     ("Munchos", ["munchos"]),
#     ("Nerds", ["nerds"]),
#     ("Nut Harvest", ["nut", "harvest", "nutharvest"]),
#     ("Oreo", ["oreo"]),
#     ("Payday", ["payday"]),
#     ("Popcorners", ["popcorners"]),
#     ("Quaker", ["quaker"]),
#     ("Ruffles", ["ruffles"]),
#     ("Sabritas", ["sabritas"]),
#     ("Sabritones", ["sabritones"]),
#     ("Skittles", ["skittles"]),
#     ("Smartfood", ["smartfood"]),
#     ("Sour Punch", ["sour", "punch", "sourpunch"]),
#     ("Sprite", ["sprite"]),
#     ("Sun Chips", ["sun", "chips", "sunchips"]),
#     ("Whatchamacallit", ["whatchamacallit"])
#     ]

#     flavors_data = [
#         ("Bargs", "Black", ["black"]),
#         ("Bird Dog", "Apple", ["apple"]),
#         ("Bird Dog", "Black Cherry", ["black", "cherry"]),
#         ("Bird Dog", "Salted Caramel Whiskey", ["salted", "caramel", "whiskey"]),
#         ("Bird Dog", "Strawberry", ["strawberry"]),
#         ("Bueno", "Share Size", ["share", "size"]),
#         ("Cheetos", "Crunchy", ["crunchy"]),
#         ("Cheetos", "Crunchy Flamin Hot", ["crunchy", "flamin", "hot"]),
#         ("Cheetos", "Crunchy Flamin Hot Limon", ["crunchy", "flamin", "hot", "limon"]),
#         ("Cheetos", "Crunchy XXTRA Flamin Hot", ["crunchy", "xxtra", "flamin", "hot"]),
#         ("Cheetos", "Crunchy XXTRA Flamin Hot Limon", ["crunchy", "xxtra", "flamin", "hot", "limon"]),
#         ("Cheetos", "Flamin Hot Puffs", ["flamin", "hot", "puffs"]),
#         ("Cheetos", "Puffs", ["puffs"]),
#         ("Cheetos", "Puffs White Cheddar", ["puffs", "white", "cheddar"]),
#         ("Coca-Cola", "Cherry", ["cherry"]),
#         ("Coca-Cola", "Cherry Vanilla", ["cherry", "vanilla"]),
#         ("Coca-Cola", "Original", ["original"]),
#         ("Coca-Cola", "Spiced", ["spiced"]),
#         ("Coca-Cola", "Zero", ["zero"]),
#         ("Coca-Cola", "Diet", ["diet"]),
#         ("Coca-Cola", "Vanilla", ["vanilla"]),
#         ("Chesters", "Fries Flamin Hot", ["fries", "flamin", "hot"]),
#         ("Chips Ahoy", "Original", ["original"]),
#         ("Chips Ahoy", "King Size", ["king", "size"]),
#         ("Crush", "Original", ["original"]),
#         ("Dekuyper", "Buttershots", ["buttershots"]),
#         ("Dekuyper", "Hot Damn 100 Proof", ["hot", "damn", "100", "proof"]),
#         ("Dekuyper", "Hot Damn 30 Proof", ["hot", "damn", "30", "proof"]),
#         ("Dekuyper", "Peachtree", ["peachtree"]),
#         ("Dekuyper", "Peppermint 100 Proof", ["peppermint", "100", "proof"]),
#         ("Dr Pepper", "Original", ["original"]),
#         ("Early Times", "Original", ["original"]),
#         ("Fanta", "Original", ["original"]),
#         ("Fanta", "Grape", ["grape"]),
#         ("Fanta", "Zero", ["zero"]),
#         ("Frito-Lay", "BBQ Sunflower Seeds", ["bbq", "sunflower", "seeds"]),
#         ("Frito-Lay", "Ranch Sunflower Seeds", ["ranch", "sunflower", "seeds"]),
#         ("Fritos", "Chili Cheese", ["chili", "cheese"]),
#         ("Funyuns", "Onion Flavored Rings", ["onion", "flavored", "rings"]),
#         ("Funyuns", "Onion Flavored Rings Flamin Hot", ["onion", "flavored", "rings", "flamin", "hot"]),
#         ("Grandma's", "Chocolate Brownie Cookies", ["chocolate", "brownie", "cookies"]),
#         ("Grandma's", "Chocolate Chip Cookies", ["chocolate", "chip", "cookies"]),
#         ("Grandma's", "Mini Chocolate Chip Cookies", ["mini", "chocolate", "chip", "cookies"]),
#         ("Grandma's", "Mini Sandwich Cremes Vanilla Flavored Cookies", ["mini", "sandwich", "cremes", "vanilla", "flavored", "cookies"]),
#         ("Grandma's", "Oatmeal Raisin Cookies", ["oatmeal", "raisin", "cookies"]),
#         ("Grandma's", "Peanut Butter Cookies", ["peanut", "butter", "cookies"]),
#         ("Grandma's", "Sandwich Creme Peanut Butter Cookies", ["sandwich", "creme", "peanut", "butter", "cookies"]),
#         ("Grandma's", "Sandwich Creme Vanilla Flavored Cookies", ["sandwich", "creme", "vanilla", "flavored", "cookies"]),
#         ("Jim Beam", "Apple", ["apple"]),
#         ("Jim Beam", "Bourbon", ["bourbon"]),
#         ("Jim Beam", "Fire", ["fire"]),
#         ("Jim Beam", "Honey", ["honey"]),
#         ("Jim Beam", "Red Stag", ["red", "stag"]),
#         ("Lay's", "Barbecue", ["barbecue", "bbq"]),
#         ("Lay's", "Classic", ["classic"]),
#         ("Lay's", "Kettle Cooked Jalapeno", ["kettle", "cooked", "jalapeno"]),
#         ("Lay's", "Limon", ["limon"]),
#         ("Lenny Larry's", "Birthday Cake", ["birthday", "cake"]),
#         ("Lenny Larry's", "Chocolate Chips", ["chocolate", "chips"]),
#         ("Lenny Larry's", "Double Chocolate Chips", ["double", "chocolate", "chips"]),
#         ("Lenny Larry's", "Peanut Butter", ["peanut", "butter"]),
#         ("Lenny Larry's", "Peanut Butter Chocolate Chips", ["peanut", "butter", "chocolate", "chips"]),
#         ("Lenny Larry's", "Snickerdoodle", ["snickerdoodle"]),
#         ("Minute Maid", "Blue Raspberry", ["blue", "raspberry"]),
#         ("Minute Maid", "Fruit Punch", ["fruit", "punch"]),
#         ("Minute Maid", "Lemonade", ["lemonade"]),
#         ("Minute Maid", "Pink Lemonade", ["pink", "lemonade"]),
#         ("Mountain Dew", "Original", ["original"]),
#         ("Munchies", "Flaming Hot Peanuts", ["flaming", "hot", "peanuts"]),
#         ("Munchies", "Honey Roasted Peanuts", ["honey", "roasted", "peanuts"]),
#         ("Munchies", "Peanut Butter", ["peanut", "butter"]),
#         ("Munchos", "Potato Crisps", ["potato", "crisps"]),
#         ("Nerds", "Share Size", ["share", "size"]),
#         ("Nut Harvest", "Trail Mix Cocoa Dusted Whole Almonds", ["trail", "mix", "cocoa", "dusted", "whole", "almonds"]),
#         ("Nut Harvest", "Trail Mix Deluxe Salted Mixed Nuts", ["trail", "mix", "deluxe", "salted", "mixed", "nuts"]),
#         ("Nut Harvest", "Trail Mix Honey Roasted Whole Cashews", ["trail", "mix", "honey", "roasted", "whole", "cashews"]),
#         ("Nut Harvest", "Trail Mix Lightly Roasted Whole Almonds", ["trail", "mix", "lightly", "roasted", "whole", "almonds"]),
#         ("Nut Harvest", "Trail Mix Nut Chocolate Sweet Salty", ["trail", "mix", "nut", "chocolate", "sweet", "salty"]),
#         ("Nut Harvest", "Trail Mix Nut Fruit Sweet Salty", ["trail", "mix", "nut", "fruit", "sweet", "salty"]),
#         ("Nut Harvest", "Trail Mix Sea Salted In Shell Pistachios", ["trail", "mix", "sea", "salted", "in", "shell", "pistachios"]),
#         ("Nut Harvest", "Trail Mix Sea Salted Whole Cashews", ["trail", "mix", "sea", "salted", "whole", "cashews"]),
#         ("Nut Harvest", "Trail Mix Spicy Flavored Pistachios", ["trail", "mix", "spicy", "flavored", "pistachios"]),
#         ("Oreo", "Original", ["original"]),
#         ("Oreo", "Double Stuf King Size", ["double", "stuf", "king", "size"]),
#         ("Oreo", "King Size", ["king", "size"]),
#         ("Payday", "Share Size", ["share", "size"]),
#         ("Popcorners", "Kettle Corn", ["kettle", "corn"]),
#         ("Popcorners", "Sea Salt", ["sea", "salt"]),
#         ("Popcorners", "White Cheddar", ["white", "cheddar"]),
#         ("Quaker", "Caramel Rice Crisps", ["caramel", "rice", "crisps"]),
#         ("Ruffles", "Cheddar Sour Cream", ["cheddar", "sour", "cream"]),
#         ("Ruffles", "Queso", ["queso"]),
#         ("Sabritas", "Japanese", ["japanese"]),
#         ("Sabritas", "Salt Lime", ["salt", "lime"]),
#         ("Sabritas", "Spicy Picante", ["spicy", "picante"]),
#         ("Sabritones", "Chile Lime", ["chile", "lime"]),
#         ("Skittles", "Share Size", ["share", "size"]),
#         ("Smartfood", "White Cheddar", ["white", "cheddar"]),
#         ("Sour Punch", "Share Size", ["share", "size"]),
#         ("Sprite", "Original", ["original"]),
#         ("Sprite", "Cherry", ["cherry"]),
#         ("Sprite", "Tropical Mix", ["tropical", "mix"]),
#         ("Sprite", "Zero", ["zero"]),
#         ("Sun Chips", "Harvest Cheddar", ["harvest", "cheddar"]),
#         ("Whatchamacallit", "King Size", ["king", "size"])
#     ]

#     for brand, keywords in brands_data:
#         c.execute("INSERT OR IGNORE INTO brands (name) VALUES (?)", (brand,))
#         brand_id = c.lastrowid
#         for keyword in keywords:
#             c.execute("INSERT INTO brand_keywords (brand_id, keyword) VALUES (?, ?)", (brand_id, keyword))

#     for brand, flavor, keywords in flavors_data:
#         c.execute("SELECT id FROM brands WHERE name = ?", (brand,))
#         brand_id = c.fetchone()[0]
#         c.execute("INSERT INTO flavors (brand_id, name) VALUES (?, ?)", (brand_id, flavor))
#         flavor_id = c.lastrowid
#         for keyword in keywords:
#             c.execute("INSERT INTO flavor_keywords (flavor_id, keyword) VALUES (?, ?)", (flavor_id, keyword))

#     conn.commit()
#     conn.close()

# if __name__ == "__main__":
#     create_product_database()