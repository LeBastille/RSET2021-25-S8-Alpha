from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy # type: ignore
import json
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer as AutoTokenizerForSentiment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# Add configuration for serving images from the earphone images directory
app.config['EARPHONE_IMAGES_FOLDER'] = 'earphone images'
app.config['LAPTOP_IMAGES_FOLDER'] = 'laptops'
app.config['SMARTWATCH_IMAGES_FOLDER'] = 'smartwatches'
app.config['PHONE_IMAGES_FOLDER'] = 'mobiles'
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Initialize the BERT model and tokenizer
model_name = "facebook/bart-base"  # Changed to a more stable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize sentiment analysis model
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizerForSentiment.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    favorites = db.relationship('Favorite', backref='user', lazy=True)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_name = db.Column(db.String(200), nullable=False)
    product_category = db.Column(db.String(50), nullable=False)
    product_image = db.Column(db.String(200))
    product_price = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load products from JSON files
def load_products():
    try:
        with open('products.json', 'r') as f:
            phones = json.load(f)
    except FileNotFoundError:
        print("Warning: products.json not found. Using empty phone list.")
        phones = {}
    
    try:
        with open('laptops.json', 'r') as f:
            laptops = json.load(f)
    except FileNotFoundError:
        print("Warning: laptops.json not found. Using empty laptop list.")
        laptops = {}
    
    try:
        with open('smartwatches.json', 'r') as f:
            smartwatches = json.load(f)
    except FileNotFoundError:
        print("Warning: smartwatches.json not found. Using empty smartwatch list.")
        smartwatches = {}
    
    try:
        with open('earphones.json', 'r') as f:
            earphones = json.load(f)
    except FileNotFoundError:
        print("Warning: earphones.json not found. Using empty earphone list.")
        earphones = {}
    
    return phones, laptops, smartwatches, earphones

# Load products at startup
phones, laptops, smartwatches, earphones = load_products()

# Define criteria and weights for each category
phone_criteria = ["Processor", "Battery Life", "Screen", "Camera", "Price"]
phone_weights = [0.3, 0.2, 0.15, 0.15, 0.2]
phone_beneficial = [True, True, True, True, False]

laptop_criteria = ["Processor", "RAM", "Battery Life", "GPU", "Price"]
laptop_weights = [0.3, 0.2, 0.15, 0.15, 0.2]
laptop_beneficial = [True, True, True, True, False]

smartwatch_criteria = ["Display", "Battery Life", "GPS", "Water Resistance", "Price"]
smartwatch_weights = [0.25, 0.25, 0.15, 0.15, 0.2]
smartwatch_beneficial = [True, True, True, True, False]

earphone_criteria = ["Driver Size", "Battery Life", "Noise Cancellation", "Water Resistance", "Price"]
earphone_weights = [0.25, 0.25, 0.2, 0.15, 0.15]
earphone_beneficial = [True, True, True, True, False]

class MultiMoora:
    def __init__(self, products, criteria, weights, beneficial, category):
        self.products = products
        self.criteria = criteria
        self.weights = np.array(weights)
        self.beneficial = np.array(beneficial, dtype=bool)
        self.category = category
        self.matrix = self.convert_to_numeric()

    def convert_to_numeric(self):
        if self.category == "phones":
            # Phone-specific mappings
            processor_mapping = {
                "Snapdragon 8 Gen 3": 12,
                "Tensor G4": 11,
                "Exynos 2400": 11,
                "A19 Bionic": 11,
                "Kirin 9010": 10,
                "Dimensity 9200": 10,
            "Snapdragon 8 Gen 2": 9,
                "Tensor G3": 9,
                "Exynos 2300": 9,
                "A18 Bionic": 9,
            "Snapdragon 8 Gen 1": 8,
                "Exynos 2200": 8,
                "A17 Bionic": 8,
                "Snapdragon 7 Gen 3": 7,
                "Snapdragon 7 Gen 2": 7,
                "Exynos 1380": 7,
                "Exynos 1280": 6,
                "MediaTek Helio G99": 5,
                "Snapdragon 775": 5,
            }

            screen_mapping = {
                "Dynamic AMOLED": 10,
                "Super Retina XDR OLED": 10,
                "Fluid AMOLED": 9,
                "Super AMOLED": 9,
            "AMOLED": 8,
                "OLED": 8,
                "P-OLED": 8,
                "PureDisplay V4": 7,
                "IPS LCD": 6,
                "Liquid Retina HD": 6,
            }
        elif self.category == "laptops":
            # Laptop-specific mappings
            processor_mapping = {
                "Apple M3 Pro": 13,
                "Apple M2": 12,
                "AMD Ryzen 9 7945HX": 12,
                "Intel Core i9-13900H": 12,
                "AMD Ryzen 9 6900HS": 11,
                "Intel Core i7-13700H": 11,
                "AMD Ryzen 7 7840HS": 10,
                "Intel Core i7-1365U": 10,
                "Intel Core i7-1355U": 10,
                "AMD Ryzen 7 7730U": 9,
                "Intel Core i7-1280P": 9,
                "Intel Core i7-12700H": 9,
                "Intel Core i7-12650H": 8,
                "Intel Core i7-1255U": 8,
                "Intel Core i7-1260P": 8,
                "Intel Core i5-1240P": 7,
                "Intel Core i5-1235U": 7,
                "Intel Pentium Gold 8505": 5,
            }

            gpu_mapping = {
                "NVIDIA GeForce RTX 4080": 12,
                "NVIDIA GeForce RTX 4070": 11,
                "NVIDIA GeForce RTX 3070 Ti": 10,
                "NVIDIA GeForce RTX 3070": 10,
                "NVIDIA GeForce RTX 3060": 9,
                "NVIDIA GeForce RTX 3050": 8,
                "AMD Radeon RX 6800S": 9,
                "Integrated 16-core GPU": 8,
                "Integrated 8-core GPU": 7,
                "Integrated Intel Iris Xe": 6,
                "Integrated AMD Radeon": 6,
                "Integrated Intel UHD Graphics": 4,
                "Integrated Qualcomm Adreno": 4,
            }
        elif self.category == "smartwatches":
            # Smartwatch-specific mappings
            display_mapping = {
                "Retina": 10,
                "AMOLED": 9,
                "MIP": 8,
                "LCD": 7,
                "LED": 6,
            }

            water_resistance_mapping = {
                "100m": 10,
                "50m": 8,
                "30m": 6,
            }

            gps_mapping = {
                "Yes": 10,
                "No": 0,
            }
        else:  # earphones
            # Earphone-specific mappings
            driver_mapping = {
                "Custom High-Excursion": 12,
                "Custom Apple": 11,
                "Custom Bose": 11,
                "11mm + Planar": 10,
                "11mm + 6mm Dual": 10,
                "10mm Dual Dynamic": 9,
                "13mm Dynamic": 9,
                "12mm Dynamic": 8,
                "11.6mm Dynamic": 8,
                "11mm Dynamic": 8,
                "10mm Dynamic": 7,
                "9.2mm Dynamic": 7,
                "8.4mm Dynamic": 7,
                "7mm Dynamic": 6,
                "6mm Dynamic": 5,
                "5mm Dynamic": 4,
            }

            water_resistance_mapping = {
                "IP57": 10,
                "IPX7": 9,
                "IP55": 8,
                "IP54": 7,
                "IPX5": 6,
                "IPX4": 5,
            }

            noise_cancellation_mapping = {
                "Active": 10,
                "Passive": 5,
            }

        matrix = []
        for specs in self.products.values():
            row = []
            for key, value in specs.items():
                if key == "Processor" and isinstance(value, str) and self.category == "phones":
                    best_match = max(processor_mapping.items(), 
                                  key=lambda x: x[0] in value or value in x[0])
                    row.append(best_match[1])
                elif key == "Screen" and isinstance(value, str) and self.category == "phones":
                    best_match = max(screen_mapping.items(), 
                                  key=lambda x: x[0] in value or value in x[0])
                    row.append(best_match[1])
                elif key == "GPU" and isinstance(value, str) and self.category == "laptops":
                    best_match = max(gpu_mapping.items(), 
                                  key=lambda x: x[0] in value or value in x[0])
                    row.append(best_match[1])
                elif key == "Display" and isinstance(value, str) and self.category == "smartwatches":
                    best_match = max(display_mapping.items(), 
                                  key=lambda x: x[0] in value or value in x[0])
                    row.append(best_match[1])
                elif key == "Water Resistance" and isinstance(value, str):
                    if self.category in ["smartwatches", "earphones"]:
                        best_match = max(water_resistance_mapping.items(), 
                                      key=lambda x: x[0] in value or value in x[0])
                        row.append(best_match[1])
                elif key == "GPS" and isinstance(value, str) and self.category == "smartwatches":
                    row.append(gps_mapping.get(value, 0))
                elif key == "Driver Size" and isinstance(value, str) and self.category == "earphones":
                    best_match = max(driver_mapping.items(), 
                                  key=lambda x: x[0] in value or value in x[0])
                    row.append(best_match[1])
                elif key == "Noise Cancellation" and isinstance(value, str) and self.category == "earphones":
                    row.append(noise_cancellation_mapping.get(value, 0))
                elif key == "RAM" and isinstance(value, str) and self.category == "laptops":
                    # Convert RAM string to numeric value (e.g., "16GB" -> 16)
                    ram_value = int(value.replace("GB", ""))
                    row.append(ram_value)
                elif key == "Battery Life" and isinstance(value, str):
                    # Convert battery life string to numeric value
                    if "days" in value:
                        days = float(value.split()[0])
                        hours = days * 24
                    elif "hours" in value:
                        hours = float(value.split()[0])
                    else:
                        hours = 0
                    row.append(hours)
                elif isinstance(value, (int, float)):
                    row.append(value)
                else:
                    raise ValueError(f"Invalid data type for value: {value}")
            matrix.append(row)
        return np.array(matrix, dtype=float)

    def normalize_matrix(self):
        return self.matrix / np.sqrt((self.matrix ** 2).sum(axis=0))

    def apply_moora(self):
        norm_matrix = self.normalize_matrix()
        weighted_matrix = norm_matrix * self.weights
        beneficial_scores = (weighted_matrix * self.beneficial).sum(axis=1)
        non_beneficial_scores = (weighted_matrix * ~self.beneficial).sum(axis=1)
        moora_scores = beneficial_scores - non_beneficial_scores
        rankings = np.argsort(-moora_scores)  # Higher score = better ranking
        return rankings, moora_scores

    def compare_products(self):
        rankings, scores = self.apply_moora()
        comparison_result = []
        for rank, index in enumerate(rankings):
            product_name = list(self.products.keys())[index]
            comparison_result.append({
                "rank": rank + 1, 
                "name": product_name, 
                "score": round(scores[index], 3),
                "price": self.products[product_name]["Price"]
            })
        return comparison_result

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash("Username already taken. Please choose another one.")
            return redirect(url_for('register'))
        
        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please use another one.")
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful! Please log in.")
        return redirect(url_for('login'))
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password")
    
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/delete_account", methods=["POST"])
@login_required
def delete_account():
    try:
        # First, delete all favorites associated with the user
        Favorite.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        
        # Then delete the user account
        db.session.delete(current_user)
        db.session.commit()
        
        logout_user()
        flash("Your account has been deleted.")
        return redirect(url_for('login'))
    except Exception as e:
        db.session.rollback()
        flash("An error occurred while deleting your account. Please try again.")
        return redirect(url_for('accounts'))

@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare():
    if request.method == "GET":
        return render_template('comparisonpage.html')  # Fixed indentation
    
    try:
        data = request.get_json()
        if not data or 'products' not in data:
            return jsonify({"error": "No products selected for comparison"}), 400
        
        selected_products = data['products']
        if not isinstance(selected_products, list):
            return jsonify({"error": "Invalid product selection format"}), 400
    
        if len(selected_products) != 2:
            return jsonify({"error": "Select exactly two products to compare!"}), 400
        
        # Determine the category of the first product
        first_product = selected_products[0]
        if first_product in phones:
            category = "phones"
            products_dict = phones
            criteria = phone_criteria
            weights = phone_weights
            beneficial = phone_beneficial
        elif first_product in laptops:
            category = "laptops"
            products_dict = laptops
            criteria = laptop_criteria
            weights = laptop_weights
            beneficial = laptop_beneficial
        elif first_product in smartwatches:
            category = "smartwatches"
            products_dict = smartwatches
            criteria = smartwatch_criteria
            weights = smartwatch_weights
            beneficial = smartwatch_beneficial
        elif first_product in earphones:
            category = "earphones"
            products_dict = earphones
            criteria = earphone_criteria
            weights = earphone_weights
            beneficial = earphone_beneficial
        else:
            return jsonify({"error": f"Product not found: {first_product}"}), 400
        
        # Validate that all selected products exist and are from the same category
        missing_products = [name for name in selected_products if name not in products_dict]
        if missing_products:
            return jsonify({"error": f"Products not found: {', '.join(missing_products)}"}), 400
        
        # Check if second product is from a different category
        if (category == "phones" and selected_products[1] in laptops) or \
           (category == "laptops" and selected_products[1] in phones) or \
           (category == "smartwatches" and selected_products[1] in phones) or \
           (category == "smartwatches" and selected_products[1] in laptops) or \
           (category == "phones" and selected_products[1] in smartwatches) or \
           (category == "laptops" and selected_products[1] in smartwatches) or \
           (category == "earphones" and selected_products[1] in phones) or \
           (category == "earphones" and selected_products[1] in laptops) or \
           (category == "earphones" and selected_products[1] in smartwatches) or \
           (category == "phones" and selected_products[1] in earphones) or \
           (category == "laptops" and selected_products[1] in earphones) or \
           (category == "smartwatches" and selected_products[1] in earphones):
            return jsonify({"error": "Cannot compare products from different categories!"}), 400
        
        # Create a dictionary with only the selected products
        selected_data = {name: products_dict[name] for name in selected_products}
        
        # Validate that all required specs are present
        for name, specs in selected_data.items():
            missing_specs = [spec for spec in criteria if spec not in specs]
            if missing_specs:
                return jsonify({"error": f"Missing specifications for {name}: {', '.join(missing_specs)}"}), 400
        
        # Perform comparison
        moora = MultiMoora(selected_data, criteria, weights, beneficial, category)
        comparison_result = moora.compare_products()  # Fixed indentation
    
        return jsonify({"comparison": comparison_result})  # Fixed indentation

    except Exception as e:
        print(f"Comparison error: {str(e)}")  # Log the error
        return jsonify({"error": f"An error occurred during comparison: {str(e)}"}), 500

@app.route("/accounts", methods=["GET", "POST"])
@login_required
def accounts():
    return render_template('accounts.html', user=current_user)

@app.route("/update_profile", methods=["POST"])
@login_required
def update_profile():
    username = request.form.get("username")
    email = request.form.get("email")
    
    # Check if username is already taken by another user
    if username != current_user.username:
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already taken. Please choose another one.")
            return redirect(url_for('accounts'))
    
    # Check if email is already taken by another user
    if email != current_user.email:
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash("Email already registered. Please use another one.")
            return redirect(url_for('accounts'))
    
    # Update user information
    current_user.username = username
    current_user.email = email
    db.session.commit()
    
    flash("Profile updated successfully!")
    return redirect(url_for('accounts'))

@app.route("/update_password", methods=["POST"])
@login_required
def update_password():
    current_password = request.form.get("current_password")
    new_password = request.form.get("new_password")
    confirm_password = request.form.get("confirm_password")
    
    # Verify current password
    if not check_password_hash(current_user.password, current_password):
        flash("Current password is incorrect.")
        return redirect(url_for('accounts'))
    
    # Check if new passwords match
    if new_password != confirm_password:
        flash("New passwords do not match.")
        return redirect(url_for('accounts'))
    
    # Update password
    current_user.password = generate_password_hash(new_password)
    db.session.commit()
    
    flash("Password updated successfully!")
    return redirect(url_for('accounts'))

@app.route('/earphone_images/<path:filename>')
def serve_earphone_image(filename):
    return send_from_directory(app.config['EARPHONE_IMAGES_FOLDER'], filename)

@app.route('/laptop_images/<path:filename>')
def serve_laptop_image(filename):
    return send_from_directory(app.config['LAPTOP_IMAGES_FOLDER'], filename)

@app.route('/smartwatch_images/<path:filename>')
def serve_smartwatch_image(filename):
    return send_from_directory(app.config['SMARTWATCH_IMAGES_FOLDER'], filename)

@app.route('/phone_images/<path:filename>')
def serve_phone_image(filename):
    return send_from_directory(app.config['PHONE_IMAGES_FOLDER'], filename)

def clean_filename(filename):
    # Remove invalid characters from filename
    return re.sub(r'[<>:"/\\|?*]', '', filename)

@app.route("/get_products")
def get_products():
    try:
        # Get all products
        all_products = []
        
        # Process phones
        for name, specs in phones.items():
            # Check for image in mobiles directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_found = False
            image_url = None
            
            for ext in image_extensions:
                # Try exact match first
                if os.path.exists(os.path.join(app.config['PHONE_IMAGES_FOLDER'], f"{name}{ext}")):
                    image_url = f"/phone_images/{name}{ext}"
                    image_found = True
                    break
                
                # Try case-insensitive match
                for filename in os.listdir(app.config['PHONE_IMAGES_FOLDER']):
                    if filename.lower() == f"{name.lower()}{ext}":
                        image_url = f"/phone_images/{filename}"
                        image_found = True
                        break
                if image_found:
                    break
            
            if not image_found:
                image_url = "/static/placeholder.jpg"
            
            all_products.append({
                "name": name,
                "category": "Smartphones",
                "specs": specs,
                "imageUrl": image_url
            })
        
        # Process laptops
        for name, specs in laptops.items():
            # Check for image in laptops directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_found = False
            image_url = None
            
            for ext in image_extensions:
                # Try exact match first
                if os.path.exists(os.path.join(app.config['LAPTOP_IMAGES_FOLDER'], f"{name}{ext}")):
                    image_url = f"/laptop_images/{name}{ext}"
                    image_found = True
                    break
                
                # Try case-insensitive match
                for filename in os.listdir(app.config['LAPTOP_IMAGES_FOLDER']):
                    if filename.lower() == f"{name.lower()}{ext}":
                        image_url = f"/laptop_images/{filename}"
                        image_found = True
                        break
                if image_found:
                    break
            
            if not image_found:
                image_url = "/static/placeholder.jpg"
            
            all_products.append({
                "name": name,
                "category": "Laptops",
                "specs": specs,
                "imageUrl": image_url
            })
        
        # Process smartwatches
        for name, specs in smartwatches.items():
            # Check for image in smartwatches directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_found = False
            image_url = None
            
            for ext in image_extensions:
                # Try exact match first
                if os.path.exists(os.path.join(app.config['SMARTWATCH_IMAGES_FOLDER'], f"{name}{ext}")):
                    image_url = f"/smartwatch_images/{name}{ext}"
                    image_found = True
                    break
                
                # Try case-insensitive match
                for filename in os.listdir(app.config['SMARTWATCH_IMAGES_FOLDER']):
                    if filename.lower() == f"{name.lower()}{ext}":
                        image_url = f"/smartwatch_images/{filename}"
                        image_found = True
                        break
                if image_found:
                    break
            
            if not image_found:
                image_url = "/static/placeholder.jpg"
            
            all_products.append({
                "name": name,
                "category": "Smartwatches",
                "specs": specs,
                "imageUrl": image_url
            })
        
        # Process earphones
        for name, specs in earphones.items():
            # Check for image in earphone images directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            
            # Clean the product name for image search
            clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name).strip()
            
            # Try different variations of the name
            name_variations = [
                clean_name,
                clean_name.replace(' ', ''),
                clean_name.replace(' ', '_'),
                clean_name.replace(' ', '-'),
                name.replace(' ', ''),
                name.replace(' ', '_'),
                name.replace('', '-'),
                name,  # Try the original name as is
                name.replace('(', '').replace(')', '').strip(),  # Remove parentheses
                name.replace('(2nd Gen)', '').strip(),  # Remove generation info
                name.replace('(2nd)', '').strip(),  # Remove generation info
                name.replace('2nd Gen', '').strip(),  # Remove generation info
                name.replace('2nd', '').strip()  # Remove generation info
            ]
            
            image_found = False
            image_url = None
            
            for ext in image_extensions:
                for variation in name_variations:
                    # Try exact match
                    if os.path.exists(os.path.join(app.config['EARPHONE_IMAGES_FOLDER'], f"{variation}{ext}")):
                        image_url = f"/earphone_images/{variation}{ext}"
                        image_found = True
                        break
                    
                    # Try case-insensitive match
                    for filename in os.listdir(app.config['EARPHONE_IMAGES_FOLDER']):
                        if filename.lower() == f"{variation.lower()}{ext}":
                            image_url = f"/earphone_images/{filename}"
                            image_found = True
                            break
                    if image_found:
                        break
                if image_found:
                    break
            
            # Skip products without images
            if not image_found:
                print(f"Warning: No image found for {name}, skipping product")
                continue
            
            # Add product only once with validated image
            all_products.append({
                "name": name,
                "category": "Earphones",
                "specs": specs,
                "imageUrl": image_url
            })
        
        return jsonify(all_products)
    except Exception as e:
        print(f"Error in get_products: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/compare_products", methods=["POST"])
@login_required
def compare_products():
    try:
        data = request.get_json()
        if not data or 'products' not in data:
            return jsonify({"error": "No products selected for comparison"}), 400
        
        selected_products = data['products']
        if not isinstance(selected_products, list):
            return jsonify({"error": "Invalid product selection format"}), 400
        
        if len(selected_products) != 2:
            return jsonify({"error": "Select exactly two products to compare!"}), 400
        
        # Determine the category of the first product
        first_product = selected_products[0]
        if first_product in phones:
            category = "phones"
            products_dict = phones
            criteria = phone_criteria
            weights = phone_weights
            beneficial = phone_beneficial
        elif first_product in laptops:
            category = "laptops"
            products_dict = laptops
            criteria = laptop_criteria
            weights = laptop_weights
            beneficial = laptop_beneficial
        elif first_product in smartwatches:
            category = "smartwatches"
            products_dict = smartwatches
            criteria = smartwatch_criteria
            weights = smartwatch_weights
            beneficial = smartwatch_beneficial
        elif first_product in earphones:
            category = "earphones"
            products_dict = earphones
            criteria = earphone_criteria
            weights = earphone_weights
            beneficial = earphone_beneficial
        else:
            return jsonify({"error": f"Product not found: {first_product}"}), 400
        
        # Validate that all selected products exist and are from the same category
        missing_products = [name for name in selected_products if name not in products_dict]
        if missing_products:
            return jsonify({"error": f"Products not found: {', '.join(missing_products)}"}), 400
        
        # Check if second product is from a different category
        if (category == "phones" and selected_products[1] in laptops) or \
           (category == "laptops" and selected_products[1] in phones) or \
           (category == "smartwatches" and selected_products[1] in phones) or \
           (category == "smartwatches" and selected_products[1] in laptops) or \
           (category == "phones" and selected_products[1] in smartwatches) or \
           (category == "laptops" and selected_products[1] in smartwatches) or \
           (category == "earphones" and selected_products[1] in phones) or \
           (category == "earphones" and selected_products[1] in laptops) or \
           (category == "earphones" and selected_products[1] in smartwatches) or \
           (category == "phones" and selected_products[1] in earphones) or \
           (category == "laptops" and selected_products[1] in earphones) or \
           (category == "smartwatches" and selected_products[1] in earphones):
            return jsonify({"error": "Cannot compare products from different categories!"}), 400
        
        # Create a dictionary with only the selected products
        selected_data = {name: products_dict[name] for name in selected_products}
        
        # Validate that all required specs are present
        for name, specs in selected_data.items():
            missing_specs = [spec for spec in criteria if spec not in specs]
            if missing_specs:
                return jsonify({"error": f"Missing specifications for {name}: {', '.join(missing_specs)}"}), 400
        
        # Perform comparison
        moora = MultiMoora(selected_data, criteria, weights, beneficial, category)
        comparison_result = moora.compare_products()
        
        return jsonify({"comparison": comparison_result})

    except Exception as e:
        print(f"Comparison error: {str(e)}")  # Log the error
        return jsonify({"error": f"An error occurred during comparison: {str(e)}"}), 500

@app.route("/laptops")
def laptop_listing():
    return render_template('laptop_listing.html')

@app.route("/smartwatches")
def smartwatch_listing():
    return render_template('smartwatch_listing.html')

@app.route("/phones")
def phone_listing():
    return render_template('phone_listing.html')

@app.route("/earphones")
def earphone_listing():
    return render_template('earphone_listing.html')

@app.route("/earphone_detail")
def earphone_detail():
    return render_template('earphone_detail.html')

@app.route("/laptop_detail")
def laptop_detail():
    return render_template('laptop_detail.html')

@app.route("/smartwatch_detail")
def smartwatch_detail():
    return render_template('smartwatch_detail.html')

@app.route("/phone_detail")
def phone_detail():
    product_name = request.args.get('name')
    if not product_name:
        return redirect(url_for('phone_listing'))
    return render_template("phone_detail.html", product_name=product_name)

@app.route("/generate_summary", methods=["POST"])
def generate_summary():
    try:
        data = request.get_json()
        specs = data.get('specs', {})
        name = data.get('name', '')
        category = data.get('category', '')
        
        if not specs or not name or not category:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Load reviews based on category
        reviews = []
        if category == "Earphones":
            with open('reviews_earphones.json', 'r', encoding='utf-8') as f:
                reviews_data = json.load(f)
                reviews = reviews_data.get(name, [])
        elif category == "Smartphones":
            with open('reviews_phones.json', 'r', encoding='utf-8') as f:
                reviews_data = json.load(f)
                reviews = reviews_data.get(name, [])
        elif category == "Laptops":
            with open('reviews_laptops.json', 'r', encoding='utf-8') as f:
                reviews_data = json.load(f)
                reviews = reviews_data.get(name, [])
        elif category == "Smartwatches":
            with open('reviews_smartwatches.json', 'r', encoding='utf-8') as f:
                reviews_data = json.load(f)
                reviews = reviews_data.get(name, [])
        
        # Generate feature-based summary
        feature_summary = ""
        if category == "Earphones":
            feature_summary = f"These {specs.get('Driver Size', '')} earphones offer {specs.get('Battery Life', '')} hours of battery life. "
            feature_summary += f"With {specs.get('Noise Cancellation', '')} noise cancellation and {specs.get('Water Resistance', '')} water resistance, "
            feature_summary += f"they are priced at ₹{specs.get('Price', {}).get('amazon', 0):,}."
        elif category == "Smartphones":
            feature_summary = f"This {specs.get('RAM', '')} RAM smartphone features a {specs.get('Display', '')} display. "
            feature_summary += f"Powered by {specs.get('Processor', '')} processor, it offers {specs.get('Battery', '')} battery capacity. "
            feature_summary += f"The {specs.get('Camera', '')} camera setup ensures high-quality photos and videos."
        elif category == "Laptops":
            feature_summary = f"This {specs.get('Processor', '')} laptop comes with {specs.get('RAM', '')} RAM and {specs.get('Storage', '')} storage. "
            feature_summary += f"The {specs.get('Display', '')} display provides crisp visuals, while the {specs.get('Battery', '')} battery ensures long-lasting performance."
        elif category == "Smartwatches":
            feature_summary = f"This smartwatch features a {specs.get('Display', '')} display and offers {specs.get('Battery Life', '')} battery life. "
            feature_summary += f"With {specs.get('Water Resistance', '')} water resistance and {specs.get('Health Features', '')} health tracking capabilities, "
            feature_summary += f"it's designed for both style and functionality."

        # Generate review-based summary
        review_summary = ""
        if reviews:
            # Analyze sentiment of reviews
            sentiments = []
            for review in reviews:
                sentiment_inputs = sentiment_tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
                sentiment_outputs = sentiment_model(**sentiment_inputs)
                sentiment_score = torch.argmax(sentiment_outputs.logits).item()
                sentiments.append(sentiment_score)
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_label = "positive" if avg_sentiment > 2 else "negative" if avg_sentiment < 2 else "neutral"
            
            # Prepare review text for BART
            reviews_text = " ".join(reviews)
            review_input = tokenizer(
                reviews_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Generate review summary with sentiment context
            review_summary_ids = model.generate(
                review_input["input_ids"],
                max_length=100,
                min_length=30,
                num_beams=5,
                length_penalty=1.2,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
                no_repeat_ngram_size=3
            )
            review_summary = tokenizer.decode(review_summary_ids[0], skip_special_tokens=True)
            
            # Clean up the review summary
            review_summary = re.sub(r'\b(I|you|your|my|me|we|our|us)\b', '', review_summary, flags=re.IGNORECASE)
            review_summary = re.sub(r'\s+', ' ', review_summary).strip()
            
            # Remove any prompt text that might have been included
            review_summary = re.sub(r'Summarize these reviews.*?Keep it brief and natural:', '', review_summary, flags=re.IGNORECASE)
            review_summary = re.sub(r'Based on user reviews.*?sentiment is \w+\.', '', review_summary, flags=re.IGNORECASE)
            
            # Ensure proper punctuation
            review_summary = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', review_summary)  # Add line breaks after sentences
            review_summary = re.sub(r'([.!?])\s*$', r'\1', review_summary)  # Ensure ending punctuation
            
            # Add sentiment context to the summary
            review_summary = f"Based on user reviews, the overall sentiment is {sentiment_label}. {review_summary}"
        
        # Clean up summaries
        feature_summary = " ".join(feature_summary.split())
        if not feature_summary.endswith('.'):
            feature_summary += '.'
        
        review_summary = " ".join(review_summary.split())
        if not review_summary.endswith('.'):
            review_summary += '.'
        
        return jsonify({
            "feature_summary": feature_summary,
            "review_summary": review_summary if review_summary else "No reviews available for this product."
        })
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return jsonify({"error": f"An error occurred while generating summaries: {str(e)}"}), 500

@app.route("/favorites")
@login_required
def favorites():
    favorites = Favorite.query.filter_by(user_id=current_user.id).all()
    return render_template('favorites.html', favorites=favorites)

@app.route("/add_to_favorites", methods=["POST"])
@login_required
def add_to_favorites():
    try:
        data = request.get_json()
        product_name = data.get('name')
        product_category = data.get('category')
        product_image = data.get('imageUrl')
        specs = data.get('specs', {})
        product_price = float(specs.get('Price', 0))

        # Check if product is already in favorites
        existing_favorite = Favorite.query.filter_by(
            user_id=current_user.id,
            product_name=product_name,
            product_category=product_category
        ).first()

        if existing_favorite:
            return jsonify({"message": "Product already in favorites"}), 200

        # Add new favorite
        new_favorite = Favorite(
            user_id=current_user.id,
            product_name=product_name,
            product_category=product_category,
            product_image=product_image,
            product_price=product_price
        )
        db.session.add(new_favorite)
        db.session.commit()

        return jsonify({"message": "Product added to favorites"}), 200
    except Exception as e:
        print(f"Error adding to favorites: {str(e)}")  # Add logging
        db.session.rollback()  # Rollback on error
        return jsonify({"error": str(e)}), 500

@app.route("/remove_from_favorites", methods=["POST"])
@login_required
def remove_from_favorites():
    try:
        data = request.get_json()
        product_name = data.get('name')
        product_category = data.get('category')

        favorite = Favorite.query.filter_by(
            user_id=current_user.id,
            product_name=product_name,
            product_category=product_category
        ).first()

        if favorite:
            db.session.delete(favorite)
            db.session.commit()
            return jsonify({"message": "Product removed from favorites"}), 200
        else:
            return jsonify({"message": "Product not found in favorites"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/check_favorite", methods=["POST"])
@login_required
def check_favorite():
    try:
        data = request.get_json()
        product_name = data.get('name')
        product_category = data.get('category')

        favorite = Favorite.query.filter_by(
            user_id=current_user.id,
            product_name=product_name,
            product_category=product_category
        ).first()

        return jsonify({"is_favorite": favorite is not None}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Create the database and add a test user
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='adish').first():
        test_user = User(
            username='adish',
            email='adishrafique@gmail.com',
            password=generate_password_hash('adishadish')
        )
        db.session.add(test_user)
        db.session.commit()

if __name__ == "__main__":
    app.run(debug=True)
