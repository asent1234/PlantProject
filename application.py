from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import uuid
from dotenv import load_dotenv
import threading
import time

# Load environment variables from .env file
load_dotenv()

# Critical imports needed at startup
from datetime import datetime
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy  # Required for database setup
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

# Heavy ML-related imports are still deferred
# import cv2
# import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('SQLALCHEMY_DATABASE_URI', 'sqlite:///database.db')  # Use RDS or other DB in production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    # the content we want to get from users
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    history = db.relationship('History', backref='user', lazy=True)


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    disease = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    notes = db.Column(db.Text, nullable=True)


class SignUpForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_unique_filename(filename):
    """Generate a unique filename to prevent overwriting"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    new_filename = f"{uuid.uuid4().hex}.{ext}"
    return new_filename


# Placeholder for the plant disease detection model
# This would be replaced with your actual model
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os

class PlantDiseaseModel:
    def __init__(self):
        # Model and class setup
        model_path = os.path.join(
            os.path.dirname(__file__),
            "ConvNextAIPredictionModel",
            "plant_disease_convnext_best.pth"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Provided classes
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
            'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        from torchvision import models
        import torch.nn as nn
        self.model = models.convnext_tiny(weights=None)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Mapping for user-friendly names
        self.pretty_map = self._make_pretty_map(self.class_names)

    def _make_pretty_map(self, class_names):
        pretty = {}
        for raw in class_names:
            if '___' in raw:
                plant, disease = raw.split('___', 1)
                disease = disease.replace('_', ' ')
                if disease == 'healthy':
                    pretty[raw] = f"{plant.replace('_', ' ')} Healthy"
                elif plant == 'Background_without_leaves':
                    pretty[raw] = "No Plant Detected"
                else:
                    pretty[raw] = f"{plant.replace('_', ' ')} - {disease}"
            else:
                pretty[raw] = raw.replace('_', ' ')
        return pretty

    def predict(self, image_path):
        try:
            # Plant presence check
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                return {"error": "Could not read image file."}
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = cv2.countNonZero(mask) / (img_cv.shape[0] * img_cv.shape[1])
            if green_ratio < 0.05:
                return {"error": "No plants detected in the image"}
            # Model prediction
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type):
                    output = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            top_prob, top_idx = torch.max(probabilities, 0)
            pred_class = self.class_names[top_idx.item()]
            pretty = self.pretty_map[pred_class]
            # Handle background class as error
            if pred_class == 'Background_without_leaves':
                return {"error": "No plants detected in the image"}
            return {
                "disease": pretty,
                "confidence": float(top_prob.item()),
                "error": None
            }
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}


# Initialize models lazily to reduce startup time and memory usage
import gc

plant_model = None
chatbot_engine = None

def get_plant_model():
    global plant_model
    if plant_model is None:
        # Force garbage collection before loading model
        gc.collect()
        # Only import the model class when needed
        import torch
        torch.set_num_threads(1)  # Further restrict threads
        plant_model = PlantDiseaseModel()
    return plant_model

def get_chatbot_engine():
    global chatbot_engine
    if chatbot_engine is None:
        # Force garbage collection before loading chatbot
        gc.collect()
        # Only import the chatbot engine when needed
        from PlantRagChatbot.query_plant_data import PlantQueryEngine
        chatbot_engine = PlantQueryEngine()
    return chatbot_engine

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get('message')
    context = data.get('context', '')
    # Compose prompt with context
    prompt = f"Context:\n{context}\n---\nUser question: {user_message}"
    try:
        # Lazy load the chatbot engine only when needed
        response = get_chatbot_engine().get_deepseek_completion(prompt, stream=False)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'[Chatbot error] {str(e)}'}), 500

# Global flag to indicate if we're in startup mode
APP_INITIALIZING = True

# Start a background thread to initialize heavy components
def initialize_background():
    global APP_INITIALIZING
    
    # Force a delay to let the health checks pass
    time.sleep(3)
    
    # Start importing heavy libraries
    try:
        print("Background initialization starting...")
        # Import other heavy dependencies
        import cv2
        import numpy as np
        from plant_general_recommendations import PLANT_GENERAL_RECOMMENDATIONS
        
        # Preload ML models to memory
        if 'get_plant_model' in globals():
            print("Preloading plant model...")
            get_plant_model()
            
        if 'get_chatbot_engine' in globals():
            print("Preloading chatbot engine...")
            get_chatbot_engine()
        
        # Once everything is imported, mark app as initialized
        APP_INITIALIZING = False
        print("Background initialization complete!")
    except Exception as e:
        print(f"Error in background initialization: {e}")

@app.route('/health')
def health_check():
    """A lightweight health check endpoint for AWS Lightsail"""
    return jsonify({
        'status': 'healthy',
        'message': 'Plant Disease Detection API is running'
    })


@app.route('/')
def home():
    return render_template('home.html')


# Health check endpoint already defined above


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignUpForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('signup.html', form=form)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate a secure unique filename
            filename = secure_filename(file.filename)
            unique_filename = generate_unique_filename(filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Process the image with the model
            return redirect(url_for('process_image', filename=unique_filename))
    
    return render_template('upload.html')


@app.route('/camera')
def camera():
    """Route for capturing images directly from camera"""
    return render_template('camera.html')


@app.route('/save-camera-image', methods=['POST'])
def save_camera_image():
    """API endpoint to save images captured from camera"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image data'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a secure unique filename
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        return jsonify({'success': True, 'filename': unique_filename})
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/process-image/<filename>')
def process_image(filename):
    """Process the uploaded image and detect plant disease"""
    from plant_general_recommendations import PLANT_GENERAL_RECOMMENDATIONS
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Run the model prediction (lazy load the model first time it's needed)
    result = get_plant_model().predict(file_path)
    
    if result.get('error'):
        # Handle error case
        return render_template('error.html', error=result['error'], filename=filename)
    
    # Save to history if user is logged in
    if current_user.is_authenticated:
        history_entry = History(
            user_id=current_user.id,
            filename=filename,
            disease=result['disease'],
            confidence=result['confidence']
        )
        db.session.add(history_entry)
        db.session.commit()
    
    # Find the raw class name from the pretty name
    raw_class = None
    for k, v in get_plant_model().pretty_map.items():
        if v == result['disease']:
            raw_class = k
            break
    rec = PLANT_GENERAL_RECOMMENDATIONS.get(raw_class)
    
    return render_template(
        'result.html',
        filename=filename,
        disease=result['disease'],
        confidence=result['confidence'],
        recommendation=rec
    )


@app.route('/history')
@login_required
def view_history():
    """View user's upload history"""
    history = History.query.filter_by(user_id=current_user.id).order_by(History.upload_date.desc()).all()
    return render_template('history.html', history=history)


@app.route('/history/delete/<int:history_id>', methods=['POST'])
@login_required
def delete_history(history_id):
    """Delete a history entry"""
    history_item = History.query.get_or_404(history_id)
    
    # Check if the history item belongs to the current user
    if history_item.user_id != current_user.id:
        flash('You do not have permission to delete this item')
        return redirect(url_for('view_history'))
    
    db.session.delete(history_item)
    db.session.commit()
    flash('History item deleted successfully')
    return redirect(url_for('view_history'))


# Define 'application' at module level for WSGI/Gunicorn
application = app  # For WSGI compatibility with production servers

# Start background initialization in a separate thread when the application is imported
background_thread = threading.Thread(target=initialize_background)
background_thread.daemon = True  # Thread will exit when main process exits
background_thread.start()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(host='0.0.0.0', port=8080, debug=False)  # Use port 8080 for deployment compatibility
