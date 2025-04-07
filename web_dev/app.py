from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'web_dev/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
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


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

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
class PlantDiseaseModel:
    def __init__(self):
        # Initialize your model here
        # For example: self.model = load_model('path/to/model')
        pass
    
    def predict(self, image_path):
        """
        Placeholder for model prediction
        In a real implementation, this would use your trained model
        """
        # Simulate model prediction
        # In reality, you would:
        # 1. Load the image
        # 2. Preprocess it for your model
        # 3. Run inference
        # 4. Return the results
        
        # For demonstration, we'll return a mock result
        import random
        diseases = [
            "Tomato Early Blight", 
            "Apple Black Rot", 
            "Corn Common Rust", 
            "Potato Late Blight",
            "Healthy Plant"
        ]
        
        # Check if image contains plants (placeholder logic)
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Invalid image file"}
            
            # Simple check for green content (very basic plant detection)
            # In a real app, you'd use a more sophisticated method
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
            
            if green_ratio < 0.05:
                return {"error": "No plants detected in the image"}
            
            # Mock prediction
            disease = random.choice(diseases)
            confidence = random.uniform(0.7, 0.99)
            
            return {
                "disease": disease,
                "confidence": confidence,
                "error": None
            }
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}


# Initialize the model
plant_model = PlantDiseaseModel()


@app.route('/')
def home():
    return render_template('home.html')


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


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


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
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Run the model prediction
    result = plant_model.predict(file_path)
    
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
    
    return render_template('result.html', 
                          filename=filename, 
                          disease=result['disease'], 
                          confidence=result['confidence'])


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


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True)