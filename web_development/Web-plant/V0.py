from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO, emit
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plantcare.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    confirmed = db.Column(db.Boolean, default=False)

class Plant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    image_path = db.Column(db.String(200))
    plant_type = db.Column(db.String(100))
    disease = db.Column(db.String(100))
    care_instructions = db.Column(db.Text)

class ChatLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))

        new_user = User(email=email, password=generate_password_hash(password, method='sha256'))
        db.session.add(new_user)
        db.session.commit()

        # Here you would typically send a confirmation email
        flash('Registered successfully. Please check your email to confirm your account.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))

        if not user.confirmed:
            flash('Please confirm your email before logging in.')
            return redirect(url_for('login'))

        login_user(user)
        return redirect(url_for('dashboard'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    plants = Plant.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', plants=plants)

@app.route('/upload', methods=['POST'])
@login_required
def upload_plant():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('dashboard'))
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Here you would typically call your AI service to identify the plant and disease
        plant_type = "Sample Plant"  # Replace with AI identification
        disease = "Sample Disease"  # Replace with AI identification
        care_instructions = "Sample care instructions"  # Replace with AI generated instructions

        new_plant = Plant(user_id=current_user.id, image_path=filename, plant_type=plant_type, 
                          disease=disease, care_instructions=care_instructions)
        db.session.add(new_plant)
        db.session.commit()

        flash('Plant uploaded and analyzed successfully')
        return redirect(url_for('dashboard'))

@socketio.on('send_message')
def handle_message(message):
    chat_log = ChatLog(user_id=current_user.id, message=message)
    db.session.add(chat_log)
    db.session.commit()
    emit('receive_message', {'message': message, 'user': current_user.email}, broadcast=True)

if __name__ == '__main__':
    with app.app_context():  # Ensures we have an active Flask application context
        db.create_all()
    socketio.run(app, debug=True)
