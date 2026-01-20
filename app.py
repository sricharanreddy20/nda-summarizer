from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import mysql.connector
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import chromadb
from document_processor import DocumentProcessor
from database import Database
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'csv', 'xlsx', 'xls'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize database
db = Database()
doc_processor = DocumentProcessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        if db.get_user_by_username(username):
            flash('Username already exists!')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        if db.create_user(username, hashed_password, role):
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        else:
            flash('Registration failed. Please try again.')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = db.get_user_by_username(username)
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = username
            session['role'] = user['role']
            return redirect(url_for('notes'))
        
        flash('Invalid username or password!')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return json.dumps({'error': 'No file uploaded'})
    
    file = request.files['document']
    if file.filename == '':
        return json.dumps({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the document and store in ChromaDB
        try:
            doc_processor.process_document(filepath, session.get('user_id', 'anonymous'))
            return json.dumps({'success': 'Document uploaded and processed successfully'})
        except Exception as e:
            return json.dumps({'error': f'Error processing document: {str(e)}'})
    
    return json.dumps({'error': 'Invalid file type'})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    if not question:
        return json.dumps({'error': 'No question provided'})
    
    # Get answer using RAG
    answer = doc_processor.get_answer(question, session.get('user_id', 'anonymous'))
    
    # Format the answer for the web interface - ensure it's displayed as paragraphs
    formatted_answer = answer.replace('\n', '<br>')
    
    return json.dumps({'answer': formatted_answer})

@app.route('/notes')
def notes():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    notes_list = db.get_notes()
    return render_template('notes.html', notes=notes_list)

@app.route('/upload_note', methods=['POST'])
def upload_note():
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    if 'note' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('notes'))
    
    file = request.files['note']
    title = request.form.get('title', '')
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('notes'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        db.add_note(title, filepath, session['user_id'])
        flash('Note uploaded successfully')
    else:
        flash('Invalid file type')
    
    return redirect(url_for('notes'))

@app.route('/delete_note/<int:note_id>')
def delete_note(note_id):
    if 'user_id' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))
    
    db.delete_note(note_id)
    flash('Note deleted successfully')
    return redirect(url_for('notes'))

@app.route('/download_note/<int:note_id>')
def download_note(note_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    note = db.get_note_by_id(note_id)
    if note:
        return send_file(note['filepath'], as_attachment=True)
    
    flash('Note not found')
    return redirect(url_for('notes'))

if __name__ == '__main__':
    app.run(debug=True) 