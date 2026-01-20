import mysql.connector
from mysql.connector import Error

class Database:
    def __init__(self):
        try:
            self.conn = mysql.connector.connect(
                host='localhost',
                user='root',  # Change these credentials
                password='root',  # according to your MySQL setup
                database='rag_system'
            )
            self.cursor = self.conn.cursor(dictionary=True)
            self.create_tables()
        except Error as e:
            print(f"Error connecting to MySQL Database: {e}")
    
    def create_tables(self):
        # Create users table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                role ENUM('student', 'teacher') NOT NULL
            )
        ''')
        
        # Create notes table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                filepath VARCHAR(255) NOT NULL,
                uploaded_by INT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (uploaded_by) REFERENCES users(id)
            )
        ''')
        
        self.conn.commit()
    
    def create_user(self, username, password, role):
        try:
            self.cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                (username, password, role)
            )
            self.conn.commit()
            return True
        except Error as e:
            print(f"Error creating user: {e}")
            return False
    
    def get_user_by_username(self, username):
        try:
            self.cursor.execute(
                "SELECT * FROM users WHERE username = %s",
                (username,)
            )
            return self.cursor.fetchone()
        except Error as e:
            print(f"Error fetching user: {e}")
            return None
    
    def add_note(self, title, filepath, uploaded_by):
        try:
            self.cursor.execute(
                "INSERT INTO notes (title, filepath, uploaded_by) VALUES (%s, %s, %s)",
                (title, filepath, uploaded_by)
            )
            self.conn.commit()
            return True
        except Error as e:
            print(f"Error adding note: {e}")
            return False
    
    def get_notes(self):
        try:
            self.cursor.execute('''
                SELECT notes.*, users.username as uploaded_by_username 
                FROM notes 
                JOIN users ON notes.uploaded_by = users.id 
                ORDER BY upload_date DESC
            ''')
            return self.cursor.fetchall()
        except Error as e:
            print(f"Error fetching notes: {e}")
            return []
    
    def get_note_by_id(self, note_id):
        try:
            self.cursor.execute(
                "SELECT * FROM notes WHERE id = %s",
                (note_id,)
            )
            return self.cursor.fetchone()
        except Error as e:
            print(f"Error fetching note: {e}")
            return None
    
    def delete_note(self, note_id):
        try:
            self.cursor.execute(
                "DELETE FROM notes WHERE id = %s",
                (note_id,)
            )
            self.conn.commit()
            return True
        except Error as e:
            print(f"Error deleting note: {e}")
            return False
    
    def __del__(self):
        if hasattr(self, 'conn') and self.conn.is_connected():
            self.cursor.close()
            self.conn.close() 