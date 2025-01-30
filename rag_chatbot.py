# Import necessary libraries
import os
import json
import time
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import mysql.connector
from datetime import datetime
from embed_store import retrieve_chunks

# Initialize Flask app
app = Flask(__name__)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Example embedding model

# Vector database setup
vector_dim = 384  # Dimension of the embeddings (for 'all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(vector_dim)

# In-memory store for mapping document chunks to their source
chunk_mapping = {}

# MySQL Database setup
DB_CONFIG = {
    "host": "localhost",  # Update if using a different host
    "port": 3306,         # Update if using a different port
    "user": "root",
    "password": "Aksh@8590",
    "database": "chatbot_db"
}

# Connect to MySQL database
def connect_db():
    return mysql.connector.connect(**DB_CONFIG)

# Create table for chat history
def setup_database():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            role VARCHAR(10),
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

setup_database()

# Chat endpoint
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "Invalid JSON format. Expected {'query': 'your message'}"}), 400

    query = data['query']

    # Retrieve top chunks
    top_chunks = retrieve_chunks(query)
    answer = "\n".join(top_chunks)

    # Store chat history in database
    conn = connect_db()
    cursor = conn.cursor()
    timestamp = datetime.now()
    cursor.execute('INSERT INTO chat_history (timestamp, role, content) VALUES (%s, %s, %s)', (timestamp, 'user', query))
    cursor.execute('INSERT INTO chat_history (timestamp, role, content) VALUES (%s, %s, %s)', (timestamp, 'system', answer))
    conn.commit()
    conn.close()

    return jsonify({'query': query, 'answer': answer, 'top_chunks': top_chunks})

# History endpoint
@app.route('/history', methods=['GET'])
def history():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM chat_history ORDER BY timestamp ASC')
    history = cursor.fetchall()
    conn.close()
    return jsonify(history)

# Main function to run Flask app
if __name__ == '__main__':
    app.run(debug=True)