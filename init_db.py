import os
import psycopg2
from werkzeug.security import generate_password_hash

# Get the database URL from the environment variable set on Render
DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    print("Error: DATABASE_URL is not set. This script should be run on Render.")
    exit()

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

print("Creating tables for PostgreSQL...")

# Create tables with PostgreSQL-compatible syntax (e.g., SERIAL PRIMARY KEY)
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'teacher'
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS classes (
    id SERIAL PRIMARY KEY,
    class_name TEXT UNIQUE NOT NULL
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS students (
    id SERIAL PRIMARY KEY,
    roll_number TEXT NOT NULL,
    name TEXT NOT NULL,
    class_name TEXT NOT NULL DEFAULT 'General',
    image_path TEXT
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
    student_roll_number TEXT NOT NULL,
    name TEXT NOT NULL,
    class_name TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS teacher_classes (
    id SERIAL PRIMARY KEY,
    teacher_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    FOREIGN KEY (teacher_id) REFERENCES users(id)
);
""")

# Insert the default admin user using %s placeholders for psycopg2
print("Checking for default admin user...")
cur.execute("SELECT id FROM users WHERE username = %s;", ('admin',))
if cur.fetchone() is None:
    print("Inserting default admin user...")
    cur.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s);",
        ('admin', generate_password_hash('admin123'), 'admin')
    )

conn.commit()
cur.close()
conn.close()

print("Database initialized successfully.")
