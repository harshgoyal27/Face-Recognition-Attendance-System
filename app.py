import os
import io
import csv
import time
import math
import sqlite3
import tempfile
import base64
import zipfile
import re
from datetime import datetime, date, timedelta
from collections import defaultdict

import cv2
import numpy as np
from deepface import DeepFace

# AI / ML Imports
from sklearn.linear_model import LinearRegression

from flask import (
    Flask, render_template, Response,
    request, redirect, url_for, flash, send_file, jsonify
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, UserMixin, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key_change_me")
UPLOAD_DIR = os.path.join("static", "students")
LEAVE_DOCS_DIR = os.path.join("static", "leave_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LEAVE_DOCS_DIR, exist_ok=True)
login_manager = LoginManager(app)
login_manager.login_view = "login"
DB_PATH = "attendance.db"

# --- Configuration ---
CLASSROOM_LAT = 28.655983
CLASSROOM_LON = 77.291283
MAX_DISTANCE_KM = 0.2
TOTAL_SEMESTER_CLASSES = 40

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- Database Connection ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_user_classes(user_id, role):
    conn = get_db_connection()
    cur = conn.cursor()
    if role == "admin":
        cur.execute("SELECT class_name FROM classes ORDER BY class_name;")
    else:
        cur.execute("SELECT class_name FROM teacher_classes WHERE teacher_id = ? ORDER BY class_name;", (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def setup_database():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'teacher');""")
    cur.execute("""CREATE TABLE IF NOT EXISTS classes (id INTEGER PRIMARY KEY AUTOINCREMENT, class_name TEXT UNIQUE NOT NULL, branch TEXT, year TEXT);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY AUTOINCREMENT, roll_number TEXT NOT NULL UNIQUE, name TEXT NOT NULL, branch TEXT NOT NULL DEFAULT 'General', image_path TEXT, class_name TEXT NOT NULL DEFAULT 'General', password_hash TEXT, year TEXT DEFAULT '1st Year');""")
    cur.execute("""CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, student_roll_number TEXT NOT NULL, name TEXT NOT NULL, class_name TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS teacher_classes (id INTEGER PRIMARY KEY AUTOINCREMENT, teacher_id INTEGER NOT NULL, class_name TEXT NOT NULL, FOREIGN KEY (teacher_id) REFERENCES users(id));""")
    cur.execute("""CREATE TABLE IF NOT EXISTS timetable (id INTEGER PRIMARY KEY AUTOINCREMENT, class_id INTEGER NOT NULL, day_of_week TEXT NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL, FOREIGN KEY (class_id) REFERENCES classes(id));""")
    cur.execute("""CREATE TABLE IF NOT EXISTS leaves (id INTEGER PRIMARY KEY AUTOINCREMENT, student_roll_number TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL, reason_type TEXT NOT NULL, description TEXT, status TEXT DEFAULT 'Pending', document_path TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);""")
    
    # NEW: Class Sessions Table (For Mass Bunk, Holidays, etc.)
    cur.execute("""CREATE TABLE IF NOT EXISTS class_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_name TEXT NOT NULL,
        date TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Regular'
    );""")

    # Migrations
    cur.execute("PRAGMA table_info(students)")
    columns = [info[1] for info in cur.fetchall()]
    if 'password_hash' not in columns:
        cur.execute("ALTER TABLE students ADD COLUMN password_hash TEXT")
        rows = cur.execute("SELECT roll_number FROM students").fetchall()
        for row in rows:
            default_pw = generate_password_hash(row['roll_number'])
            cur.execute("UPDATE students SET password_hash = ? WHERE roll_number = ?", (default_pw, row['roll_number']))
    if 'year' not in columns:
        cur.execute("ALTER TABLE students ADD COLUMN year TEXT DEFAULT '1st Year'")

    cur.execute("PRAGMA table_info(classes)")
    cls_columns = [info[1] for info in cur.fetchall()]
    if 'branch' not in cls_columns: cur.execute("ALTER TABLE classes ADD COLUMN branch TEXT")
    if 'year' not in cls_columns: cur.execute("ALTER TABLE classes ADD COLUMN year TEXT")

    cur.execute("SELECT id, username FROM users WHERE username = ?;", ("admin",))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?);", ("admin", generate_password_hash("admin123"), "admin"))
    conn.commit()
    conn.close()

# ---------- Flask-Login User Models ----------
class User(UserMixin):
    def __init__(self, user_id, username, password_hash, role):
        self.id = user_id; self.username = username; self.password_hash = password_hash; self.role = role; self.is_student = False
    def get_id(self): return f"u_{self.id}"

class StudentUser(UserMixin):
    def __init__(self, roll_number, name, password_hash, branch, image_path, year):
        self.id = roll_number; self.username = name; self.password_hash = password_hash; self.branch = branch; self.image_path = image_path; self.year = year; self.role = "student"; self.is_student = True
    def get_id(self): return f"s_{self.id}"

@login_manager.user_loader
def load_user(user_id_str):
    conn = get_db_connection()
    cur = conn.cursor()
    if user_id_str.startswith("u_"):
        row = cur.execute("SELECT * FROM users WHERE id = ?", (user_id_str.split("_", 1)[1],)).fetchone()
        if row: return User(row["id"], row["username"], row["password_hash"], row["role"])
    elif user_id_str.startswith("s_"):
        row = cur.execute("SELECT * FROM students WHERE roll_number = ?", (user_id_str.split("_", 1)[1],)).fetchone()
        if row: return StudentUser(row["roll_number"], row["name"], row["password_hash"], row["branch"], row["image_path"], row["year"])
    conn.close()
    return None

# ---------- Scheduling & Leave Logic ----------
def is_class_active(class_name):
    conn = get_db_connection()
    class_info = conn.execute("SELECT id FROM classes WHERE class_name = ?", (class_name,)).fetchone()
    if not class_info:
        conn.close(); return False, "Class not found."
    now = datetime.now()
    slot = conn.execute("""SELECT * FROM timetable WHERE class_id = ? AND day_of_week = ? AND ? BETWEEN start_time AND end_time""",
                        (class_info['id'], now.strftime('%A'), now.strftime('%H:%M'))).fetchone()
    conn.close()
    return (True, f"Active ({slot['start_time']}-{slot['end_time']})") if slot else (False, "Not scheduled.")

def get_next_occurrences(day_name, weeks=4):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    today = date.today()
    try: target_idx = days.index(day_name)
    except ValueError: return []
    current_idx = today.weekday()
    days_ahead = target_idx - current_idx
    if days_ahead < 0: days_ahead += 7
    next_date = today + timedelta(days=days_ahead)
    dates = []
    for i in range(weeks):
        dates.append((next_date + timedelta(weeks=i)).isoformat())
    return dates

def predict_risk(history_data):
    if not history_data: return 0.0
    X = np.array(range(1, len(history_data) + 1)).reshape(-1, 1)
    y = np.cumsum(history_data)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.array([[TOTAL_SEMESTER_CLASSES]])
    predicted_total = model.predict(future_X)[0]
    predicted_total = max(0, min(TOTAL_SEMESTER_CLASSES, predicted_total))
    return round((predicted_total / TOTAL_SEMESTER_CLASSES) * 100, 1)

def check_if_on_leave(roll, check_date):
    conn = get_db_connection()
    leave = conn.execute("""SELECT 1 FROM leaves WHERE student_roll_number = ? AND status = 'Approved' AND ? BETWEEN start_date AND end_date""", (roll, check_date)).fetchone()
    conn.close()
    return leave is not None

def already_marked_today(roll, class_name):
    conn = get_db_connection()
    hit = conn.execute("SELECT 1 FROM attendance WHERE student_roll_number=? AND class_name=? AND DATE(timestamp)=DATE('now','localtime')", (roll, class_name)).fetchone()
    conn.close()
    return hit is not None

def mark_attendance(roll, name, class_name):
    conn = get_db_connection()
    conn.execute("INSERT INTO attendance (student_roll_number, name, class_name, timestamp) VALUES (?,?,?,?)", (roll, name, class_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit(); conn.close()

# ---------- Recognition ----------
MODEL_NAME = "Facenet512"
student_embeddings = {}

def build_student_embeddings():
    global student_embeddings
    student_embeddings.clear()
    conn = get_db_connection()
    rows = conn.execute("SELECT roll_number, name, image_path FROM students").fetchall()
    conn.close()
    for roll, name, image_path in rows:
        if image_path and os.path.exists(image_path):
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    rep = DeepFace.represent(img, model_name=MODEL_NAME, detector_backend="opencv", enforce_detection=False)
                    if rep: student_embeddings[roll] = (name, np.array(rep[0]["embedding"], dtype="float32"))
            except: pass

def recognize_face(frame_bgr):
    try:
        rep = DeepFace.represent(frame_bgr, model_name=MODEL_NAME, detector_backend="opencv", enforce_detection=False)
        if not rep: return None, None, None
        emb = np.array(rep[0]["embedding"], dtype="float32")
        best_roll, best_name, best_score = None, None, -1.0
        for roll, (name, ref_emb) in student_embeddings.items():
            score = np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb))
            if score > best_score: best_score, best_roll, best_name = score, roll, name
        return (best_roll, best_name, best_score) if best_score >= 0.35 else (None, None, None)
    except: return None, None, None

# ---------- CORE ROUTES ----------

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated: return redirect(url_for("student_dashboard" if getattr(current_user, 'is_student', False) else "index"))
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM users WHERE username=?", (u,)).fetchone()
        conn.close()
        if row and check_password_hash(row["password_hash"], p):
            login_user(User(row["id"], row["username"], row["password_hash"], row["role"]))
            return redirect(url_for("index"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    if current_user.is_authenticated: return redirect(url_for("student_dashboard" if getattr(current_user, 'is_student', False) else "index"))
    if request.method == "POST":
        roll, pw = request.form["roll_number"], request.form["password"]
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM students WHERE roll_number=?", (roll,)).fetchone()
        conn.close()
        if row and row["password_hash"] and check_password_hash(row["password_hash"], pw):
            login_user(StudentUser(row["roll_number"], row["name"], row["password_hash"], row["branch"], row["image_path"], row["year"]))
            return redirect(url_for("student_dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("student_login.html")

@app.route("/logout")
@login_required
def logout(): logout_user(); return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    if getattr(current_user, 'is_student', False): return redirect(url_for("student_dashboard"))
    conn = get_db_connection()
    selected_class = request.args.get("class_filter")
    selected_date = request.args.get("date_filter") or date.today().isoformat()
    allowed = [c["class_name"] for c in get_user_classes(current_user.id, current_user.role)]
    
    q = """SELECT a.id, s.roll_number, s.name, s.branch, s.year, a.class_name, a.timestamp 
           FROM attendance a JOIN students s ON a.student_roll_number = s.roll_number"""
    cond, params = [], []
    if current_user.role == "teacher" and allowed:
        cond.append(f"a.class_name IN ({','.join(['?']*len(allowed))})"); params.extend(allowed)
    if selected_class and selected_class != "all": cond.append("a.class_name = ?"); params.append(selected_class)
    if selected_date: cond.append("DATE(a.timestamp) = ?"); params.append(selected_date)
    if cond: q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY a.timestamp DESC"
    records = conn.execute(q, tuple(params)).fetchall()
    
    total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    present_today = 0
    if allowed:
        present_today = conn.execute(f"SELECT COUNT(DISTINCT student_roll_number) FROM attendance WHERE DATE(timestamp) = ? AND class_name IN ({','.join(['?']*len(allowed))})", (selected_date, *allowed)).fetchone()[0]
    else:
        present_today = conn.execute("SELECT COUNT(DISTINCT student_roll_number) FROM attendance WHERE DATE(timestamp) = ?", (selected_date,)).fetchone()[0]
    all_students = conn.execute("SELECT roll_number, name FROM students ORDER BY name").fetchall()
    conn.close()
    return render_template("index.html", records=records, classes=allowed, selected_class=selected_class, selected_date=selected_date, total_students=total_students, present_today=present_today, absent_today=total_students-present_today, total_classes=len(allowed), class_labels=allowed, present_counts=[], absent_counts=[], all_students=all_students)

# ---------- LEAVE MANAGEMENT ----------
@app.route("/student/leave", methods=["GET", "POST"])
@login_required
def student_leave():
    if not getattr(current_user, 'is_student', False): return redirect(url_for("index"))
    conn = get_db_connection()
    if request.method == "POST":
        start, end = request.form.get("start_date"), request.form.get("end_date")
        reason, desc = request.form.get("reason_type"), request.form.get("description")
        file = request.files.get("document")
        doc_path = None
        if file and file.filename:
            filename = secure_filename(f"{current_user.id}_{int(time.time())}_{file.filename}")
            path = os.path.join(LEAVE_DOCS_DIR, filename)
            file.save(path)
            doc_path = path.replace("\\", "/")
        conn.execute("INSERT INTO leaves (student_roll_number, start_date, end_date, reason_type, description, document_path) VALUES (?,?,?,?,?,?)",
                     (current_user.id, start, end, reason, desc, doc_path))
        conn.commit()
        flash("Leave requested successfully.", "success")
        return redirect(url_for("student_leave"))
    my_leaves = conn.execute("SELECT * FROM leaves WHERE student_roll_number=? ORDER BY created_at DESC", (current_user.id,)).fetchall()
    conn.close()
    return render_template("student_leave.html", leaves=my_leaves)

@app.route("/admin/leaves", methods=["GET", "POST"])
@login_required
def manage_leaves():
    # Allow Teachers and Admins to view
    if getattr(current_user, 'is_student', False): return redirect(url_for("student_dashboard"))
    conn = get_db_connection()
    if request.method == "POST":
        if current_user.role != "admin":
            flash("Only Admins can approve/reject leaves.", "danger")
        else:
            lid = request.form.get("leave_id")
            action = request.form.get("action")
            status = "Approved" if action == "approve" else "Rejected"
            conn.execute("UPDATE leaves SET status = ? WHERE id = ?", (status, lid))
            conn.commit()
            flash(f"Leave {status}.", "success")
        return redirect(url_for("manage_leaves"))
    leaves = conn.execute("""
        SELECT l.*, s.name, s.branch, s.year 
        FROM leaves l 
        JOIN students s ON l.student_roll_number = s.roll_number 
        ORDER BY CASE WHEN l.status = 'Pending' THEN 1 ELSE 2 END, l.start_date DESC
    """).fetchall()
    conn.close()
    return render_template("manage_leaves.html", leaves=leaves)

@app.route("/student/dashboard")
@login_required
def student_dashboard():
    if not getattr(current_user, 'is_student', False): return redirect(url_for("index"))
    start_date, end_date = request.args.get('start_date'), request.args.get('end_date')
    conn = get_db_connection(); cur = conn.cursor()
    relevant_classes_rows = cur.execute("SELECT class_name FROM classes WHERE branch = ? AND year = ?", (current_user.branch, current_user.year)).fetchall()
    relevant_class_names = [row['class_name'] for row in relevant_classes_rows]
    if not relevant_class_names:
        relevant_class_names = [row['class_name'] for row in cur.execute("SELECT DISTINCT class_name FROM attendance WHERE student_roll_number = ?", (current_user.id,)).fetchall()]
    date_filter_sql, params = "", []
    if start_date: date_filter_sql += " AND DATE(timestamp) >= ?"; params.append(start_date)
    if end_date: date_filter_sql += " AND DATE(timestamp) <= ?"; params.append(end_date)
    history, stats = [], {}
    if relevant_class_names:
        placeholders = ','.join(['?'] * len(relevant_class_names))
        # 1. Get Schedule & Attendance
        all_sessions = cur.execute(f"""SELECT class_name, DATE(timestamp) as date, MAX(timestamp) as last_time FROM attendance WHERE class_name IN ({placeholders}) {date_filter_sql} GROUP BY class_name, DATE(timestamp) ORDER BY date DESC""", tuple(relevant_class_names + params)).fetchall()
        my_attendance_rows = cur.execute("SELECT class_name, DATE(timestamp) as date, timestamp FROM attendance WHERE student_roll_number = ?", (current_user.id,)).fetchall()
        my_attendance_set = {f"{row['class_name']}|{row['date']}" for row in my_attendance_rows}
        my_attendance_times = {f"{row['class_name']}|{row['date']}": row['timestamp'] for row in my_attendance_rows}
        
        # 2. Get Special Statuses (Mass Bunk, Holidays)
        session_status_rows = cur.execute(f"SELECT class_name, date, status FROM class_sessions WHERE class_name IN ({placeholders}) {date_filter_sql.replace('timestamp', 'date')}", tuple(relevant_class_names + params)).fetchall()
        session_status_map = {f"{row['class_name']}|{row['date']}": row['status'] for row in session_status_rows}

        # 3. Get Leaves
        approved_leaves = cur.execute("SELECT start_date, end_date FROM leaves WHERE student_roll_number=? AND status='Approved'", (current_user.id,)).fetchall()
        
        prediction_data = defaultdict(list)
        stats = {c: {'present': 0, 'total': 0, 'risk': 0} for c in relevant_class_names}
        
        for session in all_sessions:
            c_name, s_date = session['class_name'], session['date']
            key = f"{c_name}|{s_date}"
            
            is_present = key in my_attendance_set
            
            # Check special status first
            special_status = session_status_map.get(key)
            
            is_on_leave = False
            if not is_present and not special_status:
                for leave in approved_leaves:
                    if leave['start_date'] <= s_date <= leave['end_date']:
                        is_on_leave = True
                        break
            
            # Determine Final Status Display
            if special_status:
                status = special_status # e.g. "Mass Bunk", "Holiday"
            elif is_present:
                status = 'Present'
            elif is_on_leave:
                status = 'On Leave'
            else:
                status = 'Absent'

            if c_name in stats:
                # Stats Logic:
                # If "Holiday" or "Rescheduled", don't count in total
                # If "Mass Bunk", count in total but not present
                
                if special_status in ['Holiday', 'Teacher on Leave', 'Class Rescheduled', 'Gazette Holiday']:
                    pass # Don't add to total
                else:
                    stats[c_name]['total'] += 1
                    
                    # Risk Logic
                    val = 1 if (is_present or is_on_leave) else 0
                    prediction_data[c_name].append(val)
                    
                    if is_present: stats[c_name]['present'] += 1
            
            history.append({'date': s_date, 'class_name': c_name, 'status': status, 'time': my_attendance_times.get(key, 'N/A')})
        
        for cls in stats:
            data_series = list(reversed(prediction_data[cls]))
            stats[cls]['risk'] = predict_risk(data_series)

    conn.close()
    return render_template("student_dashboard.html", student=current_user, history=history, stats=stats, start_date=start_date, end_date=end_date)

# ---------- NEW: Report Card Generator ----------
@app.route("/student/<roll_number>/report_card")
@login_required
def generate_report_card(roll_number):
    if getattr(current_user, 'is_student', False) and current_user.id != roll_number:
        flash("Unauthorized access.", "danger")
        return redirect(url_for("student_dashboard"))
        
    conn = get_db_connection()
    student = conn.execute("SELECT * FROM students WHERE roll_number = ?", (roll_number,)).fetchone()
    if not student:
        return "Student not found", 404
        
    relevant_classes = conn.execute("SELECT class_name FROM classes WHERE branch=? AND year=?", (student['branch'], student['year'])).fetchall()
    class_names = [r['class_name'] for r in relevant_classes]
    
    stats_data = []
    total_p, total_c = 0, 0
    
    for cls in class_names:
        total = conn.execute("SELECT COUNT(DISTINCT DATE(timestamp)) FROM attendance WHERE class_name=?", (cls,)).fetchone()[0]
        present = conn.execute("SELECT COUNT(DISTINCT DATE(timestamp)) FROM attendance WHERE class_name=? AND student_roll_number=?", (cls, roll_number)).fetchone()[0]
        
        if total > 0:
            pct = (present/total)*100
            grade = 'A+' if pct >= 90 else 'A' if pct >= 80 else 'B' if pct >= 70 else 'C' if pct >= 60 else 'F'
            stats_data.append([cls, total, present, f"{pct:.1f}%", grade])
            total_p += present
            total_c += total
    
    overall_pct = (total_p/total_c*100) if total_c > 0 else 0
    overall_grade = 'A+' if overall_pct >= 90 else 'A' if overall_pct >= 80 else 'B' if overall_pct >= 70 else 'C' if overall_pct >= 60 else 'F'

    # --- WATERMARK FUNCTION ---
    def add_watermark(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica-Bold', 70)
        canvas.setFillColor(colors.lightgrey)
        canvas.setFillAlpha(0.3)
        width, height = A4
        canvas.translate(width / 2, height / 2)
        canvas.rotate(45)
        canvas.drawCentredString(0, 0, "NSUT OFFICIAL")
        canvas.restoreState()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    elements = []
    
    # --- LOGO ---
    logo_path = os.path.join("static", "nsut_logo.png")
    if os.path.exists(logo_path):
        im = RLImage(logo_path, width=1.5*inch, height=1.5*inch)
        im.hAlign = 'CENTER'
        elements.append(im)
        elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>NSUT Attendance Report Card</b>", styles['Title']))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph(f"<b>Name:</b> {student['name']}", styles['Normal']))
    elements.append(Paragraph(f"<b>Roll Number:</b> {student['roll_number']}", styles['Normal']))
    elements.append(Paragraph(f"<b>Branch/Year:</b> {student['branch']} - {student['year']}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date:</b> {date.today().strftime('%Y-%m-%d')}", styles['Normal']))
    elements.append(Spacer(1, 24))

    # Pie Chart
    if total_c > 0 and plt:
        fig = plt.figure(figsize=(4, 3))
        fig.patch.set_alpha(0.0) # Transparent background
        
        # Transparent Wedges
        plt.pie([total_p, total_c - total_p], labels=['Present', 'Absent'], autopct='%1.1f%%',
                colors=['#198754', '#dc3545'], startangle=90,
                wedgeprops={'alpha': 0.6})
                
        plt.title("Overall Attendance")
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', transparent=True)
        img_buffer.seek(0)
        elements.append(RLImage(img_buffer, width=250, height=180))
        plt.close()
        elements.append(Spacer(1, 24))

    data = [['Subject', 'Total', 'Present', '%', 'Grade']] + stats_data
    t = Table(data, colWidths=[200, 60, 60, 60, 60])
    
    transparent_grey = colors.Color(0.85, 0.85, 0.85, alpha=0.5)
    transparent_beige = colors.Color(0.96, 0.96, 0.86, alpha=0.5)

    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), transparent_grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), transparent_beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 24))

    elements.append(Paragraph(f"<b>Overall Percentage:</b> {overall_pct:.1f}%", styles['Heading3']))
    grade_color = "green" if overall_grade in ['A+', 'A', 'B'] else "orange" if overall_grade == 'C' else "red"
    elements.append(Paragraph(f"Final Grade: <font color='{grade_color}'>{overall_grade}</font>", styles['Heading3']))
    elements.append(Spacer(1, 48))

    elements.append(Paragraph("_______________________", styles['Normal']))
    elements.append(Paragraph("<b>HOD / Principal Signature</b>", styles['Normal']))

    doc.build(elements, onFirstPage=add_watermark, onLaterPages=add_watermark)
    buffer.seek(0)
    conn.close()
    return send_file(buffer, as_attachment=True, download_name=f"{roll_number}_Report_Card.pdf", mimetype='application/pdf')

# ---------- Teacher History ----------
@app.route("/teacher/history")
@login_required
def teacher_history():
    if getattr(current_user, 'is_student', False): return redirect(url_for("student_dashboard"))
    conn = get_db_connection()

    # --- 1. Filter Logic (Branch/Year) ---
    filter_branch, filter_year = request.args.get('branch'), request.args.get('year')
    all_branches = [r['branch'] for r in conn.execute("SELECT DISTINCT branch FROM classes WHERE branch IS NOT NULL ORDER BY branch").fetchall()]
    all_years = [r['year'] for r in conn.execute("SELECT DISTINCT year FROM classes WHERE year IS NOT NULL ORDER BY year").fetchall()]
    
    # Determine Allowed Classes based on Role
    if current_user.role == 'admin':
        if filter_branch and filter_year:
            rows = conn.execute("SELECT class_name FROM classes WHERE branch = ? AND year = ?", (filter_branch, filter_year)).fetchall()
            allowed = [r['class_name'] for r in rows]
        else:
            allowed = [c["class_name"] for c in get_user_classes(current_user.id, current_user.role)]
    else:
        allowed = [c["class_name"] for c in get_user_classes(current_user.id, current_user.role)]
    
    calendar_events = []

    # --- 2. Fetch Standard Attendance (Green Events) ---
    history_q = "SELECT class_name, DATE(timestamp) as date, COUNT(DISTINCT student_roll_number) as present_count FROM attendance WHERE 1=1"
    hist_params = []
    if allowed:
        history_q += f" AND class_name IN ({','.join(['?']*len(allowed))})"; hist_params.extend(allowed)
    else: history_q += " AND 1=0"
    
    history_q += " GROUP BY class_name, DATE(timestamp) ORDER BY date DESC, class_name ASC"
    all_past_sessions = conn.execute(history_q, tuple(hist_params)).fetchall()
    
    for session in all_past_sessions:
        calendar_events.append({
            'title': f"{session['class_name']} ({session['present_count']})",
            'start': session['date'],
            'color': '#198754', # Green (Success)
            'textColor': '#ffffff',
            'display': 'block',
            'extendedProps': {'type': 'past', 'class_name': session['class_name'], 'count': session['present_count'], 'date': session['date']}
        })

    if allowed:
        placeholders = ','.join(['?'] * len(allowed))
        
        # --- 3. Fetch Special Sessions (Mass Bunk, Rescheduled, Holidays) ---
        special_sessions = conn.execute(f"SELECT class_name, date, status FROM class_sessions WHERE class_name IN ({placeholders})", tuple(allowed)).fetchall()
        
        for s in special_sessions:
            status = s['status']
            
            # --- FIX: Skip "Regular" statuses so they don't show as special grey events ---
            if status in ['Regular', 'Regular Class']:
                continue
            
            # DEFAULT COLORS
            bg_color = '#6c757d' # Grey
            text_color = '#ffffff'
            
            # MAP STATUS TO COLORS
            if status in ['Mass Bunk', 'Teacher on Leave', 'Gazette Holiday']:
                bg_color = '#dc3545' # Red (Danger/Cancelled)
            
            elif status == 'Class Rescheduled':
                bg_color = '#ffc107' # Yellow/Orange (Warning)
                text_color = '#000000' # Black text for contrast

            calendar_events.append({
                'title': f"{s['class_name']}: {status}",
                'start': s['date'],
                'color': bg_color,
                'textColor': text_color,
                'display': 'block',
                'extendedProps': {
                    'type': 'special',
                    'class_name': s['class_name'],
                    'status': status,
                    'date': s['date']
                }
            })

        # --- 4. Fetch Recurring Timetable (Blue Events) ---
        schedule = conn.execute(f"""SELECT t.*, c.class_name, c.branch, c.year, GROUP_CONCAT(u.username, ', ') as teachers 
                                    FROM timetable t 
                                    JOIN classes c ON t.class_id = c.id 
                                    LEFT JOIN teacher_classes tc ON c.class_name = tc.class_name 
                                    LEFT JOIN users u ON tc.teacher_id = u.id 
                                    WHERE c.class_name IN ({placeholders}) GROUP BY t.id""", tuple(allowed)).fetchall()
        for s in schedule:
            dates = get_next_occurrences(s['day_of_week'], weeks=4)
            for d in dates:
                calendar_events.append({
                    'title': f"{s['class_name']}",
                    'start': f"{d}T{s['start_time']}",
                    'end': f"{d}T{s['end_time']}",
                    'color': '#0d6efd', # Blue (Primary)
                    'textColor': '#ffffff',
                    'display': 'block',
                    'extendedProps': {
                        'type': 'future',
                        'class_name': s['class_name'],
                        'branch': s['branch'],
                        'year': s['year'],
                        'teacher': s['teachers'] or 'Unassigned',
                        'time': f"{s['start_time']} - {s['end_time']}"
                    }
                })

    conn.close()
    return render_template("teacher_history.html",
                           past_sessions=all_past_sessions[:50],
                           calendar_events=calendar_events,
                           all_branches=all_branches,
                           all_years=all_years,
                           selected_branch=filter_branch,
                           selected_year=filter_year)
# ---------- Attendance Tools ----------
@app.route("/attendance/manual", methods=["GET", "POST"])
@login_required
def mark_manual():
    if getattr(current_user, 'is_student', False): return redirect(url_for("index"))
    conn = get_db_connection()
    if request.method == "POST":
        roll = request.form.get("roll_number")
        class_name = request.form.get("class_name")
        allowed = [c["class_name"] for c in get_user_classes(current_user.id, current_user.role)]
        if not roll or not class_name: flash("Missing data.", "warning")
        elif current_user.role != "admin" and class_name not in allowed: flash("Unauthorized.", "danger")
        else:
            student = conn.execute("SELECT name FROM students WHERE roll_number = ?", (roll,)).fetchone()
            if not student: flash("Student not found.", "danger")
            elif not already_marked_today(roll, class_name):
                mark_attendance(roll, student["name"], class_name)
                flash(f"Marked {student['name']} present.", "success")
            else: flash("Already marked.", "info")
        conn.close(); return redirect(url_for("mark_manual"))
    students = conn.execute("SELECT roll_number, name FROM students ORDER BY name").fetchall()
    classes = get_user_classes(current_user.id, current_user.role)
    allowed_classes = [c["class_name"] for c in classes]
    conn.close()
    return render_template("manual_attendance.html", students=students, classes=allowed_classes)

@app.route("/attendance/edit", methods=["GET"])
@login_required
def edit_attendance_page():
    if getattr(current_user, 'is_student', False): return redirect(url_for("index"))
    cls, dt = request.args.get("class_name"), request.args.get("date")
    conn = get_db_connection()
    students = conn.execute("SELECT * FROM students").fetchall()
    
    # Check for session status
    session_info = conn.execute("SELECT status FROM class_sessions WHERE class_name=? AND date=?", (cls, dt)).fetchone()
    current_status = session_info['status'] if session_info else "Regular"
    
    present = {r["student_roll_number"] for r in conn.execute("SELECT student_roll_number FROM attendance WHERE class_name=? AND DATE(timestamp)=?", (cls, dt)).fetchall()}
    conn.close()
    return render_template("edit_attendance.html",
                           classes=[c["class_name"] for c in get_user_classes(current_user.id, current_user.role)],
                           students=students,
                           present_rolls=present,
                           selected_class=cls,
                           selected_date=dt,
                           current_status=current_status)

@app.route("/attendance/save", methods=["POST"])
@login_required
def save_attendance():
    cls, dt = request.form.get("class_name"), request.form.get("date")
    status = request.form.get("session_status")
    present_rolls = request.form.getlist("present_rolls")
    
    conn = get_db_connection()
    
    # 1. Update Session Status
    exists = conn.execute("SELECT 1 FROM class_sessions WHERE class_name=? AND date=?", (cls, dt)).fetchone()
    if exists:
        conn.execute("UPDATE class_sessions SET status=? WHERE class_name=? AND date=?", (status, cls, dt))
    else:
        conn.execute("INSERT INTO class_sessions (class_name, date, status) VALUES (?,?,?)", (cls, dt, status))

    # 2. Handle Attendance
    conn.execute("DELETE FROM attendance WHERE class_name=? AND DATE(timestamp)=?", (cls, dt))
    
    if status == 'Regular':
        for roll in present_rolls:
            s = conn.execute("SELECT name FROM students WHERE roll_number=?", (roll,)).fetchone()
            if s: conn.execute("INSERT INTO attendance (student_roll_number, name, class_name, timestamp) VALUES (?,?,?,?)", (roll, s["name"], cls, f"{dt} 12:00:00"))
    
    conn.commit(); conn.close()
    flash(f"Attendance updated (Status: {status}).", "success")
    return redirect(url_for("edit_attendance_page", class_name=cls, date=dt))

@app.route("/attendance/update", methods=["POST"])
@login_required
def update_attendance():
    aid, new_cls, new_tm = request.form.get("attendance_id"), request.form.get("class_name"), request.form.get("timestamp")
    if aid and new_cls and new_tm:
        conn = get_db_connection()
        conn.execute("UPDATE attendance SET class_name=?, timestamp=? WHERE id=?", (new_cls, new_tm.replace("T", " ")+":00" if len(new_tm)==16 else new_tm.replace("T", " "), aid))
        conn.commit(); conn.close(); flash("Updated.", "success")
    return redirect(url_for("index"))

@app.route("/attendance/delete/<int:aid>", methods=["POST"])
@login_required
def delete_attendance(aid):
    conn = get_db_connection(); conn.execute("DELETE FROM attendance WHERE id=?", (aid,)); conn.commit(); conn.close(); flash("Deleted.", "success"); return redirect(url_for("index"))

@app.route('/recognize', methods=['POST'])
@login_required
def recognize():
    if getattr(current_user, 'is_student', False): return jsonify({"status": "error"})
    data = request.get_json()
    
    user_lat, user_lon = data.get('latitude'), data.get('longitude')
    if user_lat is None or user_lon is None: return jsonify({"status": "error", "message": "Location Required."})
    dist = haversine(user_lat, user_lon, CLASSROOM_LAT, CLASSROOM_LON)
    if dist > MAX_DISTANCE_KM and (CLASSROOM_LAT != 0.0 or CLASSROOM_LON != 0.0):
        return jsonify({"status": "error", "message": f"Too far ({dist:.2f}km). Go to class."})
    
    class_name = data.get('class_name', 'General')
    active, msg = is_class_active(class_name)
    if not active and current_user.role != "admin":
        return jsonify({"status": "error", "message": msg})

    img_bytes = base64.b64decode(data['image_data'].split(',')[1])
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    roll, name, score = recognize_face(frame)
    if roll:
        if not already_marked_today(roll, data.get('class_name', 'General')):
            mark_attendance(roll, name, data.get('class_name', 'General'))
            return jsonify({"status": "success", "name": name, "roll": roll})
        return jsonify({"status": "already_marked", "name": name})
    return jsonify({"status": "not_recognized"})

# ---------- NEW: Student Bulk Upload ----------
@app.route("/students/upload", methods=["POST"])
@login_required
def upload_students_zip():
    if getattr(current_user, 'is_student', False) or current_user.role != "admin": return redirect(url_for("index"))
    file = request.files.get('zip_file')
    def_branch = request.form.get('default_branch', 'General')
    def_year = request.form.get('default_year', '1st Year')
    if not file or not file.filename.endswith('.zip'): flash("Please upload a valid .zip file", "danger"); return redirect(url_for("students_page"))
    added_count = 0
    try:
        with zipfile.ZipFile(file, 'r') as z:
            for filename in z.namelist():
                if filename.startswith('__MACOSX') or filename.endswith('/'): continue
                ext = os.path.splitext(filename)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']: continue
                base = os.path.basename(filename)
                name_part = os.path.splitext(base)[0]
                
                # FLEXIBLE PARSING (Name + Roll, any order)
                roll_match = re.search(r'(\d{4}[a-zA-Z]{3}\d{4})', name_part)
                
                if roll_match:
                    roll = roll_match.group(1).upper()
                    
                    # Extract Name
                    remaining = name_part.replace(roll_match.group(0), '')
                    name_clean = re.sub(r'[_\d]', ' ', remaining).strip()
                    name = name_clean if name_clean else roll

                    safe_name = f"{roll}_{name.replace(' ', '_')}{ext}"
                    target_path = os.path.join(UPLOAD_DIR, safe_name)
                    with open(target_path, "wb") as f: f.write(z.read(filename))
                    
                    conn = get_db_connection()
                    web_path = target_path.replace("\\", "/")
                    exists = conn.execute("SELECT 1 FROM students WHERE roll_number=?", (roll,)).fetchone()
                    if exists: conn.execute("UPDATE students SET name=?, branch=?, year=?, image_path=? WHERE roll_number=?", (name, def_branch, def_year, web_path, roll))
                    else: conn.execute("INSERT INTO students (roll_number, name, branch, year, image_path, password_hash) VALUES (?,?,?,?,?,?)", (roll, name, def_branch, def_year, web_path, generate_password_hash(roll)))
                    conn.commit(); conn.close()
                    added_count += 1
        build_student_embeddings()
        flash(f"Successfully processed {added_count} students.", "success")
    except Exception as e: flash(f"Error processing zip: {str(e)}", "danger")
    return redirect(url_for("students_page"))

# ---------- Standard Routes ----------
@app.route("/students", methods=["GET", "POST"])
@login_required
def students_page():
    # 1. Security Check
    if getattr(current_user, 'is_student', False): return redirect(url_for("index"))
    
    conn = get_db_connection()
    
    # 2. Handle Add/Edit (POST)
    if request.method == "POST":
        roll, name, branch, year = request.form.get("roll"), request.form.get("name"), request.form.get("branch"), request.form.get("year")
        file = request.files.get('image')
        img_path = None
        
        if file and file.filename:
            path = os.path.join(UPLOAD_DIR, f"{roll}_{name}{os.path.splitext(file.filename)[1]}")
            file.save(path); img_path = path.replace("\\", "/")
            
        if conn.execute("SELECT 1 FROM students WHERE roll_number=?", (roll,)).fetchone():
            if img_path: conn.execute("UPDATE students SET name=?, branch=?, year=?, image_path=? WHERE roll_number=?", (name, branch, year, img_path, roll))
            else: conn.execute("UPDATE students SET name=?, branch=?, year=? WHERE roll_number=?", (name, branch, year, roll))
        else:
            conn.execute("INSERT INTO students (roll_number, name, branch, year, image_path, password_hash) VALUES (?,?,?,?,?,?)", (roll, name, branch, year, img_path, generate_password_hash(roll)))
        
        conn.commit()
        build_student_embeddings()
        return redirect(url_for('students_page'))

    # 3. Handle Filters (GET) - UPDATED
    q = request.args.get('q', '').strip()
    f_branch = request.args.get('branch', '')
    f_year = request.args.get('year', '')
    
    # Start building dynamic query
    sql = "SELECT * FROM students WHERE 1=1"
    params = []

    # Text Search (Roll or Name)
    if q:
        sql += " AND (roll_number LIKE ? OR name LIKE ?)"
        wildcard = f"%{q}%"
        params.extend([wildcard, wildcard])
    
    # Branch Filter
    if f_branch:
        sql += " AND branch = ?"
        params.append(f_branch)
        
    # Year Filter
    if f_year:
        sql += " AND year = ?"
        params.append(f_year)

    sql += " ORDER BY roll_number ASC"
    
    rows = conn.execute(sql, tuple(params)).fetchall()
    conn.close()
    
    # Return filters to template to keep dropdowns selected
    return render_template("students.html", rows=rows, q=q, f_branch=f_branch, f_year=f_year)

@app.route("/students/<roll_number>/delete", methods=["POST"])
@login_required
def delete_student(roll_number):
    conn = get_db_connection()
    conn.execute("DELETE FROM students WHERE roll_number=?", (roll_number,))
    conn.commit(); conn.close(); build_student_embeddings()
    return redirect(url_for("students_page"))

@app.route("/admin")
@login_required
def admin_panel(): return render_template("admin.html", users=get_db_connection().execute("SELECT * FROM users").fetchall())

@app.route("/admin/assign_class", methods=["GET", "POST"])
@login_required
def assign_class():
    conn = get_db_connection()
    if request.method == "POST":
        if "assign" in request.form:
            conn.execute("INSERT INTO teacher_classes (teacher_id, class_name) VALUES (?, ?);", (request.form.get("teacher_id"), request.form.get("class_name")))
        elif "delete" in request.form:
            conn.execute("DELETE FROM teacher_classes WHERE id = ?;", (request.form.get("assign_id"),))
        conn.commit(); return redirect(url_for("assign_class"))
    teachers = conn.execute("SELECT id, username FROM users WHERE role='teacher';").fetchall()
    classes = conn.execute("SELECT class_name, branch, year FROM classes;").fetchall()
    assigned = conn.execute("SELECT t.id, u.username, t.class_name FROM teacher_classes t JOIN users u ON u.id = t.teacher_id;").fetchall()
    return render_template("assign_class.html", teachers=teachers, classes=classes, assigned=assigned)

@app.route("/admin/create", methods=["POST"])
@login_required
def create_user():
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?);", (request.form.get("username"), generate_password_hash(request.form.get("password")), request.form.get("role")))
        conn.commit(); flash("Created.", "success")
    except: flash("Exists.", "warning")
    return redirect(url_for("admin_panel"))

@app.route("/admin/delete/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    conn = get_db_connection(); conn.execute("DELETE FROM users WHERE id=?", (user_id,)); conn.commit(); flash("Deleted.", "success"); return redirect(url_for("admin_panel"))

@app.route("/capture_student", methods=["GET", "POST"])
@login_required
def capture_student():
    if request.method == "POST": return redirect(url_for("students_page"))
    return render_template("capture_student.html")

@app.route("/classes", methods=["GET", "POST"])
@login_required
def classes_page():
    conn = get_db_connection()
    # Handle Add Class
    if request.method == "POST" and "class_id" not in request.form:
        try:
            conn.execute("INSERT INTO classes (class_name, branch, year) VALUES (?, ?, ?);",
                         (request.form.get("class_name"), request.form.get("branch"), request.form.get("year")))
            conn.commit(); flash("Added.", "success")
        except: flash("Exists.", "warning")
    
    # Handle Timetable Management
    timetable = {} # {class_id: [slots]}
    classes = conn.execute("SELECT * FROM classes").fetchall()
    slots = conn.execute("SELECT * FROM timetable ORDER BY day_of_week, start_time").fetchall()
    
    for s in slots:
        s_dict = dict(s)
        if s_dict['class_id'] not in timetable:
            timetable[s_dict['class_id']] = []
        timetable[s_dict['class_id']].append(s_dict)
        
    conn.close()
    return render_template("classes.html", classes=classes, timetable=timetable)

@app.route("/classes/edit", methods=["POST"])
@login_required
def edit_class():
    if getattr(current_user, 'is_student', False) or current_user.role != "admin": return redirect(url_for("index"))
    cid, name, branch, year = request.form.get("class_id"), request.form.get("class_name"), request.form.get("branch"), request.form.get("year")
    if cid and name:
        conn = get_db_connection()
        try: conn.execute("UPDATE classes SET class_name=?, branch=?, year=? WHERE id=?", (name, branch, year, cid)); conn.commit(); flash("Updated.", "success")
        except: flash("Error.", "danger")
        conn.close()
    return redirect(url_for("classes_page"))

@app.route("/classes/<int:cid>/delete", methods=["POST"])
@login_required
def delete_class(cid):
    conn = get_db_connection(); conn.execute("DELETE FROM classes WHERE id=?", (cid,)); conn.commit(); flash("Deleted.", "success"); return redirect(url_for("classes_page"))

@app.route("/timetable/add", methods=["POST"])
@login_required
def add_timetable():
    if getattr(current_user, 'is_student', False) or current_user.role != "admin": return redirect(url_for("index"))
    cid, day, start, end = request.form.get("class_id"), request.form.get("day"), request.form.get("start"), request.form.get("end")
    conn = get_db_connection()
    conn.execute("INSERT INTO timetable (class_id, day_of_week, start_time, end_time) VALUES (?, ?, ?, ?)", (cid, day, start, end))
    conn.commit(); conn.close(); flash("Schedule added.", "success")
    return redirect(url_for("classes_page"))

@app.route("/timetable/delete/<int:tid>", methods=["POST"])
@login_required
def delete_timetable(tid):
    if getattr(current_user, 'is_student', False) or current_user.role != "admin": return redirect(url_for("index"))
    conn = get_db_connection(); conn.execute("DELETE FROM timetable WHERE id=?", (tid,)); conn.commit(); conn.close(); flash("Slot removed.", "success")
    return redirect(url_for("classes_page"))

@app.route("/camera")
@login_required
def camera_page(): return render_template("camera.html", classes=get_user_classes(current_user.id, current_user.role))

@app.route("/reset_attendance", methods=["POST"])
@login_required
def reset_attendance():
    cls, dt = request.form.get("class_name"), request.form.get("date_filter")
    conn = get_db_connection()
    if dt: conn.execute("DELETE FROM attendance WHERE class_name=? AND DATE(timestamp)=?", (cls, dt))
    else: conn.execute("DELETE FROM attendance WHERE class_name=?", (cls,))
    conn.commit(); flash("Reset.", "success"); return redirect(url_for("camera_page"))

@app.route("/download")
@login_required
def download():
    conn = get_db_connection()
    rows = conn.execute("SELECT s.roll_number, s.name, s.branch, s.year, a.class_name, a.timestamp FROM attendance a JOIN students s ON a.student_roll_number = s.roll_number ORDER BY a.timestamp DESC").fetchall()
    conn.close()
    out = io.StringIO(); w = csv.writer(out); w.writerow(["Roll", "Name", "Branch", "Year", "Class", "Time"])
    for r in rows: w.writerow([r["roll_number"], r["name"], r["branch"], r["year"], r["class_name"], r["timestamp"]])
    mem = io.BytesIO(out.getvalue().encode("utf-8")); mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")

@app.route("/download_pdf")
@login_required
def download_pdf():
    conn = get_db_connection()
    rows = conn.execute("SELECT s.roll_number, s.name, s.branch, s.year, a.class_name, a.timestamp FROM attendance a JOIN students s ON a.student_roll_number = s.roll_number ORDER BY a.timestamp DESC").fetchall()
    conn.close()
    if not rows or not canvas: return redirect(url_for("summary"))
    out = io.BytesIO(); c = canvas.Canvas(out, pagesize=landscape(A4)); width, height = landscape(A4); y = height - 40
    c.setFont("Helvetica-Bold", 14); c.drawString(40, y, "Attendance Report"); y -= 30
    c.setFont("Helvetica-Bold", 10); headers = ["Roll", "Name", "Branch", "Year", "Class", "Time"]; x = [40, 120, 250, 320, 390, 550]
    for i, h in enumerate(headers): c.drawString(x[i], y, h)
    y -= 20; c.setFont("Helvetica", 10)
    for r in rows:
        data = [r['roll_number'], r['name'], r['branch'], r['year'], r['class_name'], str(r['timestamp'])]
        for i, d in enumerate(data): c.drawString(x[i], y, str(d))
        y -= 15
        if y < 40: c.showPage(); y = height - 40; c.setFont("Helvetica", 10)
    c.save(); out.seek(0)
    return send_file(out, as_attachment=True, download_name="attendance.pdf", mimetype="application/pdf")

@app.route("/rebuild_embeddings")
@login_required
def rebuild_embeddings():
    build_student_embeddings()
    return "Rebuilt.", 200

@app.route("/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        curr, new, conf = request.form.get("current_password"), request.form.get("new_password"), request.form.get("confirm_password")
        if not curr or not new or not conf: flash("Required.", "warning")
        elif new != conf: flash("Mismatch.", "danger")
        elif not check_password_hash(current_user.password_hash, curr): flash("Incorrect.", "danger")
        else:
            h = generate_password_hash(new); conn = get_db_connection()
            if getattr(current_user, 'is_student', False): conn.execute("UPDATE students SET password_hash=? WHERE roll_number=?", (h, current_user.id))
            else: conn.execute("UPDATE users SET password_hash=? WHERE id=?", (h, current_user.id))
            conn.commit(); conn.close(); flash("Updated.", "success"); return redirect(url_for("student_dashboard" if getattr(current_user, 'is_student', False) else "index"))
    return render_template("change_password.html")

# ---------- Reports (Summary) Code Block ----------
def calculate_attendance_data(start, end, class_filter=None):
    conn = get_db_connection()
    date_cond, params = "", []
    if start:
        date_cond += " AND DATE(timestamp) >= ?"
        params.append(start)
    if end:
        date_cond += " AND DATE(timestamp) <= ?"
        params.append(end)
        
    all_classes_counts = dict(conn.execute(f"SELECT class_name, COUNT(DISTINCT DATE(timestamp)) FROM attendance WHERE 1=1 {date_cond} GROUP BY class_name", tuple(params)).fetchall())
    student_rows = conn.execute(f"SELECT student_roll_number, class_name, COUNT(DISTINCT DATE(timestamp)) as attended FROM attendance WHERE 1=1 {date_cond} GROUP BY student_roll_number, class_name", tuple(params)).fetchall()
    all_students = conn.execute("SELECT roll_number, name, branch FROM students").fetchall()
    student_map = {s['roll_number']: s for s in all_students}
    conn.close()
    
    data = []
    target_classes = [class_filter] if class_filter and class_filter != "all" else list(all_classes_counts.keys())
    attendance_map = defaultdict(lambda: defaultdict(int))
    
    for r in student_rows:
        attendance_map[r['student_roll_number']][r['class_name']] = r['attended']
        
    for roll, student in student_map.items():
        for cls in target_classes:
            total = all_classes_counts.get(cls, 0)
            if total == 0: continue
            attended = attendance_map[roll].get(cls, 0)
            data.append({
                'roll': roll,
                'name': student['name'],
                'branch': student['branch'],
                'class_name': cls,
                'attended': attended,
                'total': total,
                'percent': round((attended/total)*100, 1)
            })
    return sorted(data, key=lambda x: (x['name'], x['class_name']))

@app.route("/summary")
@login_required
def summary():
    if getattr(current_user, 'is_student', False):
        return redirect(url_for("student_dashboard"))
        
    start = request.args.get("start")
    end = request.args.get("end")
    class_filter = request.args.get("class_filter")
    threshold = request.args.get("threshold")
    
    allowed = [c["class_name"] for c in get_user_classes(current_user.id, current_user.role)]
    effective_class = class_filter
    if current_user.role == "teacher":
        if class_filter and class_filter not in allowed:
            effective_class = None
            
    all_data = calculate_attendance_data(start, end, effective_class)
    final_rows = []
    
    for row in all_data:
        if current_user.role == "teacher" and row['class_name'] not in allowed:
            continue
        if threshold and threshold.strip():
            try:
                if row['percent'] >= float(threshold): continue
            except: pass
        final_rows.append(row)
        
    return render_template("summary.html", rows=final_rows, start=start, end=end, classes=allowed, class_filter=class_filter, threshold=threshold)

@app.route("/summary/download_excel")
@login_required
def download_summary_excel():
    if getattr(current_user, 'is_student', False):
        return redirect(url_for("student_dashboard"))
        
    start = request.args.get("start")
    end = request.args.get("end")
    class_filter = request.args.get("class_filter")
    threshold = request.args.get("threshold")
    
    allowed = [c["class_name"] for c in get_user_classes(current_user.id, current_user.role)]
    data = calculate_attendance_data(start, end, class_filter)
    filtered = []
    
    for row in data:
        if current_user.role == "teacher" and row['class_name'] not in allowed:
            continue
        if threshold and threshold.strip():
            try:
                if row['percent'] >= float(threshold): continue
            except: pass
        filtered.append(row)
        
    if not filtered:
        flash("No data.", "warning")
        return redirect(url_for("summary"))
    
    # Generate Excel
    if pd:
        df = pd.DataFrame(filtered)[['roll', 'name', 'branch', 'class_name', 'attended', 'total', 'percent']]
        df.columns = ['Roll', 'Name', 'Branch', 'Class', 'Attended', 'Total', '%']
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            df.to_excel(w, index=False)
        out.seek(0)
        return send_file(out, as_attachment=True, download_name="report.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # Fallback to CSV
    out = io.StringIO()
    w = csv.writer(output)
    w.writerow(['Roll', 'Name', 'Branch', 'Class', 'Attended', 'Total', '%'])
    for r in filtered:
        w.writerow([r['roll'], r['name'], r['branch'], r['class_name'], r['attended'], r['total'], r['percent']])
    mem = io.BytesIO(out.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="report.csv", mimetype="text/csv")

@app.route("/download_excel")
@login_required
def download_excel():
    # Redirect legacy route to new one
    return redirect(url_for('download_summary_excel', **request.args))

if __name__ == "__main__":
    setup_database()
    build_student_embeddings()
    app.run(host="0.0.0.0", port=5001, debug=True)
