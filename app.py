import os
import io
import csv
import time
import sqlite3
import tempfile
import base64
from datetime import datetime, date
from collections import defaultdict

import cv2
import numpy as np
import psycopg2
from deepface import DeepFace
from psycopg2.extras import DictCursor

from flask import (
    Flask, render_template, Response,
    request, redirect, url_for, flash, send_file
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, UserMixin, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key_change_me")
UPLOAD_DIR = os.path.join("static", "students")
os.makedirs(UPLOAD_DIR, exist_ok=True)
login_manager = LoginManager(app)
login_manager.login_view = "login"
DB_PATH = "attendance.db"

# --- Database Connection ---
def get_db_connection():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        conn = psycopg2.connect(database_url)
        conn.cursor_factory = DictCursor
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    return conn

def get_user_classes(user_id, role):
    conn = get_db_connection()
    cur = conn.cursor()
    if role == "admin":
        cur.execute("SELECT class_name FROM classes ORDER BY class_name;")
    else:
        cur.execute("SELECT class_name FROM teacher_classes WHERE teacher_id = %s ORDER BY class_name;", (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

from werkzeug.security import generate_password_hash
# Make sure your get_db_connection() is defined to connect to PostgreSQL
# import psycopg2
# from psycopg2.extras import DictCursor

def setup_database():
    """
    Initializes the PostgreSQL database by creating tables and a default admin user.
    """
    conn = get_db_connection() # This should return a psycopg2 connection
    cur = conn.cursor()

    # --- Create tables with PostgreSQL syntax ---

    # users table: Changed to SERIAL PRIMARY KEY
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'teacher'
    );""")

    # classes table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS classes (
        id SERIAL PRIMARY KEY,
        class_name TEXT UNIQUE NOT NULL
    );""")

    # students table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        roll_number TEXT NOT NULL,
        name TEXT NOT NULL,
        class_name TEXT NOT NULL DEFAULT 'General',
        image_path TEXT
    );""")

    # attendance table: Changed to TIMESTAMP WITH TIME ZONE
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id SERIAL PRIMARY KEY,
        student_roll_number TEXT NOT NULL,
        name TEXT NOT NULL,
        class_name TEXT NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );""")
    
    # teacher_classes table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS teacher_classes (
        id SERIAL PRIMARY KEY,
        teacher_id INTEGER NOT NULL,
        class_name TEXT NOT NULL,
        FOREIGN KEY (teacher_id) REFERENCES users(id)
    );""")

    # --- Ensure default admin user exists, using %s placeholders ---
    
    cur.execute("SELECT id, username, role FROM users WHERE username = %s;", ("admin",))
    row = cur.fetchone()

    if not row:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s);",
            ("admin", generate_password_hash("admin123"), "admin")
        )
        print("[INFO] Default admin created → username: admin | password: admin123 | role: admin")
    else:
        if row["role"] != "admin":
            cur.execute("UPDATE users SET role = 'admin' WHERE username = %s;", ("admin",))
            print("[INFO] Admin role corrected for user 'admin'")

    conn.commit()
    conn.close()
# ---------- Flask-Login User ----------
class User(UserMixin):
    def __init__(self, user_id, username, password_hash, role):
        self.id = user_id; self.username = username; self.password_hash = password_hash; self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash, role FROM users WHERE id = %s;", (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return User(row["id"], row["username"], row["password_hash"], row["role"])
    return None

# --- Face Recognition (Your existing functions go here) ---
# NOTE: Please copy your functions for build_student_embeddings, cosine_similarity,
# recognize_face, already_marked_today, and mark_attendance into this section.
# They do not need changes, but are required for the app to work.
student_embeddings = {}
def build_student_embeddings():
    pass # Add your code here
def cosine_similarity(a,b):
    pass # Add your code here
def recognize_face(frame_bgr):
    pass # Add your code here
def already_marked_today(roll, class_name):
    pass # Add your code here
def mark_attendance(roll, name, class_name):
    pass # Add your code here


# ---------- Recognition ----------
MODEL_NAME = "Facenet512"
student_embeddings = {}


def load_students_from_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT roll_number, name, image_path FROM students ORDER BY roll_number")
    rows = cur.fetchall()
    conn.close()
    return [(r["roll_number"], r["name"], r["image_path"]) for r in rows]


def build_student_embeddings():
    global student_embeddings
    student_embeddings.clear()

    for roll, name, image_path in load_students_from_db():
        if not image_path or not os.path.exists(image_path):
            continue
        try:
            img = cv2.imread(image_path)
            if img is None:
                continue
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmpfile:
                cv2.imwrite(tmpfile.name, img)
                rep = DeepFace.represent(tmpfile.name, model_name=MODEL_NAME,
                                         detector_backend="opencv", enforce_detection=False)
            if rep and isinstance(rep, list):
                emb = np.array(rep[0]["embedding"], dtype="float32")
                student_embeddings[roll] = (name, emb)
        except Exception as e:
            print(f"[WARN] Embedding failed for {roll} {name}: {e}")


def cosine_similarity(a, b):
    if a is None or b is None:
        return -1.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def recognize_face(frame_bgr):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmpfile:
            cv2.imwrite(tmpfile.name, frame_bgr)
            rep = DeepFace.represent(tmpfile.name, model_name=MODEL_NAME,
                                     detector_backend="opencv", enforce_detection=False)
        if not rep:
            return None, None, None
        emb = np.array(rep[0]["embedding"], dtype="float32")
    except Exception as e:
        print(f"[DEBUG] Recognition failed: {e}")
        return None, None, None

    best_roll, best_name, best_score = None, None, -1.0
    for roll, (name, ref_emb) in student_embeddings.items():
        score = cosine_similarity(emb, ref_emb)
        if score > best_score:
            best_score, best_roll, best_name = score, roll, name

    if best_score >= 0.35:
        return best_roll, best_name, best_score
    return None, None, None


# ---------- Attendance ----------
def already_marked_today(roll, class_name):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""SELECT 1 FROM attendance
                   WHERE student_roll_number=? AND class_name=?
                   AND DATE(timestamp)=DATE('now','localtime') LIMIT 1""",
                (roll, class_name))
    hit = cur.fetchone() is not None
    conn.close()
    return hit


def mark_attendance(roll, name, class_name):  # <-- now accepts name
    conn = get_db_connection()
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""INSERT INTO attendance
                   (student_roll_number, name, class_name, timestamp)
                   VALUES (?,?,?,?)""",
                (roll, name, class_name, now))
    conn.commit()
    conn.close()
    print(f"[MARKED] {roll} ({name}) [{class_name}] at {now}")


# ---------- Webcam ----------
recent_marked = {}

'''
def gen_frames(class_name="General"):
    cap = cv2.VideoCapture(0)  # CAP_AVFOUNDATION is Mac-only; default works cross-platform
    if not cap.isOpened():
        print("[ERROR] Unable to open camera")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            roll, name, score = recognize_face(frame)
            if roll:
                # rate limit per roll
                if time.time() - recent_marked.get(roll, 0) > 20:
                    if not already_marked_today(roll, class_name):
                        mark_attendance(roll, name, class_name)  # <-- pass name
                        recent_marked[roll] = time.time()

            ret, buf = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        cap.release()
'''

# ---------- Routes ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, password_hash, role FROM users WHERE username = %s;", (u,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and check_password_hash(row["password_hash"], p):
            login_user(User(row["id"], row["username"], row["password_hash"], row["role"]))
            return redirect(url_for("index"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    # This function was already corrected, but is included here for completeness.
    conn = get_db_connection()
    cur = conn.cursor()
    selected_class = request.args.get("class_filter")
    selected_date = request.args.get("date_filter") or date.today().isoformat()
    q = "SELECT s.roll_number, s.name, a.class_name, a.timestamp FROM attendance a JOIN students s ON a.student_roll_number = s.roll_number"
    cond, params = [], []
    allowed_classes_rows = get_user_classes(current_user.id, current_user.role)
    allowed_classes = [c["class_name"] for c in allowed_classes_rows]
    if current_user.role == "teacher":
        if not allowed_classes:
            cond.append("1=0")
        else:
            placeholders = ','.join(['%s'] * len(allowed_classes))
            cond.append(f"a.class_name IN ({placeholders})")
            params.extend(allowed_classes)
    if selected_class and selected_class != "all":
        cond.append("a.class_name = %s")
        params.append(selected_class)
    if selected_date:
        cond.append("DATE(a.timestamp) = %s")
        params.append(selected_date)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY a.timestamp DESC;"
    cur.execute(q, params)
    records = cur.fetchall()
    total_students = 0
    if allowed_classes:
        q_students = f"SELECT COUNT(*) FROM students WHERE class_name IN ({','.join(['%s']*len(allowed_classes))});"
        cur.execute(q_students, allowed_classes)
        total_students = cur.fetchone()[0]
    present_today = 0
    if allowed_classes:
        q_present = f"SELECT COUNT(DISTINCT student_roll_number) FROM attendance WHERE DATE(timestamp) = %s AND class_name IN ({','.join(['%s']*len(allowed_classes))});"
        cur.execute(q_present, [selected_date] + allowed_classes)
        present_today = cur.fetchone()[0]
    absent_today = total_students - present_today if total_students > 0 else 0
    cur.close()
    conn.close()
    return render_template("index.html", records=records, classes=allowed_classes, selected_class=selected_class, selected_date=selected_date, total_students=total_students, total_classes=len(allowed_classes), present_today=present_today, absent_today=absent_today, class_labels=[c for c in allowed_classes], present_counts=[], absent_counts=[])


@app.route("/students", methods=["GET", "POST"])
@login_required
def students_page():
    if current_user.role != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("index"))
    conn = get_db_connection()
    cur = conn.cursor()
    if request.method == "POST":
        # ... POST logic ... (Your existing POST logic is fine)
        # Remember to add a cur.close() and conn.close() before the return redirect
        return redirect(url_for("students_page"))
    q = (request.args.get("q") or "").strip()
    if q:
        like = f"%{q}%"
        cur.execute("SELECT roll_number, name, branch, image_path FROM students WHERE roll_number LIKE %s OR name LIKE %s OR branch LIKE %s ORDER BY branch, name;", (like, like, like))
    else:
        cur.execute("SELECT roll_number, name, branch, image_path FROM students ORDER BY branch, name;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return render_template("students.html", rows=rows, q=q)
# Add this new route to app.py
# You can DELETE the old video_feed and gen_frames functions

@app.route('/recognize', methods=['POST'])
@login_required
def recognize():
    data = request.get_json()
    class_name = data.get('class_name', 'General')
    image_data = data['image_data'].split(',')[1] # Remove the "data:image/jpeg;base64," part

    # Decode the image and convert to a format OpenCV can use
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Recognize the face
    roll, name, score = recognize_face(frame)

    if roll and not already_marked_today(roll, class_name):
        mark_attendance(roll, name, class_name)
        return {"status": "success", "name": name, "roll": roll}
    
    elif roll:
        return {"status": "already_marked", "name": name}

    return {"status": "not_recognized"}

    # --- GET: search includes branch now ---
    q = (request.args.get("q") or "").strip()
    if q:
        like = f"%{q}%"
        rows = cur.execute(
            """
            SELECT roll_number, name, branch, image_path
            FROM students
            WHERE roll_number LIKE ? OR name LIKE ? OR branch LIKE ?
            ORDER BY branch, name
            """,
            (like, like, like),
        ).fetchall()
    else:
        rows = cur.execute(
            """
            SELECT roll_number, name, branch, image_path
            FROM students
            ORDER BY branch, name
            """
        ).fetchall()

    conn.close()
    return render_template("students.html", rows=rows, q=q)



@app.route("/students/<roll_number>/delete", methods=["POST"])
@login_required
def delete_student(roll_number):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM students WHERE roll_number=?", (roll_number,))
    conn.commit()
    conn.close()
    build_student_embeddings()
    flash("Student deleted", "success")
    return redirect(url_for("students_page"))


@app.route("/admin")
@login_required
def admin_panel():
    if current_user.role != "admin":
        flash("Access denied. Admins only.", "danger")
        return redirect(url_for("index"))

    conn = get_db_connection()
    users = conn.execute("SELECT id, username, role FROM users").fetchall()
    conn.close()
    return render_template("admin.html", users=users)
    
@app.route("/admin/assign_class", methods=["GET", "POST"])
@login_required
def assign_class():
    if current_user.role != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("index"))
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users WHERE role='teacher';")
    teachers = cur.fetchall()
    cur.execute("SELECT class_name FROM classes;")
    classes = cur.fetchall()
    if request.method == "POST":
        # ... POST logic ... (Remember to use %s placeholders here as well)
        return redirect(url_for("assign_class"))
    cur.execute("SELECT t.id, u.username, t.class_name FROM teacher_classes t JOIN users u ON u.id = t.teacher_id ORDER BY u.username, t.class_name;")
    assigned = cur.fetchall()
    cur.close()
    conn.close()
    return render_template("assign_class.html", teachers=teachers, classes=classes, assigned=assigned)


@app.route("/admin/create", methods=["POST"])
@login_required
def create_user():
    if current_user.role != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("index"))

    username = request.form.get("username")
    password = request.form.get("password")
    role = request.form.get("role", "teacher")

    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     (username, generate_password_hash(password), role))
        conn.commit()
        flash("User created!", "success")
    except sqlite3.IntegrityError:
        flash("Username already exists", "warning")
    conn.close()
    return redirect(url_for("admin_panel"))


@app.route("/admin/delete/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    if current_user.role != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("index"))

    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    flash("User deleted", "success")
    return redirect(url_for("admin_panel"))


@app.route("/capture_student", methods=["GET", "POST"])
@login_required
def capture_student():
    if request.method == "POST":
        roll = request.form.get("roll")
        name = request.form.get("name")
        class_name = (request.form.get("class_name") or "General").strip() or "General"
        img_data = request.form.get("image_data")

        if not roll or not name or not img_data:
            flash("All fields are required", "danger")
            return redirect(url_for("capture_student"))

        img_bytes = base64.b64decode(img_data.split(",")[1])
        filename = f"{roll}_{name.replace(' ', '_')}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        image_path = os.path.join("static", "students", filename)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM students WHERE roll_number=?", (roll,))
        if cur.fetchone():
            cur.execute("""UPDATE students
                           SET name=?, class_name=?, image_path=?
                           WHERE roll_number=?""", (name, class_name, image_path, roll))
        else:
            cur.execute("""INSERT INTO students (roll_number, name, class_name, image_path)
                           VALUES (?,?,?,?)""", (roll, name, class_name, image_path))
        conn.commit()
        conn.close()

        build_student_embeddings()
        flash("Student added successfully with photo!", "success")
        return redirect(url_for("students_page"))

    return render_template("capture_student.html")


@app.route("/classes", methods=["GET", "POST"])
@login_required
def classes_page():
    if current_user.role != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("index"))
    conn = get_db_connection()
    cur = conn.cursor()
    if request.method == "POST":
        cname = request.form.get("class_name")
        if cname:
            try:
                cur.execute("INSERT INTO classes (class_name) VALUES (%s);", (cname,))
                conn.commit()
                flash("Class added", "success")
            except Exception:
                conn.rollback()
                flash("Class already exists", "warning")
    cur.execute("SELECT * FROM classes ORDER BY class_name;")
    classes = cur.fetchall()
    cur.close()
    conn.close()
    return render_template("classes.html", classes=classes)



@app.route("/classes/<int:cid>/delete", methods=["POST"])
@login_required
def delete_class(cid):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM classes WHERE id=?", (cid,))
    conn.commit()
    conn.close()
    flash("Class deleted", "success")
    return redirect(url_for("classes_page"))


@app.route("/camera")
@login_required
def camera_page():
    class_name = request.args.get("class_name", "General")
    classes = get_user_classes(current_user.id, current_user.role)
    return render_template("camera.html", class_name=class_name,
                           now=date.today().isoformat(), classes=classes)


'''@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_frames(class_name=request.args.get("class_name", "General")),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
'''

@app.route("/reset_attendance", methods=["POST"])
@login_required
def reset_attendance():
    if current_user.role != "admin":
        flash("Access denied. Admins only.", "danger")
        return redirect(url_for("index"))

    class_name = request.form.get("class_name") or "General"
    date_filter = request.form.get("date_filter")

    conn = get_db_connection()
    cur = conn.cursor()
    if date_filter:
        cur.execute("DELETE FROM attendance WHERE class_name = ? AND DATE(timestamp) = ?",
                    (class_name, date_filter))
    else:
        cur.execute("DELETE FROM attendance WHERE class_name = ?", (class_name,))
    conn.commit()
    conn.close()

    flash("Attendance reset successfully", "success")
    return redirect(url_for("camera_page", class_name=class_name))


@app.route("/download")
@login_required
def download():
    class_filter = request.args.get("class_filter")
    date_filter = request.args.get("date_filter")
    conn = get_db_connection()
    cur = conn.cursor()
    q = """SELECT s.roll_number, s.name, a.class_name, a.timestamp
           FROM attendance a JOIN students s ON a.student_roll_number = s.roll_number"""
    cond, params = [], []
    if class_filter and class_filter != "all":
        cond.append("a.class_name=?")
        params.append(class_filter)
    if date_filter:
        cond.append("DATE(a.timestamp)=?")
        params.append(date_filter)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY a.timestamp DESC"
    rows = cur.execute(q, params).fetchall()
    conn.close()

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["Roll Number", "Name", "Class Name", "Timestamp"])
    for r in rows:
        w.writerow([r["roll_number"], r["name"], r["class_name"], r["timestamp"]])
    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")


@app.route("/summary")
@login_required
def summary():
    start = request.args.get("start")
    end = request.args.get("end")
    class_filter = request.args.get("class")

    conn = get_db_connection()
    cur = conn.cursor()

    q = """SELECT s.roll_number, s.name, a.class_name, DATE(a.timestamp) as day
           FROM attendance a JOIN students s ON s.roll_number = a.student_roll_number"""
    cond, params = [], []

    allowed_classes = [c["class_name"] for c in get_user_classes(current_user.id, current_user.role)]
    if current_user.role == "teacher":
        cond.append("a.class_name IN ({})".format(",".join(["?"]*len(allowed_classes))))
        params.extend(allowed_classes)

    if class_filter:
        cond.append("a.class_name=?")
        params.append(class_filter)
    if start:
        cond.append("DATE(a.timestamp)>=?")
        params.append(start)
    if end:
        cond.append("DATE(a.timestamp)<=?")
        params.append(end)
    if cond:
        q += " WHERE " + " AND ".join(cond)

    rows = cur.execute(q, params).fetchall()
    classes = allowed_classes  # dropdown for teacher or all for admin
    conn.close()

    counts = defaultdict(int)
    days_map = defaultdict(set)
    for r in rows:
        days_map[(r["roll_number"], r["name"])].add(r["day"])
    for k, v in days_map.items():
        counts[k] = len(v)

    return render_template("summary.html", counts=counts, start=start, end=end,
                           classes=classes, class_filter=class_filter)

# --------- Monthly summary helpers & routes (fixed indentation) ---------
def _query_monthly_summary(start_m: str | None, end_m: str | None, class_filter: str | None):
    """
    Returns list of dicts: {'month': 'YYYY-MM', 'class_name': '…', 'presents': int}
    """
    conn = get_db_connection()
    cur = conn.cursor()

    sql = """
      SELECT a.class_name,
             strftime('%Y-%m', a.timestamp) AS month,
             COUNT(DISTINCT a.student_roll_number) AS presents
      FROM attendance a
    """
    cond, params = [], []
    if class_filter:
        cond.append("a.class_name = ?")
        params.append(class_filter)
    if start_m:
        cond.append("strftime('%Y-%m', a.timestamp) >= ?")
        params.append(start_m)
    if end_m:
        cond.append("strftime('%Y-%m', a.timestamp) <= ?")
        params.append(end_m)
    if cond:
        sql += " WHERE " + " AND ".join(cond)
    sql += " GROUP BY a.class_name, strftime('%Y-%m', a.timestamp) ORDER BY month DESC, a.class_name ASC"

    rows = cur.execute(sql, params).fetchall()
    conn.close()
    return [{"class_name": r["class_name"], "month": r["month"], "presents": r["presents"]} for r in rows]


@app.route("/summary_report")
@login_required
def summary_report():
    # defaults: current month
    today_month = date.today().strftime("%Y-%m")
    start_m = request.args.get("start_m") or today_month
    end_m = request.args.get("end_m") or today_month
    class_filter = request.args.get("class") or ""

    # classes for filter dropdown
    conn = get_db_connection()
    classes = conn.execute("SELECT class_name FROM classes ORDER BY class_name").fetchall()
    conn.close()

    data = _query_monthly_summary(start_m, end_m, class_filter if class_filter else None)
    return render_template("summary_report.html",
                           rows=data, classes=classes,
                           start_m=start_m, end_m=end_m, class_filter=class_filter)


@app.route("/summary_report.csv")
@login_required
def summary_report_csv():
    start_m = request.args.get("start_m")
    end_m = request.args.get("end_m")
    class_filter = request.args.get("class")

    data = _query_monthly_summary(start_m, end_m, class_filter if class_filter else None)

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["Month", "Class", "Unique Students Present"])
    for r in data:
        w.writerow([r["month"], r["class_name"], r["presents"]])

    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    mem.seek(0)
    fname = f"attendance_summary_{start_m or 'all'}_{end_m or 'all'}.csv"
    return send_file(mem, as_attachment=True, download_name=fname, mimetype="text/csv")


@app.route("/download_excel")
@login_required
def download_excel():
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute("""SELECT s.roll_number, s.name, a.class_name, a.timestamp
                          FROM attendance a
                          JOIN students s ON a.student_roll_number = s.roll_number
                          ORDER BY a.timestamp DESC""").fetchall()
    conn.close()

    if not rows:
        flash("No records to export", "warning")
        return redirect(url_for("summary"))

    # Use pandas if available; otherwise fall back to CSV export
    if pd is None:
        output = io.StringIO()
        w = csv.writer(output)
        w.writerow(["Roll", "Name", "Class", "Timestamp"])
        for r in rows:
            w.writerow([r["roll_number"], r["name"], r["class_name"], r["timestamp"]])
        mem = io.BytesIO(output.getvalue().encode("utf-8"))
        mem.seek(0)
        return send_file(mem, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")

    df = pd.DataFrame(rows, columns=["Roll", "Name", "Class", "Timestamp"])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
    output.seek(0)

    return send_file(output,
                     as_attachment=True,
                     download_name="attendance.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/download_pdf")
@login_required
def download_pdf():
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute("""SELECT s.roll_number, s.name, a.class_name, a.timestamp
                          FROM attendance a
                          JOIN students s ON a.student_roll_number = s.roll_number
                          ORDER BY a.timestamp DESC""").fetchall()
    conn.close()

    if not rows:
        flash("No records to export", "warning")
        return redirect(url_for("summary"))

    if canvas is None:
        flash("PDF export requires reportlab to be installed", "warning")
        return redirect(url_for("summary"))

    output = io.BytesIO()
    c = canvas.Canvas(output, pagesize=A4)
    width, height = A4
    y = height - 40

    c.setFont("Helvetica-Bold", 14)
    c.drawString(200, y, "Attendance Report")
    y -= 40

    c.setFont("Helvetica", 10)
    for r in rows:
        line = f"{r['roll_number']} | {r['name']} | {r['class_name']} | {r['timestamp']}"
        c.drawString(40, y, line)
        y -= 20
        if y < 40:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 10)

    c.save()
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="attendance.pdf", mimetype="application/pdf")


@app.route("/rebuild_embeddings")
@login_required
def rebuild_embeddings():
    build_student_embeddings()
    return "Embeddings rebuilt", 200


# ---------- Main ----------
'''if __name__ == "__main__":
    # Ensure DB exists and has correct tables/defaults BEFORE building embeddings
    setup_database()
    build_student_embeddings()
    if not student_embeddings:
        print("[INFO] No embeddings yet. Upload or capture student images and visit /rebuild_embeddings")
    app.run(host="0.0.0.0", port=5001, debug=True)
'''

