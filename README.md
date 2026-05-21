# Face Recognition Attendance System

A Flask-based attendance management system that uses **face recognition** and **geofencing** to automate classroom attendance, originally built for **Netaji Subhas University of Technology (NSUT)**.

Teachers and admins manage classes, timetables, and reports; students mark attendance by scanning their face from the classroom; the system blocks attempts from outside a configurable radius and enforces the timetable.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Flask](https://img.shields.io/badge/flask-3.x-black) ![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

### Attendance
- **Face recognition** via DeepFace (Facenet512 embeddings) with live webcam capture
- **Geofencing** — attendance only accepted within a configurable radius of the classroom (haversine distance)
- **Timetable enforcement** — students can only mark attendance during scheduled class slots
- **Manual override** for teachers/admins
- **Edit by date** — adjust historical attendance per class/date
- **Special session statuses** — mark days as Mass Bunk, Holiday, Class Rescheduled, Teacher on Leave, Gazette Holiday

### Users & Roles
- **Three roles**: Admin, Teacher, Student
- **Admin**: full access — manage users, classes, timetable, assignments, leave approval
- **Teacher**: view/mark attendance for assigned classes, approve manual entries
- **Student**: view own attendance, request leaves, download report card

### Leave Management
- Students submit leave requests with date range, reason, and supporting document
- Admin approval workflow (Pending / Approved / Rejected)
- Approved leaves automatically reflected as "On Leave" in attendance history (not counted as absent)

### Reports & Analytics
- **PDF report cards** with watermark, NSUT logo, attendance pie chart, per-subject breakdown, and overall grade (A+ to F)
- **Excel / CSV exports** of attendance data
- **Calendar view** for teachers — past sessions, scheduled classes, and special statuses color-coded
- **Risk prediction** — linear regression projects end-of-semester attendance percentage to flag at-risk students

### Bulk Operations
- **Zip upload** for adding many students at once — flexible filename parsing (`Name_RollNumber.jpg` or `RollNumber_Name.jpg`)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask, Flask-Login |
| Face recognition | DeepFace (Facenet512), OpenCV, dlib |
| Database | SQLite (via `sqlite3`) |
| ML | scikit-learn (LinearRegression), NumPy |
| Reports | ReportLab (PDF), Matplotlib (charts), openpyxl (Excel) |
| Frontend | Jinja2 templates, Bootstrap, FullCalendar |
| Auth | Werkzeug password hashing |

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- A working webcam (for face capture)
- `cmake` and a C++ compiler (required by `dlib`)
  - **macOS**: `brew install cmake`
  - **Ubuntu/Debian**: `sudo apt install cmake build-essential libgl1`
  - **Windows**: install Visual Studio Build Tools + CMake

### Setup

```bash
# 1. Clone
git clone https://github.com/harshgoyal27/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dlib facial landmarks model
# (~100MB — not included in repo)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# 5. Configure environment (see Configuration below)
cp .env.example .env
# edit .env with your values

# 6. Run
python app.py
```

The app will start at **http://localhost:5000**. The SQLite database (`attendance.db`) and required tables are created automatically on first run.

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Required
SECRET_KEY=change-me-to-a-long-random-string

# Geofencing — replace with your classroom's coordinates
CLASSROOM_LAT=28.655983
CLASSROOM_LON=77.291283
MAX_DISTANCE_KM=0.2

# Optional
TOTAL_SEMESTER_CLASSES=40
```

> ⚠️ Currently these are hardcoded in `app.py`. Move them to env vars before deploying.

---

## 🔐 Default Credentials

On first run, an admin account is created:

| Username | Password |
|---|---|
| `admin` | `admin123` |

**Change this immediately after first login.**

Students created via bulk upload get their **roll number** as the default password — they should be required to change it on first login.

---

## 🚀 Usage

### As Admin
1. Log in at `/login` with admin credentials
2. **Create classes** at `/classes` (with branch + year)
3. **Add timetable slots** for each class (day + start/end time)
4. **Create teacher accounts** at `/admin`
5. **Assign classes to teachers** at `/admin/assign_class`
6. **Bulk-upload students** at `/students` (upload a `.zip` of face images named `RollNumber_Name.jpg`)
7. **Review and approve leave requests** at `/admin/leaves`

### As Teacher
1. Log in at `/login`
2. View attendance dashboard at `/`
3. Mark attendance manually at `/attendance/manual` or edit existing records at `/attendance/edit`
4. View calendar of past + scheduled sessions at `/teacher/history`

### As Student
1. Log in at `/student/login` (roll number + password)
2. View personal attendance + risk prediction at `/student/dashboard`
3. **Mark attendance** — webcam scan inside the configured geofence during a scheduled class
4. Submit leave requests at `/student/leave`
5. Download PDF report card from the dashboard

---

## 📁 Project Structure

```
Face-Recognition-Attendance-System/
├── app.py                    # Main Flask app (all routes)
├── dashboard.py              # Dashboard helpers
├── requirements.txt
├── attendance.db             # SQLite DB (auto-created, gitignored)
├── shape_predictor_68_face_landmarks.dat   # dlib model (downloaded)
├── templates/                # Jinja2 templates
├── static/
│   ├── students/             # Student face images
│   ├── leave_docs/           # Uploaded leave documents
│   └── nsut_logo.png
└── database/                 # DB helpers / migrations
```

---

## 🗺️ Key Routes

| Route | Method | Role | Purpose |
|---|---|---|---|
| `/login` | GET/POST | All | Admin/teacher login |
| `/student/login` | GET/POST | Student | Student login |
| `/` | GET | Admin/Teacher | Attendance dashboard |
| `/students` | GET/POST | Admin | Manage students |
| `/students/upload` | POST | Admin | Bulk upload via zip |
| `/classes` | GET/POST | Admin | Manage classes + timetable |
| `/recognize` | POST | Student | Face recognition endpoint |
| `/attendance/manual` | GET/POST | Teacher/Admin | Manual attendance |
| `/attendance/edit` | GET | Teacher/Admin | Edit past sessions |
| `/student/dashboard` | GET | Student | Personal dashboard |
| `/student/leave` | GET/POST | Student | Leave requests |
| `/admin/leaves` | GET/POST | Admin | Approve/reject leaves |
| `/student/<roll>/report_card` | GET | Student/Admin | Download PDF report |
| `/teacher/history` | GET | Teacher/Admin | Calendar view |

---

## 🧰 Troubleshooting

**`dlib` won't install**
Install CMake and a C++ compiler first. On Mac: `brew install cmake`. On Ubuntu: `sudo apt install cmake build-essential`.

**Webcam doesn't work in the browser**
Modern browsers require HTTPS for `getUserMedia`. For local development, use `localhost` (which is allowed) or `flask run --cert=adhoc`.

**Face recognition is too lenient / too strict**
Adjust the threshold in `recognize_face()` (currently `0.35`). Higher = stricter. Facenet512 cosine similarity typically performs well in the 0.4–0.5 range.

**Geofence blocks me even when I'm in the classroom**
Check your `CLASSROOM_LAT` / `CLASSROOM_LON` values, and increase `MAX_DISTANCE_KM` if needed. GPS accuracy on laptops can be poor.

**First face recognition request is very slow**
DeepFace downloads model weights on first use (~100MB). Subsequent calls are fast.

---

## 🛣️ Roadmap

- [ ] Refactor `app.py` into Flask Blueprints
- [ ] Migrate from raw SQLite to SQLAlchemy + Alembic migrations
- [ ] Add CSRF protection (Flask-WTF) and rate limiting (Flask-Limiter)
- [ ] Liveness detection (blink / head turn) to prevent photo spoofing
- [ ] Persist face embeddings to disk for multi-worker deployments
- [ ] Docker support
- [ ] Unit + integration tests (pytest)
- [ ] Force password change on first login

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repo
2. Create a branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request

For major changes, open an issue first to discuss.

---

## 📄 License

This project is licensed under the MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

- [DeepFace](https://github.com/serengil/deepface) for the face recognition pipeline
- [dlib](http://dlib.net/) for facial landmark detection
- [Flask](https://flask.palletsprojects.com/) and the Python ecosystem
- Netaji Subhas University of Technology (NSUT)

---

**Author**: [@harshgoyal27](https://github.com/harshgoyal27)
