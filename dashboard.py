# dashboard.py (With Roll Number ID)
from flask import Flask, render_template, Response, request
import sqlite3; import csv; import io

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('attendance.db'); conn.row_factory = sqlite3.Row; return conn

@app.route('/')
def index():
    conn = get_db_connection()
    selected_class = request.args.get('class_filter'); selected_date = request.args.get('date_filter')
    classes = conn.execute("SELECT DISTINCT class_name FROM attendance ORDER BY class_name").fetchall()
    
    query = """
        SELECT s.roll_number, s.name, a.class_name, a.timestamp
        FROM attendance a
        JOIN students s ON a.student_roll_number = s.roll_number
    """
    conditions = []; params = []
    if selected_class and selected_class != 'all': conditions.append("a.class_name = ?"); params.append(selected_class)
    if selected_date: conditions.append("DATE(a.timestamp) = ?"); params.append(selected_date)
    if conditions: query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY a.timestamp DESC"
    
    records = conn.execute(query, params).fetchall(); conn.close()
    return render_template('index.html', records=records, classes=classes, selected_class=selected_class, selected_date=selected_date)

@app.route('/download')
def download():
    conn = get_db_connection()
    selected_class = request.args.get('class_filter'); selected_date = request.args.get('date_filter')
    
    query = """
        SELECT s.roll_number, s.name, a.class_name, a.timestamp
        FROM attendance a
        JOIN students s ON a.student_roll_number = s.roll_number
    """
    conditions = []; params = []
    if selected_class and selected_class != 'all': conditions.append("a.class_name = ?"); params.append(selected_class)
    if selected_date: conditions.append("DATE(a.timestamp) = ?"); params.append(selected_date)
    if conditions: query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY a.timestamp ASC"
    all_records = conn.execute(query, params).fetchall(); conn.close()

    output = io.StringIO(); writer = csv.writer(output)
    writer.writerow(['Roll Number', 'Name', 'Class', 'Timestamp'])
    writer.writerows([tuple(row) for row in all_records]); output.seek(0)
    
    filename = f"attendance_report_{selected_date or 'all-dates'}.csv"
    return Response(output, mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename={filename}"})

if __name__ == '__main__':
    app.run(debug=True)
