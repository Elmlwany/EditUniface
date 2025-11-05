from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, Response, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from os.path import basename
import os
from datetime import datetime, timedelta
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from PIL import Image as PILImage
import cv2
import threading
import atexit
from flask_mail import Mail, Message
import logging

# Initialize Flask app

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your-secret-key-here")
app.jinja_env.filters['basename'] = basename

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'mustafamagdy2002@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'fczs xzgx xfmg ecby'  # Replace with the generated app password

mail = Mail(app)  # Initialize Mail after configuration

# Ensure mail.logger is not None before setting the logging level
if mail.logger:
    mail.logger.setLevel(logging.DEBUG)

# Base directory for the project (adjust if needed)
BASE_DIR = r"C:\SW\Camera attendence"

# Setup Flask-Login

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User database
def init_user_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                      ("admin", generate_password_hash("admin123")))
    conn.commit()
    conn.close()

# Initialize tracking database with sample data
def init_tracking_db():
    conn = sqlite3.connect("tracking.db")
    cursor = conn.cursor()
    
    # Check if 'floor' column exists and rename it to 'location'
    cursor.execute("PRAGMA table_info(logs)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'floor' in columns and 'location' not in columns:
        cursor.execute("ALTER TABLE logs RENAME COLUMN floor TO location")
    
    # Create logs table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            name TEXT,
            location TEXT,
            time TEXT,
            image_path TEXT
        )
    """)
    
    # Create excuses table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS excuses (
            name TEXT,
            date TEXT,
            hours INTEGER,
            reason TEXT,
            approved_by TEXT,
            created_at TEXT
        )
    """)
    
    # Update existing data to use new room labels
    cursor.execute("UPDATE logs SET location = 'Meeting Room' WHERE location = 'A'")
    cursor.execute("UPDATE logs SET location = 'Main Room' WHERE location = 'B' OR location = 'C' OR location LIKE 'Floor%'")
    cursor.execute("UPDATE logs SET location = 'Room 1' WHERE location = 'R1'")
    cursor.execute("UPDATE logs SET location = 'Room 2' WHERE location = 'R2'")
    cursor.execute("UPDATE logs SET location = 'Room 3' WHERE location = 'R3'")
    cursor.execute("UPDATE logs SET location = 'Room 4' WHERE location = 'R4'")
    cursor.execute("UPDATE logs SET location = 'Kitchen' WHERE location = 'K'")
    
    conn.commit()
    conn.close()

init_user_db()
init_tracking_db()

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1])
    return None

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("users.db")
        cursor = conn.execute("SELECT id, username, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            login_user(User(user[0], user[1]))
            return redirect(url_for("dashboard"))
        flash("Invalid username or password", "error")
    return render_template("login_2.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

@app.route("/home")
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/support")
def support():
    return render_template("support.html")

@app.route("/send_support_email", methods=["POST"])
def send_support_email():
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")

    # Compose the email
    msg = Message(
        subject="Support Request from Uniface360",
        sender=app.config['MAIL_USERNAME'],
        recipients=["mostafa.magdy@petrochoice.org"],  # Default recipient
        body=f"Name: {name}\nEmail: Admin@Uniface360\n\nMessage:\n{message}"
    )

    # Send the email
    try:
        mail.send(msg)
        flash("Your message has been sent successfully!", "success")
    except Exception as e:
        flash("Failed to send your message. Please try again later.", "error")
        print(f"Error: {e}")
    
    return redirect(url_for("support"))


@app.route("/request_demo", methods=["GET"])
def request_demo():
    return render_template("request_demo.html")


@app.route("/send_demo_request", methods=["POST"])
def send_demo_request():
    name = request.form.get("name")
    company = request.form.get("company")
    email = request.form.get("email", "Not provided")
    phone = request.form.get("phone", "Not provided")
    building_size = request.form.get("building_size", "Not provided")
    message = request.form.get("message", "")

    # Compose the email
    msg = Message(
        subject="Demo Request for Uniface360",
        sender=app.config['MAIL_USERNAME'],
        recipients=["mostafa.magdy@petrochoice.org"],  # Default recipient
        body=f"Demo Request Details:\n\nName: {name}\nCompany: {company}\nEmail: {email}\nPhone: {phone}\nBuilding Size: {building_size}\n\nAdditional Information:\n{message}"
    )

    # Send the email
    try:
        mail.send(msg)
        flash("Your demo request has been submitted successfully! Our team will contact you shortly.", "success")
    except Exception as e:
        flash("Failed to submit your demo request. Please try again later.", "danger")
        print(f"Error: {e}")
    
    return redirect(url_for("request_demo"))

@app.route("/dashboard")
@login_required
def dashboard():
    conn = sqlite3.connect("tracking.db")
    # Location data: Latest known location for each person
    cursor = conn.execute("""
        SELECT l.location, COUNT(DISTINCT l.name) as count, GROUP_CONCAT(l.name) as names
        FROM logs l
        INNER JOIN (
            SELECT name, MAX(time) as max_time
            FROM logs
            WHERE name != 'Unknown'
            GROUP BY name
        ) latest ON l.name = latest.name AND l.time = latest.max_time
        GROUP BY l.location
    """)
    location_data = {row[0]: {"count": row[1], "names": row[2].split(",") if row[2] else []} for row in cursor}
    
    # Get the latest location for each person (for the map)
    cursor = conn.execute("""
        SELECT name, location, MAX(time) as last_seen
        FROM logs
        WHERE name != 'Unknown'
        GROUP BY name
    """)
    people = [{"name": row[0], "location": row[1], "last_seen": row[2]} for row in cursor]
    
    # Attendance analysis: Analyze the most recent day for each person
    cursor = conn.execute("""
        SELECT name, date, MIN(time) as first_seen, MAX(time) as last_seen
        FROM (
            SELECT name, time, SUBSTR(time, 1, 10) as date
            FROM logs
            WHERE name != 'Unknown'
        ) sub
        GROUP BY name, date
        HAVING date = (
            SELECT SUBSTR(MAX(time), 1, 10)
            FROM logs l2
            WHERE l2.name = sub.name
        )
    """)
    attendance = []
    for row in cursor:
        name, date, first, last = row
        first_time = datetime.strptime(first, "%Y-%m-%d %H:%M:%S")
        last_time = datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
        late = first_time.hour > 8 or (first_time.hour == 8 and first_time.minute > 0)
        early_leave = last_time.hour < 16
        attendance.append({
            "name": name,
            "date": date,
            "first_seen": first,
            "last_seen": last,
            "late": late,
            "early_leave": early_leave
        })
    
    # Total detections per person
    cursor = conn.execute("SELECT name, COUNT(*) as count FROM logs WHERE name != 'Unknown' GROUP BY name")
    person_counts = {row[0]: row[1] for row in cursor}
    # Detections per location
    cursor = conn.execute("SELECT location, COUNT(*) as count FROM logs WHERE name != 'Unknown' GROUP BY location")
    location_counts = {row[0]: row[1] for row in cursor}
    # Recent activity
    cursor = conn.execute("SELECT name, location, time FROM logs WHERE name != 'Unknown' ORDER BY time DESC LIMIT 5")
    recent_logs = [{"name": row[0], "location": row[1], "time": row[2]} for row in cursor]
    conn.close()
    return render_template("dashboard_2.html", 
                         floor_data=location_data, 
                         attendance=attendance,
                         person_counts=person_counts, 
                         floor_counts=location_counts, 
                         recent_logs=recent_logs,
                         people=people)

@app.route("/report")
@login_required
def emergency_status():
    conn = sqlite3.connect("tracking.db")
    cursor = conn.execute("SELECT DISTINCT name FROM logs WHERE name != 'Unknown'")
    names = [row[0] for row in cursor]
    cursor = conn.execute("SELECT DISTINCT location FROM logs")
    locations = [row[0] for row in cursor]
    cursor = conn.execute("SELECT name, location, MAX(time) as last_seen, image_path FROM logs WHERE name != 'Unknown' GROUP BY name")
    people = [{"name": row[0], "location": row[1], "last_seen": row[2], "image_path": row[3]} for row in cursor]
    conn.close()
    return render_template("status_2.html", people=people, names=names, floors=locations)

@app.route("/hr")
@login_required
def hr_dashboard():
    conn = sqlite3.connect("tracking.db")
    cursor = conn.execute("SELECT DISTINCT name FROM logs WHERE name != 'Unknown'")
    employees = [row[0] for row in cursor]
    conn.close()
    return render_template("hr_dashboard.html", employees=employees)

@app.route("/hr/print/<employee>")
@login_required
def hr_print_attendance(employee):
    # Dummy data for print until DB integration
    today = datetime.now().strftime("%Y-%m-%d")
    monthly_stats = {
        "working_days": 20,
        "present": 18,
        "absent": 1,
        "vacation": 0,
        "sick": 0,
        "official_leave": 1,
        "late_days": 0,
        "overtime_hours": 5.58,
        "attendance_percent": 90
    }
    ytd_stats = monthly_stats.copy()
    overtime_by_month = {m: 0 for m in range(1, 13)}
    overtime_by_month[11] = 5.58
    # Sample daily rows
    daily_rows = [
        {"date": today, "day": "Monday", "time_in": "08:10", "time_out": "17:00", "total": 8.83, "status": "Present", "late": "No", "ot": 0.83, "visits": 0.00},
        {"date": (datetime.now()-timedelta(days=1)).strftime("%Y-%m-%d"), "day": "Sunday", "time_in": "08:05", "time_out": "17:10", "total": 9.08, "status": "Present", "late": "No", "ot": 0.08, "visits": 0.00},
    ]
    return render_template(
        "hr_attendance_print.html",
        employee=employee,
        monthly=monthly_stats,
        ytd=ytd_stats,
        overtime_by_month=overtime_by_month,
        daily_rows=daily_rows,
    )

@app.route("/delete_report_rows", methods=["POST"])
@login_required
def delete_report_rows():
    names = request.get_json()
    if not names:
        return Response("No rows selected", status=400)

    conn = sqlite3.connect("tracking.db")
    cursor = conn.cursor()
    for name in names:
        cursor.execute("DELETE FROM logs WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    return Response("Rows deleted", status=200)

@app.route("/export_report", methods=["POST"])
@login_required
def export_report():
    # Get format and filtered data
    export_format = request.args.get("format", "csv")
    filtered_data = request.get_json()
    if not filtered_data:
        return Response("No data to export", status=400)

    if export_format == "csv":
        # CSV Export
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=["name", "location", "last_seen", "image_path"])
        writer.writeheader()
        for row in filtered_data:
            writer.writerow({
                "name": row["name"],
                "location": row["location"],
                "last_seen": row["last_seen"],
                "image_path": row["image_path"] if row["image_path"] else "No Image"
            })
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=emergency_status_report.csv"}
        )
    else:
        # PDF Export
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Title
        styles = getSampleStyleSheet()
        elements.append(Paragraph("Emergency Status Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # Table data
        data = [["Name", "Last Known Location", "Last Seen", "Evidence"]]
        for row in filtered_data:
            image_path = row["image_path"] if row["image_path"] and row["image_path"] != "No Image" else None
            image_cell = "No Image"
            if image_path:
                try:
                    # Construct absolute path
                    abs_path = os.path.join(BASE_DIR, image_path.replace('\\', '/'))
                    if os.path.exists(abs_path):
                        # Open and resize image
                        pil_img = PILImage.open(abs_path)
                        img_width, img_height = pil_img.size
                        # Scale to 50 points wide, maintaining aspect ratio
                        scale = 50 / img_width
                        img_width, img_height = int(img_width * scale), int(img_height * scale)
                        img = Image(abs_path, width=img_width, height=img_height)
                        image_cell = img
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    image_cell = "No Image"

            data.append([
                row["name"],
                row["location"],
                row["last_seen"],
                image_cell
            ])

        # Create table with adjusted column widths
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 2*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)

        # Build PDF
        doc.build(elements)
        return Response(
            buffer.getvalue(),
            mimetype="application/pdf",
            headers={"Content-Disposition": "attachment;filename=emergency_status_report.pdf"}
        )

@app.route("/logs/<name>")
@login_required
def person_logs(name):
    conn = sqlite3.connect("tracking.db")
    cursor = conn.execute("SELECT name, location, time, image_path FROM logs WHERE name = ? ORDER BY time DESC", (name,))
    logs = [{"name": row[0], "location": row[1], "time": row[2], "image_path": row[3]} for row in cursor]
    conn.close()
    return render_template("person_logs_2.html", name=name, logs=logs)

@app.route("/delete_log_rows/<name>", methods=["POST"])
@login_required
def delete_log_rows(name):
    entries = request.get_json()
    if not entries:
        return Response("No rows selected", status=400)

    conn = sqlite3.connect("tracking.db")
    cursor = conn.cursor()
    for entry in entries:
        cursor.execute("DELETE FROM logs WHERE name = ? AND location = ? AND time = ?",
                       (entry["name"], entry["location"], entry["time"]))
    conn.commit()
    conn.close()
    return Response("Rows deleted", status=200)

@app.route("/export_logs/<name>")
@login_required
def export_logs(name):
    export_format = request.args.get("format", "csv")
    conn = sqlite3.connect("tracking.db")
    cursor = conn.execute("SELECT name, location, time, image_path FROM logs WHERE name = ? ORDER BY time DESC", (name,))
    logs = [{"name": row[0], "location": row[1], "time": row[2], "image_path": row[3] if row[3] else "No Image"} for row in cursor]
    conn.close()

    if export_format == "csv":
        # CSV Export
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=["name", "location", "time", "image_path"])
        writer.writeheader()
        writer.writerows(logs)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename={name}_logs.csv"}
        )
    else:
        # PDF Export
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Title
        styles = getSampleStyleSheet()
        elements.append(Paragraph(f"Logs for {name}", styles['Title']))
        elements.append(Spacer(1, 12))

        # Table data
        data = [["Name", "Location", "Time", "Evidence"]]
        for log in logs:
            image_path = log["image_path"] if log["image_path"] and log["image_path"] != "No Image" else None
            image_cell = "No Image"
            if image_path:
                try:
                    # Construct absolute path
                    abs_path = os.path.join(BASE_DIR, image_path.replace('\\', '/'))
                    if os.path.exists(abs_path):
                        # Open and resize image
                        pil_img = PILImage.open(abs_path)
                        img_width, img_height = pil_img.size
                        # Scale to 50 points wide, maintaining aspect ratio
                        scale = 50 / img_width
                        img_width, img_height = int(img_width * scale), int(img_height * scale)
                        img = Image(abs_path, width=img_width, height=img_height)
                        image_cell = img
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    image_cell = "No Image"

            data.append([
                log["name"],
                log["location"],
                log["time"],
                image_cell
            ])

        # Create table with adjusted column widths
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 2*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)

        # Build PDF
        doc.build(elements)
        return Response(
            buffer.getvalue(),
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment;filename={name}_logs.pdf"}
        )

@app.route("/map")
@login_required
def map():
    conn = sqlite3.connect("tracking.db")
    # Get the latest location for each person
    cursor = conn.execute("""
        SELECT name, location, MAX(time) as last_seen
        FROM logs
        WHERE name != 'Unknown'
        GROUP BY name
    """)
    people = [{"name": row[0], "location": row[1], "last_seen": row[2]} for row in cursor]
    conn.close()
    return render_template("map.html", people=people)

@app.route("/live_stream")
@login_required
def live_stream():
    return render_template("live_stream.html")

# Dictionary to store video capture objects and locks for each camera
video_streams = {}
locks = {}

# Function to initialize video capture for each camera
def init_video_stream(camera_id):
    if camera_id not in video_streams:
        video_streams[camera_id] = cv2.VideoCapture(camera_id - 1)  # Map camera_id to source (0, 1, 2)
        locks[camera_id] = threading.Lock()

@app.route("/video_feed/<int:camera_id>")
@login_required
def video_feed(camera_id):
    def generate(camera_id):
        init_video_stream(camera_id)
        cap = video_streams[camera_id]
        lock = locks[camera_id]
        while True:
            with lock:
                success, frame = cap.read()
                if not success:
                    break
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# Cleanup function to release video streams on shutdown
def cleanup_video_streams():
    for cap in video_streams.values():
        cap.release()

atexit.register(cleanup_video_streams)

@app.route("/evidence/<path:filename>")
@login_required
def serve_evidence(filename):
    return send_from_directory("evidence", filename)

@app.route("/people_directory")
@login_required
def people_directory():
    conn = sqlite3.connect("tracking.db")
    
    # Get filter parameters
    selected_date = request.args.get("date")
    selected_period = request.args.get("period", "day")  # default to "day"

    # SQL date filtering logic
    date_filter = ""
    params = []

    if selected_date:
        if selected_period == "day":
            date_filter = "AND DATE(time) = ?"
            params = [selected_date]
        elif selected_period == "month":
            date_filter = "AND strftime('%Y-%m', time) = ?"
            params = [selected_date]
        elif selected_period == "quarter":
            if "-Q" in selected_date:
                year, q = selected_date.split("-Q")
                quarter_months = {
                    "1": ["01", "02", "03"],
                    "2": ["04", "05", "06"],
                    "3": ["07", "08", "09"],
                    "4": ["10", "11", "12"]
                }.get(q, [])
                if quarter_months:
                    placeholders = ','.join('?' * len(quarter_months))
                    date_filter = f"AND strftime('%Y', time) = ? AND strftime('%m', time) IN ({placeholders})"
                    params = [year] + quarter_months

    # Get all unique people
    cursor = conn.execute("SELECT DISTINCT name FROM logs WHERE name != 'Unknown'")
    people = []

    # Photo map
    photo_map = {
        "Abdelrahman_image": "abdelrahman.png",
        "Eng.mahmoud": "Eng.mahmoud.png",
        "Mahmoud_Ahmed": "Mahmoud_Ahmed.png",
        "Mostafa": "Mostafa-2.png",
        "mohamed_ragab": "Ragab.png",
        "yousef": "yousef.png",
        "Dalia": "dalia.PNG",
        "Hagar": "hagar.jpeg",
        "Gamila": "Gamila.jpg"
    }

    roles = {
        "Eng.mahmoud": "Office Manager and Drilling Engineer",
        "Dalia": "HR Specialist",
        "Mostafa": "AI Engineer and Head of Software Team",
        "mohamed_ragab": "Software Team",
        "Abdelrahman_image": "Software Team",
        "Mahmoud_Ahmed": "Software Team",
        "Gamila": "Office Girl",
        "yousef": "Finance",
        "Hagar": "Employee"
    }

    for row in cursor:
        name = row[0]
        image = f"formal photos/{photo_map.get(name, 'default.jpg')}"
        role = roles.get(name, "Employee")

        # Calculate filtered hours
        cursor2 = conn.execute(f"""
            SELECT COUNT(DISTINCT SUBSTR(time, 1, 10)) as days_present,
                   AVG(CAST(SUBSTR(time, 12, 2) AS INTEGER)) as avg_hours,
                   COUNT(*) as total_entries
            FROM logs
            WHERE name = ? {date_filter}
        """, (name, *params))
        stats = cursor2.fetchone()
        days_present = stats[0] or 0
        avg_hours = round(stats[1], 1) if stats[1] else 0
        total_entries = stats[2] or 0

        attendance_rate = round((days_present / 20) * 100) if days_present else 0

        # Excuse hours
        if selected_date:
            if selected_period == "day":
                cursor3 = conn.execute(
                    "SELECT SUM(hours) FROM excuses WHERE name = ? AND date = ?",
                    (name, selected_date)
                )
            elif selected_period == "month":
                cursor3 = conn.execute(
                    "SELECT SUM(hours) FROM excuses WHERE name = ? AND strftime('%Y', date) = ?",
                    (name, selected_date)
                )
            elif selected_period == "quarter" and "-Q" in selected_date:
                year, q = selected_date.split("-Q")
                months = {
                    "1": ["01", "02", "03"],
                    "2": ["04", "05", "06"],
                    "3": ["07", "08", "09"],
                    "4": ["10", "11", "12"]
                }.get(q, [])
                placeholders = ','.join('?' * len(months))
                cursor3 = conn.execute(
                    f"SELECT SUM(hours) FROM excuses WHERE name = ? AND strftime('%Y', date) = ? AND strftime('%m', date) IN ({placeholders})",
                    (name, year, *months)
                )
            else:
                cursor3 = conn.execute("SELECT SUM(hours) FROM excuses WHERE name = ?", (name,))
        else:
            cursor3 = conn.execute("SELECT SUM(hours) FROM excuses WHERE name = ?", (name,))

        excuse_hours = cursor3.fetchone()[0] or 0

        # Estimated hours from total entries
        estimated_hours = round(total_entries / 2.0, 1)  # adjust logic as needed

        current_hours = estimated_hours + excuse_hours

        people.append({
            "id": name,
            "name": name,
            "image": image,
            "role": role,
            "attendance_rate": attendance_rate,
            "days_present": days_present,
            "average_hours": avg_hours,
            "current_hours": current_hours
        })

    conn.close()
    return render_template("people_directory.html", people=people, selected_date=selected_date, selected_period=selected_period)

@app.route("/generate_report/<person_id>")
@login_required
def generate_report(person_id):
    conn = sqlite3.connect("tracking.db")

    # Get person's basic info
    cursor = conn.execute("""
        SELECT name, image_path, MAX(time) as last_seen
        FROM logs
        WHERE name = ? AND name != 'Unknown'
        GROUP BY name
    """, (person_id,))
    person = cursor.fetchone()

    if not person:
        conn.close()
        return "Person not found", 404

    name, image_path, last_seen = person

    # Get attendance statistics
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT SUBSTR(time, 1, 10)) as days_present,
               AVG(CAST(SUBSTR(time, 12, 2) AS INTEGER)) as avg_hours,
               MIN(time) as first_seen,
               MAX(time) as last_seen,
               COUNT(*) as total_entries
        FROM logs
        WHERE name = ?
    """, (person_id,))
    stats = cursor.fetchone()
    days_present, avg_hours, first_seen, last_seen, total_entries = stats

    # Get most visited locations
    cursor = conn.execute("""
        SELECT location, COUNT(*) as count
        FROM logs
        WHERE name = ?
        GROUP BY location
        ORDER BY count DESC
        LIMIT 3
    """, (person_id,))
    top_locations = cursor.fetchall()

    # Get daily hours distribution
    cursor = conn.execute("""
        SELECT SUBSTR(time, 1, 10) as date,
               COUNT(*) as entries,
               MAX(CAST(SUBSTR(time, 12, 2) AS INTEGER)) - MIN(CAST(SUBSTR(time, 12, 2) AS INTEGER)) as hours
        FROM logs
        WHERE name = ?
        GROUP BY date
        ORDER BY date DESC
        LIMIT 7
    """, (person_id,))
    daily_hours = cursor.fetchall()

    # Get time distribution
    cursor = conn.execute("""
        SELECT 
            CASE 
                WHEN CAST(SUBSTR(time, 12, 2) AS INTEGER) BETWEEN 6 AND 11 THEN 'Morning'
                WHEN CAST(SUBSTR(time, 12, 2) AS INTEGER) BETWEEN 12 AND 17 THEN 'Afternoon'
                ELSE 'Evening'
            END as time_period,
            COUNT(*) as count
        FROM logs
        WHERE name = ?
        GROUP BY time_period
    """, (person_id,))
    time_distribution = cursor.fetchall()

    # Add role information
    roles = {
        "Eng. Mahmoud": "Office Manager and Drilling Engineer",
        "Dalia": "HR Specialist",
        "Mostafa": "AI Engineer and Head of Software Team",
        "Ragab": "Software Team",
        "Abdelrahman": "Software Team",
        "Mahmoud Ahmed": "Software Team",
        "Gamila": "Office Girl",
        "Yousef": "Finance"
    }
    role = roles.get(name, "Unknown Role")

    conn.close()

    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title = Paragraph(f"Person Report: {name}", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Header Section
    header_data = []
    header_data.append([
        Image(os.path.join('Formal photos', f"{name}.jpg"), width=100, height=100),
        [
            Paragraph(f"<b>Name:</b> {name}", styles['Normal']),
            Paragraph(f"<b>Role:</b> {role}", styles['Normal']),
            Paragraph(f"<b>Issue's Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
            Paragraph(f"<b>First Seen:</b> {first_seen}", styles['Normal']),
            Paragraph(f"<b>Last Seen:</b> {last_seen}", styles['Normal']),
            Paragraph(f"<b>Total Entries:</b> {total_entries}", styles['Normal'])
        ]
    ])
    header_table = Table(header_data, colWidths=[120, 400])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 10)
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 20))

    # Attendance Statistics
    elements.append(Paragraph("Attendance Statistics", styles['Heading2']))
    elements.append(Spacer(1, 10))

    stats_data = [
        ['Days Present', 'Average Hours', 'Total Entries'],
        [str(days_present), f"{avg_hours:.1f}", str(total_entries)]
    ]
    stats_table = Table(stats_data, colWidths=[200, 200, 200])
    stats_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 20))

    # Location Analytics
    elements.append(Paragraph("Location Analytics", styles['Heading2']))
    elements.append(Spacer(1, 10))

    location_data = [['Location', 'Visits']]
    for loc, count in top_locations:
        location_data.append([loc, str(count)])

    location_table = Table(location_data, colWidths=[300, 100])
    location_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(location_table)
    elements.append(Spacer(1, 20))

    # Time Distribution
    elements.append(Paragraph("Time Distribution", styles['Heading2']))
    elements.append(Spacer(1, 10))

    time_data = [['Time Period', 'Entries']]
    for period, count in time_distribution:
        time_data.append([period, str(count)])

    time_table = Table(time_data, colWidths=[200, 200])
    time_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(time_table)
    elements.append(Spacer(1, 20))

    # Recent Activity
    elements.append(Paragraph("Recent Activity (Last 7 Days)", styles['Heading2']))
    elements.append(Spacer(1, 10))

    activity_data = [['Date', 'Entries', 'Hours Present']]
    for date, entries, hours in daily_hours:
        activity_data.append([date, str(entries), f"{hours:.1f}"])

    activity_table = Table(activity_data, colWidths=[150, 150, 150])
    activity_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(activity_table)

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{name}_report.pdf",
        mimetype='application/pdf'
    )

@app.route("/formal_photos/<path:filename>")
@login_required
def serve_formal_photo(filename):
    return send_from_directory('Formal photos', filename)

@app.route("/excuses", methods=["GET", "POST"])
@login_required
def excuses():
    conn = sqlite3.connect("tracking.db")
    cursor = conn.execute("SELECT DISTINCT name FROM logs WHERE name != 'Unknown'")
    employees = [row[0] for row in cursor]

    if request.method == "POST":
        name = request.form["name"]
        date = request.form["date"]
        hours = request.form["hours"]
        reason = request.form["reason"]
        approved_by = current_user.username
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn.execute(
            "INSERT INTO excuses (name, date, hours, reason, approved_by, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (name, date, hours, reason, approved_by, created_at)
        )
        conn.commit()

    cursor = conn.execute("SELECT rowid as id, name, date, hours, reason, approved_by, created_at FROM excuses")
    excuses = [
        {
            "id": row[0],
            "name": row[1],
            "date": row[2],
            "hours": row[3],
            "reason": row[4],
            "approved_by": row[5],
            "created_at": row[6],
        }
        for row in cursor
    ]
    conn.close()
    return render_template("excuses.html", employees=employees, excuses=excuses)

@app.route("/add_excuse", methods=["POST"])
@login_required
def add_excuse():
    conn = sqlite3.connect("tracking.db")
    name = request.form["name"]
    date = request.form["date"]
    hours = request.form["hours"]
    reason = request.form["reason"]
    approved_by = current_user.username
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute(
        "INSERT INTO excuses (name, date, hours, reason, approved_by, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (name, date, hours, reason, approved_by, created_at)
    )
    conn.commit()
    conn.close()
    return redirect(url_for("excuses"))

@app.route("/edit_excuse/<int:id>", methods=["GET", "POST"])
@login_required
def edit_excuse(id):
    conn = sqlite3.connect("tracking.db")
    if request.method == "POST":
        name = request.form["name"]
        date = request.form["date"]
        hours = request.form["hours"]
        reason = request.form["reason"]
        conn.execute(
            "UPDATE excuses SET name = ?, date = ?, hours = ?, reason = ? WHERE rowid = ?",
            (name, date, hours, reason, id)
        )
        conn.commit()
        conn.close()
        return redirect(url_for("excuses"))

    cursor = conn.execute("SELECT name, date, hours, reason FROM excuses WHERE rowid = ?", (id,))
    excuse = cursor.fetchone()
    conn.close()
    if not excuse:
        return "Excuse not found", 404

    return render_template("edit_excuse.html", excuse={
        "id": id,
        "name": excuse[0],
        "date": excuse[1],
        "hours": excuse[2],
        "reason": excuse[3]
    })

@app.route("/delete_excuse/<int:id>", methods=["POST"])
@login_required
def delete_excuse(id):
    conn = sqlite3.connect("tracking.db")
    conn.execute("DELETE FROM excuses WHERE rowid = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for("excuses"))

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404_2.html"), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)