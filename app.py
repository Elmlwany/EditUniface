# from flask import Flask, render_template
# import sqlite3

# app = Flask(__name__)

# @app.route("/")
# def emergency_status():
#     conn = sqlite3.connect("tracking.db")
#     cursor = conn.execute("SELECT name, floor, MAX(time) as last_seen FROM logs GROUP BY name")
#     people = [{"name": row[0], "floor": row[1], "last_seen": row[2], "status": "Present"} for row in cursor]
#     conn.close()
#     return render_template("status.html", people=people)

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)






# version 2


# from flask import Flask, render_template, send_from_directory
# import sqlite3
# from os.path import basename

# app = Flask(__name__)
# app.jinja_env.filters['basename'] = basename

# @app.route("/")
# def emergency_status():
#     conn = sqlite3.connect("tracking.db")
#     cursor = conn.execute("SELECT name, floor, MAX(time) as last_seen, image_path FROM logs GROUP BY name")
#     people = [{"name": row[0], "floor": row[1], "last_seen": row[2], "image_path": row[3]} for row in cursor]
#     conn.close()
#     return render_template("status.html", people=people)

# @app.route("/evidence/<path:filename>")
# def serve_evidence(filename):
#     return send_from_directory("evidence", filename)

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)




from flask import Flask, render_template, send_from_directory
import sqlite3
from os.path import basename

app = Flask(__name__)
app.jinja_env.filters['basename'] = basename

@app.route("/")
def emergency_status():
    conn = sqlite3.connect("tracking.db")
    cursor = conn.execute("SELECT name, floor, MAX(time) as last_seen, image_path FROM logs GROUP BY name")
    people = [{"name": row[0], "floor": row[1], "last_seen": row[2], "image_path": row[3]} for row in cursor]
    conn.close()
    return render_template("status.html", people=people)

@app.route("/logs/<name>")
def person_logs(name):
    conn = sqlite3.connect("tracking.db")
    cursor = conn.execute("SELECT name, floor, time, image_path FROM logs WHERE name = ? ORDER BY time DESC", (name,))
    logs = [{"name": row[0], "floor": row[1], "time": row[2], "image_path": row[3]} for row in cursor]
    conn.close()
    return render_template("person_logs.html", name=name, logs=logs)

@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    return send_from_directory("evidence", filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)