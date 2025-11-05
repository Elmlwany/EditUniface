# import cv2
# import face_recognition
# import sqlite3
# import pickle
# from datetime import datetime
# import threading
# import queue
# import os

# # Load precomputed encodings
# ENCODINGS_FILE = "face_encodings.pkl"
# try:
#     with open(ENCODINGS_FILE, "rb") as f:
#         known_faces = pickle.load(f)
#     print("Loaded precomputed encodings.")
# except FileNotFoundError:
#     print(f"Error: {ENCODINGS_FILE} not found. Run train_faces.py first.")
#     exit(1)
# except Exception as e:
#     print(f"Error loading pickle file: {e}")
#     exit(1)

# # Setup database and queue
# conn = sqlite3.connect("tracking.db", check_same_thread=False)
# cursor = conn.cursor()

# # Create table if it doesn’t exist
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS logs (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         name TEXT,
#         time TEXT,
#         floor TEXT,
#         image_path TEXT
#     )
# """)
# conn.commit()
# detection_queue = queue.Queue()

# # Camera-floor mapping with RTSP URL
# CAMERA_FLOORS = {
#     "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/601": "Floor 1"
# }

# # Track last detection time per person to debounce
# last_detection = {}
# DEBOUNCE_SECONDS = 5

# # Directory for saving evidence images
# EVIDENCE_DIR = "evidence"
# if not os.path.exists(EVIDENCE_DIR):
#     os.makedirs(EVIDENCE_DIR)

# # Function to process database writes
# def database_writer():
#     while True:
#         try:
#             name, time_now, floor, image_path = detection_queue.get(timeout=1)
#             current_time = datetime.strptime(time_now, "%Y-%m-%d %H:%M:%S")
#             last_time = last_detection.get(name)

#             if last_time is None or (current_time - last_time).total_seconds() >= DEBOUNCE_SECONDS:
#                 conn.execute("INSERT INTO logs (name, time, floor, image_path) VALUES (?, ?, ?, ?)", 
#                             (name, time_now, floor, image_path))
#                 conn.commit()
#                 last_detection[name] = current_time
#                 print(f"Logged: {name} on {floor} at {time_now} with image {image_path}")
#             detection_queue.task_done()
#         except queue.Empty:
#             if not any(thread.is_alive() for thread in threading.enumerate() if thread != threading.current_thread()):
#                 break

# # Function to process a single camera
# def process_camera(rtsp_url, floor):
#     video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
#     video_capture.set(cv2.CAP_PROP_FPS, 30)

#     if not video_capture.isOpened():
#         print(f"Error: Could not open RTSP stream {rtsp_url} for {floor}")
#         return

#     # Try reading initial frames to handle stream initialization
#     for _ in range(10):
#         ret, frame = video_capture.read()
#         if ret and frame is not None:
#             print(f"Frame captured successfully from {rtsp_url}")
#             break
#         print(f"Warning: Failed to capture frame from {rtsp_url}, retrying...")
#     else:
#         print(f"Error: Cannot read frame from {rtsp_url} after multiple attempts")
#         video_capture.release()
#         return

#     window_name = f"Camera {rtsp_url} - {floor}"
#     frame_count = 0
#     PROCESS_EVERY_N_FRAMES = 5

#     while True:
#         ret, frame = video_capture.read()
#         if not ret or frame is None:
#             print(f"Error: Failed to grab frame from {rtsp_url}")
#             break

#         # Resize frame for display
#         frame = cv2.resize(frame, (640, 480))
#         cv2.putText(frame, f"{floor}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.imshow(window_name, frame)

#         frame_count += 1
#         if frame_count % PROCESS_EVERY_N_FRAMES == 0:
#             small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#             rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#             face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1)
#             face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#             for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
#                 matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding, tolerance=0.5)
#                 face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
#                 name = "Unknown"
#                 if matches and min(face_distances) < 0.5:
#                     first_match_index = matches.index(True)
#                     name = list(known_faces.keys())[first_match_index]

#                 if name != "Unknown":
#                     time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     top, right, bottom, left = [x * 2 for x in face_location]
#                     face_image = frame[top:bottom, left:right]
#                     image_filename = f"{name}_{time_now.replace(':', '-')}_{floor.replace(' ', '_')}.jpg"
#                     image_path = os.path.join(EVIDENCE_DIR, image_filename)
#                     cv2.imwrite(image_path, face_image)
#                     detection_queue.put((name, time_now, floor, image_path))

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     video_capture.release()
#     cv2.destroyWindow(window_name)

# # Start database writer thread
# writer_thread = threading.Thread(target=database_writer)
# writer_thread.start()

# # Start camera threads
# threads = []
# for rtsp_url, floor in CAMERA_FLOORS.items():
#     thread = threading.Thread(target=process_camera, args=(rtsp_url, floor))
#     thread.start()
#     threads.append(thread)

# # Wait for camera threads to finish
# for thread in threads:
#     thread.join()

# detection_queue.join()
# conn.close()
# print("All cameras stopped.")








import cv2
import face_recognition
import sqlite3
import pickle
from datetime import datetime
import threading
import queue
import os

# Load precomputed encodings
ENCODINGS_FILE = "face_encodings.pkl"
try:
    with open(ENCODINGS_FILE, "rb") as f:
        known_faces = pickle.load(f)
    print("Loaded precomputed encodings.")
except FileNotFoundError:
    print(f"Error: {ENCODINGS_FILE} not found. Run train_faces.py first.")
    exit(1)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit(1)

# Setup database and queue
conn = sqlite3.connect("tracking.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if it doesn’t exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        time TEXT,
        location TEXT,
        image_path TEXT
    )
""")
conn.commit()
detection_queue = queue.Queue()

# Camera-location mapping with RTSP URLs
CAMERA_FLOORS = {
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/101": "Finance Room",
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/201": "AI Room",
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/601": "reciption"
}

# Track last detection time per person to debounce
last_detection = {}
DEBOUNCE_SECONDS = 5

# Directory for saving evidence images
EVIDENCE_DIR = "evidence"
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

# Function to process database writes
def database_writer():
    while True:
        try:
            name, time_now, location, image_path = detection_queue.get(timeout=1)
            current_time = datetime.strptime(time_now, "%Y-%m-%d %H:%M:%S")
            last_time = last_detection.get(name)

            if last_time is None or (current_time - last_time).total_seconds() >= DEBOUNCE_SECONDS:
                conn.execute("INSERT INTO logs (name, time, location, image_path) VALUES (?, ?, ?, ?)", 
                            (name, time_now, location, image_path))
                conn.commit()
                last_detection[name] = current_time
                print(f"Logged: {name} in {location} at {time_now} with image {image_path}")
            detection_queue.task_done()
        except queue.Empty:
            if not any(thread.is_alive() for thread in threading.enumerate() if thread != threading.current_thread()):
                break

# Function to process a single camera
def process_camera(rtsp_url, location):
    video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
    video_capture.set(cv2.CAP_PROP_FPS, 10)       # Lower FPS
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not video_capture.isOpened():
        print(f"Error: Could not open RTSP stream {rtsp_url} for {location}")
        return

    # Try reading initial frames
    for _ in range(30):
        ret, frame = video_capture.read()
        if ret and frame is not None:
            print(f"Frame captured successfully from {rtsp_url}")
            break
        print(f"Warning: Failed to capture frame from {rtsp_url}, retrying...")
    else:
        print(f"Error: Cannot read frame from {rtsp_url}")
        video_capture.release()
        return

    window_name = f"Camera {location}"
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 10

    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            print(f"Error: Failed to grab frame from {rtsp_url}")
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        cv2.imwrite(f"processed_frame_{location.replace(' ', '_')}_{frame_count}.jpg", frame)  # Debug
        cv2.putText(frame, f"{location}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if not face_locations:
                print(f"No faces detected in frame {frame_count} from {location}")
            else:
                print(f"Detected {len(face_locations)} faces in frame {frame_count} from {location}")

            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding, tolerance=0.4)
                face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
                name = "Unknown"
                if matches and min(face_distances) < 0.4:
                    first_match_index = matches.index(True)
                    name = list(known_faces.keys())[first_match_index]
                    print(f"Recognized {name} in frame {frame_count} from {location}")

                if name != "Unknown":
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    top, right, bottom, left = [x * 2 for x in face_location]
                    face_image = frame[top:bottom, left:right]
                    image_filename = f"{name}_{time_now.replace(':', '-')}_{location.replace(' ', '_')}.jpg"
                    image_path = os.path.join(EVIDENCE_DIR, image_filename)
                    cv2.imwrite(image_path, face_image)
                    detection_queue.put((name, time_now, location, image_path))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyWindow(window_name)

# Start database writer thread
writer_thread = threading.Thread(target=database_writer)
writer_thread.start()

# Start camera threads
threads = []
for rtsp_url, location in CAMERA_FLOORS.items():
    thread = threading.Thread(target=process_camera, args=(rtsp_url, location))
    thread.start()
    threads.append(thread)

# Wait for camera threads to finish
for thread in threads:
    thread.join()

detection_queue.join()
conn.close()
print("All cameras stopped.")