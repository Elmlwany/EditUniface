# # main_app.py

# import cv2
# import face_recognition
# import sqlite3
# import pickle
# import threading
# import queue
# import os
# import signal
# import sys
# import time
# from datetime import datetime
# import numpy as np

# # Import settings from the configuration file
# import config

# # --- Global Control ---
# shutdown_event = threading.Event()
# frame_queue = queue.Queue(maxsize=100)  # Queue for frames from all cameras
# db_queue = queue.Queue()             # Queue for detections to be written to DB

# # --- Load Known Faces ---
# def load_known_faces():
#     """Loads face encodings from the pickle file."""
#     try:
#         with open(config.ENCODINGS_FILE, "rb") as f:
#             return pickle.load(f)
#     except FileNotFoundError:
#         print(f"[ERROR] Encodings file not found: {config.ENCODINGS_FILE}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"[ERROR] Could not load face encodings: {e}")
#         sys.exit(1)

# # --- Camera Streaming Thread ---
# class CameraStreamer(threading.Thread):
#     """A thread that continuously reads frames from an RTSP stream."""
#     def __init__(self, rtsp_url, location):
#         super().__init__()
#         self.rtsp_url = rtsp_url
#         self.location = location
#         self.daemon = True # Allows main thread to exit even if this thread is running

#     def run(self):
#         print(f"[INFO] Starting camera: {self.location}")
#         while not shutdown_event.is_set():
#             cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
#             if not cap.isOpened():
#                 print(f"[ERROR] Cannot open stream: {self.location}. Retrying in {config.RECONNECT_DELAY_SECONDS}s...")
#                 time.sleep(config.RECONNECT_DELAY_SECONDS)
#                 continue

#             while not shutdown_event.is_set():
#                 ret, frame = cap.read()
#                 if not ret:
#                     print(f"[WARN] Lost connection to {self.location}. Reconnecting...")
#                     break # Break inner loop to trigger reconnect
                
#                 # Put the frame and its source location into the shared queue
#                 if not frame_queue.full():
#                     frame_queue.put((frame, self.location))
                
#                 # Optional: display the raw feed
#                 if config.SHOW_WINDOWS:
#                     display_frame = frame.copy()
#                     cv2.putText(display_frame, self.location, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                     cv2.imshow(self.location, display_frame)
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         shutdown_event.set()

#             cap.release()
#             if config.SHOW_WINDOWS:
#                 cv2.destroyWindow(self.location)
#         print(f"[INFO] Camera thread stopped: {self.location}")

# # --- Face Processing Thread ---
# class FaceProcessor(threading.Thread):
#     """A thread that processes frames from the queue for face recognition."""
#     def __init__(self, known_faces):
#         super().__init__()
#         self.known_faces = known_faces
#         self.frame_count = 0
#         self.daemon = True

#     def run(self):
#         print("[INFO] Starting face processor thread.")
#         while not shutdown_event.is_set():
#             try:
#                 frame, location = frame_queue.get(timeout=1)
#                 self.frame_count += 1

#                 if self.frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
#                     continue

#                 # Resize for faster processing
#                 small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#                 rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#                 face_locations = face_recognition.face_locations(rgb_frame, model=config.FACE_DETECTION_MODEL)
#                 face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#                 for encoding in face_encodings:
#                     distances = face_recognition.face_distance(list(self.known_faces.values()), encoding)
#                     if len(distances) == 0:
#                         continue
                    
#                     best_match_index = np.argmin(distances)
#                     if distances[best_match_index] < 0.5: # Tolerance
#                         name = list(self.known_faces.keys())[best_match_index]
                        
#                         # Prepare data for the database
#                         timestamp = datetime.now()
#                         safe_time = timestamp.strftime("%Y%m%d_%H%M%S")
#                         filename = f"{name}_{location.replace(' ', '_')}_{safe_time}.jpg"
#                         filepath = os.path.join(config.EVIDENCE_DIR, filename)
                        
#                         # Save the original, full-resolution frame as evidence
#                         cv2.imwrite(filepath, frame)
                        
#                         # Put recognition result into the database queue
#                         db_queue.put((name, timestamp, location, filepath))

#                 frame_queue.task_done()
#             except queue.Empty:
#                 continue # If queue is empty, just loop again
#         print("[INFO] Face processor thread stopped.")

# # --- Database Writer Thread ---
# class DatabaseWriter(threading.Thread):
#     """A thread that writes detection logs to the SQLite database."""
#     def __init__(self):
#         super().__init__()
#         self.last_detection = {}
#         self.daemon = True

#     def run(self):
#         print("[INFO] Starting database writer thread.")
#         conn = sqlite3.connect(config.DB_FILE)
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS logs (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 name TEXT,
#                 time TEXT,
#                 location TEXT,
#                 image_path TEXT
#             )""")
#         conn.commit()

#         while not shutdown_event.is_set() or not db_queue.empty():
#             try:
#                 name, timestamp, location, image_path = db_queue.get(timeout=1)
                
#                 # Debounce: check if we logged this person recently
#                 last_time = self.last_detection.get((name, location))
#                 if last_time and (timestamp - last_time).total_seconds() < config.DEBOUNCE_SECONDS:
#                     db_queue.task_done()
#                     continue

#                 # Log the new detection
#                 self.last_detection[(name, location)] = timestamp
#                 time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
#                 cursor.execute("INSERT INTO logs (name, time, location, image_path) VALUES (?, ?, ?, ?)",
#                                (name, time_str, location, image_path))
#                 conn.commit()
#                 print(f"✅ [LOGGED] Found {name} at {location}")
#                 db_queue.task_done()

#             except queue.Empty:
#                 continue

#         conn.close()
#         print("[INFO] Database writer thread stopped and connection closed.")


# # --- Main Application Logic ---
# def main():
#     """Main function to start and manage all threads."""
#     # Graceful shutdown handler
#     def handle_exit(sig, frame):
#         print("\n[INFO] Shutdown signal received. Closing all threads...")
#         shutdown_event.set()
    
#     signal.signal(signal.SIGINT, handle_exit)
#     signal.signal(signal.SIGTERM, handle_exit)

#     # Prepare environment
#     os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
#     known_faces = load_known_faces()
#     print(f"[INFO] Loaded {len(known_faces)} known faces.")

#     # Start all threads
#     threads = [
#         FaceProcessor(known_faces),
#         DatabaseWriter()
#     ]
#     for url, loc in config.CAMERA_STREAMS.items():
#         threads.append(CameraStreamer(url, loc))

#     for t in threads:
#         t.start()

#     print("\n[INFO] Application running. Press Ctrl+C to exit.")
    
#     # Keep the main thread alive to handle signals
#     while not shutdown_event.is_set():
#         try:
#             time.sleep(1)
#         except KeyboardInterrupt:
#             handle_exit(None, None)

#     # Wait for all threads to complete
#     for t in threads:
#         t.join()

#     # Final cleanup
#     if config.SHOW_WINDOWS:
#         cv2.destroyAllWindows()
        
#     print("[INFO] Application shut down gracefully.")


# if __name__ == "__main__":
#     main()


















# main_app.py

# import cv2
# import face_recognition
# import sqlite3
# import pickle
# import threading
# import queue
# import os
# import signal
# import sys
# import time
# from datetime import datetime
# import numpy as np

# # Import settings from the configuration file
# import config

# # --- Global Control (No Changes) ---
# shutdown_event = threading.Event()
# frame_queue = queue.Queue(maxsize=100)
# db_queue = queue.Queue()

# # --- Load Known Faces (No Changes) ---
# def load_known_faces():
#     """Loads face encodings from the pickle file."""
#     try:
#         with open(config.ENCODINGS_FILE, "rb") as f:
#             return pickle.load(f)
#     except FileNotFoundError:
#         print(f"[ERROR] Encodings file not found: {config.ENCODINGS_FILE}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"[ERROR] Could not load face encodings: {e}")
#         sys.exit(1)

# # # --- Camera Streaming Thread (No Changes) ---
# # class CameraStreamer(threading.Thread):
# #     """A thread that continuously reads frames from an RTSP stream."""
# #     def __init__(self, rtsp_url, location):
# #         super().__init__()
# #         self.rtsp_url = rtsp_url
# #         self.location = location
# #         self.daemon = True

# #     def run(self):
# #         print(f"[INFO] Starting camera: {self.location}")
# #         while not shutdown_event.is_set():
# #             cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
# #             if not cap.isOpened():
# #                 print(f"[ERROR] Cannot open stream: {self.location}. Retrying in {config.RECONNECT_DELAY_SECONDS}s...")
# #                 time.sleep(config.RECONNECT_DELAY_SECONDS)
# #                 continue

# #             while not shutdown_event.is_set():
# #                 ret, frame = cap.read()
# #                 if not ret:
# #                     print(f"[WARN] Lost connection to {self.location}. Reconnecting...")
# #                     break 
                
# #                 if not frame_queue.full():
# #                     frame_queue.put((frame, self.location))
                
# #                 if config.SHOW_WINDOWS:
# #                     display_frame = frame.copy()
# #                     cv2.putText(display_frame, self.location, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# #                     cv2.imshow(self.location, display_frame)
# #                     if cv2.waitKey(1) & 0xFF == ord('q'):
# #                         shutdown_event.set()

# #             cap.release()
# #             if config.SHOW_WINDOWS:
# #                 cv2.destroyWindow(self.location)
# #         print(f"[INFO] Camera thread stopped: {self.location}")


# # --- Camera Streaming Thread (CORRECTED SHUTDOWN LOGIC) ---
# class CameraStreamer(threading.Thread):
#     """A thread that continuously reads frames from an RTSP stream."""
#     def __init__(self, rtsp_url, location):
#         super().__init__()
#         self.rtsp_url = rtsp_url
#         self.location = location
#         self.daemon = True

#     def run(self):
#         print(f"[INFO] Starting camera: {self.location}")
#         # The main loop continues as long as the application is running. It handles reconnects.
#         while not shutdown_event.is_set():
#             cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
#             if not cap.isOpened():
#                 print(f"[ERROR] Cannot open stream: {self.location}. Retrying in {config.RECONNECT_DELAY_SECONDS}s...")
#                 time.sleep(config.RECONNECT_DELAY_SECONDS)
#                 continue # Retry connection

#             # Inner loop processes frames from the active connection.
#             while not shutdown_event.is_set():
#                 ret, frame = cap.read()
#                 if not ret or frame is None:
#                     print(f"[WARN] Lost connection to {self.location}. Reconnecting...")
#                     break # Break inner loop to trigger cleanup and reconnect

#                 if not frame_queue.full():
#                     frame_queue.put((frame, self.location))
                
#                 if config.SHOW_WINDOWS:
#                     display_frame = frame.copy()

#                     if config.CROP_FRAME:
#                         h, w, _ = display_frame.shape
#                         start_percent, end_percent = config.CROP_HORIZONTAL_REGION_PERCENT
#                         start_col = int(w * start_percent)
#                         end_col = int(w * end_percent)
#                         overlay = display_frame.copy()
#                         cv2.rectangle(overlay, (start_col, 0), (end_col, h), (0, 255, 0), -1)
#                         alpha = 0.3
#                         display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
#                         cv2.line(display_frame, (start_col, 0), (start_col, h), (0, 255, 0), 2)
#                         cv2.line(display_frame, (end_col, 0), (end_col, h), (0, 255, 0), 2)

#                     cv2.putText(display_frame, self.location, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                     cv2.imshow(self.location, display_frame)
                    
#                     # <<< FIX: The key press check is crucial for shutdown >>>
#                     key = cv2.waitKey(1) & 0xFF
#                     if key == ord('q'):
#                         print("[INFO] 'q' key pressed. Initiating shutdown...")
#                         shutdown_event.set()
#                         break # Exit inner loop immediately

#             # <<< FIX: Graceful cleanup for THIS thread's connection and window >>>
#             cap.release()
#             if config.SHOW_WINDOWS:
#                 try:
#                     # Destroy only the window specific to this thread
#                     cv2.destroyWindow(self.location)
#                 except cv2.error:
#                     # This error can occur if the window was already closed, which is safe to ignore.
#                     pass
        
#         # Final confirmation message when the thread's main loop exits.
#         print(f"[INFO] Camera thread stopped: {self.location}")




# # --- Face Processing Thread (UPDATED) ---
# class FaceProcessor(threading.Thread):
#     """A thread that processes frames from the queue for face recognition."""
#     def __init__(self, known_faces):
#         super().__init__()
#         self.known_faces = known_faces
#         self.frame_count = 0
#         self.daemon = True

#     def run(self):
#         print("[INFO] Starting face processor thread.")
#         while not shutdown_event.is_set():
#             try:
#                 frame, location = frame_queue.get(timeout=1)
#                 self.frame_count += 1

#                 if self.frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
#                     frame_queue.task_done() # <<< NEW: Ensure task_done is called even on skipped frames
#                     continue

#                 # <<< NEW: Cropping logic starts here >>>
#                 if config.CROP_FRAME:
#                     h, w, _ = frame.shape
#                     start_percent, end_percent = config.CROP_HORIZONTAL_REGION_PERCENT
#                     start_col = int(w * start_percent)
#                     end_col = int(w * end_percent)
                    
#                     # This is the area we will perform recognition on
#                     processing_area = frame[:, start_col:end_col]
#                 else:
#                     # If cropping is disabled, process the whole frame
#                     processing_area = frame
#                 # <<< NEW: Cropping logic ends here >>>

#                 # Resize for faster processing, now using the potentially smaller 'processing_area'
#                 small_frame = cv2.resize(processing_area, (0, 0), fx=0.5, fy=0.5) # <<< CHANGED
#                 rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#                 face_locations = face_recognition.face_locations(rgb_frame, model=config.FACE_DETECTION_MODEL)
#                 face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#                 for encoding in face_encodings:
#                     distances = face_recognition.face_distance(list(self.known_faces.values()), encoding)
#                     if len(distances) == 0:
#                         continue
                    
#                     best_match_index = np.argmin(distances)
#                     if distances[best_match_index] < 0.5:
#                         name = list(self.known_faces.keys())[best_match_index]
                        
#                         timestamp = datetime.now()
#                         safe_time = timestamp.strftime("%Y%m%d_%H%M%S")
#                         filename = f"{name}_{location.replace(' ', '_')}_{safe_time}.jpg"
#                         filepath = os.path.join(config.EVIDENCE_DIR, filename)
                        
#                         # IMPORTANT: Save the original, full-resolution frame for evidence
#                         cv2.imwrite(filepath, frame)
                        
#                         db_queue.put((name, timestamp, location, filepath))

#                 frame_queue.task_done()
#             except queue.Empty:
#                 continue
#         print("[INFO] Face processor thread stopped.")


# # --- Database Writer Thread (No Changes) ---
# class DatabaseWriter(threading.Thread):
#     """A thread that writes detection logs to the SQLite database."""
#     def __init__(self):
#         super().__init__()
#         self.last_detection = {}
#         self.daemon = True

#     def run(self):
#         print("[INFO] Starting database writer thread.")
#         conn = sqlite3.connect(config.DB_FILE)
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS logs (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 name TEXT,
#                 time TEXT,
#                 location TEXT,
#                 image_path TEXT
#             )""")
#         conn.commit()

#         while not shutdown_event.is_set() or not db_queue.empty():
#             try:
#                 name, timestamp, location, image_path = db_queue.get(timeout=1)
                
#                 last_time = self.last_detection.get((name, location))
#                 if last_time and (timestamp - last_time).total_seconds() < config.DEBOUNCE_SECONDS:
#                     db_queue.task_done()
#                     continue

#                 self.last_detection[(name, location)] = timestamp
#                 time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
#                 cursor.execute("INSERT INTO logs (name, time, location, image_path) VALUES (?, ?, ?, ?)",
#                                (name, time_str, location, image_path))
#                 conn.commit()
#                 print(f"✅ [LOGGED] Found {name} at {location}")
#                 db_queue.task_done()

#             except queue.Empty:
#                 continue

#         conn.close()
#         print("[INFO] Database writer thread stopped and connection closed.")


# # --- Main Application Logic (No Changes) ---
# def main():
#     """Main function to start and manage all threads."""
#     def handle_exit(sig, frame):
#         print("\n[INFO] Shutdown signal received. Closing all threads...")
#         shutdown_event.set()
    
#     signal.signal(signal.SIGINT, handle_exit)
#     signal.signal(signal.SIGTERM, handle_exit)

#     os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
#     known_faces = load_known_faces()
#     print(f"[INFO] Loaded {len(known_faces)} known faces.")

#     threads = [
#         FaceProcessor(known_faces),
#         DatabaseWriter()
#     ]
#     for url, loc in config.CAMERA_STREAMS.items():
#         threads.append(CameraStreamer(url, loc))

#     for t in threads:
#         t.start()

#     print("\n[INFO] Application running. Press Ctrl+C to exit.")
    
#     while not shutdown_event.is_set():
#         try:
#             time.sleep(1)
#         except KeyboardInterrupt:
#             handle_exit(None, None)

#     for t in threads:
#         t.join()

#     if config.SHOW_WINDOWS:
#         cv2.destroyAllWindows()
        
#     print("[INFO] Application shut down gracefully.")


# if __name__ == "__main__":
#     main()




# main_app.py

import cv2
import face_recognition
import sqlite3
import pickle
import threading
import queue
import os
import signal
import sys
import time
from datetime import datetime
import numpy as np

# Import settings from the configuration file
import config

# --- Global Control ---
shutdown_event = threading.Event()
frame_queue = queue.Queue(maxsize=100)  # Queue for frames from all cameras
db_queue = queue.Queue()             # Queue for detections to be written to DB

# --- Load Known Faces ---
def load_known_faces():
    """Loads face encodings from the pickle file."""
    try:
        with open(config.ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Encodings file not found: {config.ENCODINGS_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not load face encodings: {e}")
        sys.exit(1)

# --- Camera Streaming Thread ---
class CameraStreamer(threading.Thread):
    """
    A thread that continuously reads frames from an RTSP stream,
    handles reconnects, and provides visual feedback.
    """
    def __init__(self, rtsp_url, location):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.location = location
        self.daemon = True

    def run(self):
        print(f"[INFO] Starting camera: {self.location}")
        # The main loop continues as long as the application is running, handling reconnects.
        while not shutdown_event.is_set():
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open stream: {self.location}. Retrying in {config.RECONNECT_DELAY_SECONDS}s...")
                time.sleep(config.RECONNECT_DELAY_SECONDS)
                continue # Retry connection

            # Inner loop processes frames from the active connection.
            while not shutdown_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"[WARN] Lost connection to {self.location}. Reconnecting...")
                    break # Break inner loop to trigger cleanup and reconnect

                if not frame_queue.full():
                    frame_queue.put((frame, self.location))
                
                if config.SHOW_WINDOWS:
                    display_frame = frame.copy()

                    # Draw a semi-transparent box to show the active processing region
                    if config.CROP_FRAME:
                        h, w, _ = display_frame.shape
                        start_percent, end_percent = config.CROP_HORIZONTAL_REGION_PERCENT
                        start_col = int(w * start_percent)
                        end_col = int(w * end_percent)
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (start_col, 0), (end_col, h), (0, 255, 0), -1)
                        alpha = 0.3
                        display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
                        cv2.line(display_frame, (start_col, 0), (start_col, h), (0, 255, 0), 2)
                        cv2.line(display_frame, (end_col, 0), (end_col, h), (0, 255, 0), 2)

                    cv2.putText(display_frame, self.location, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(self.location, display_frame)
                    
                    # Check for 'q' key press to initiate a graceful shutdown
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] 'q' key pressed. Initiating shutdown...")
                        shutdown_event.set()
                        break 

            # Gracefully release the capture and destroy THIS thread's window before reconnecting
            cap.release()
            if config.SHOW_WINDOWS:
                try:
                    cv2.destroyWindow(self.location)
                except cv2.error:
                    # This can occur if the window was already closed. It's safe to ignore.
                    pass
        
        print(f"[INFO] Camera thread stopped: {self.location}")


# --- Face Processing Thread ---
class FaceProcessor(threading.Thread):
    """
    A thread that processes frames for face recognition, optimized for
    low-resolution streams.
    """
    def __init__(self, known_faces):
        super().__init__()
        self.known_faces = known_faces
        self.frame_count = 0
        self.daemon = True

    def run(self):
        print("[INFO] Starting face processor thread.")
        while not shutdown_event.is_set():
            try:
                frame, location = frame_queue.get(timeout=1)
                self.frame_count += 1

                if self.frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
                    frame_queue.task_done()
                    continue

                # If enabled, crop the frame to the middle section for processing
                if config.CROP_FRAME:
                    h, w, _ = frame.shape
                    start_percent, end_percent = config.CROP_HORIZONTAL_REGION_PERCENT
                    start_col = int(w * start_percent)
                    end_col = int(w * end_percent)
                    processing_area = frame[:, start_col:end_col]
                else:
                    processing_area = frame

                # OPTIMIZATION: Process the substream at its native resolution.
                # Do not resize the 'processing_area' down further, as it's already low-resolution.
                rgb_frame = cv2.cvtColor(processing_area, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_frame, model=config.FACE_DETECTION_MODEL)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for encoding in face_encodings:
                    distances = face_recognition.face_distance(list(self.known_faces.values()), encoding)
                    if len(distances) == 0:
                        continue
                    
                    best_match_index = np.argmin(distances)
                    
                    # OPTIMIZATION: Use a slightly higher tolerance for blurry substream images.
                    if distances[best_match_index] < 0.55:
                        name = list(self.known_faces.keys())[best_match_index]
                        
                        timestamp = datetime.now()
                        safe_time = timestamp.strftime("%Y%m%d_%H%M%S")
                        filename = f"{name}_{location.replace(' ', '_')}_{safe_time}.jpg"
                        filepath = os.path.join(config.EVIDENCE_DIR, filename)
                        
                        # Save the original, full-resolution frame as evidence for context
                        cv2.imwrite(filepath, frame)
                        
                        db_queue.put((name, timestamp, location, filepath))

                frame_queue.task_done()
            except queue.Empty:
                continue
        print("[INFO] Face processor thread stopped.")


# --- Database Writer Thread ---
class DatabaseWriter(threading.Thread):
    """A thread that writes detection logs to the SQLite database with debouncing."""
    def __init__(self):
        super().__init__()
        self.last_detection = {}
        self.daemon = True

    def run(self):
        print("[INFO] Starting database writer thread.")
        conn = sqlite3.connect(config.DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                time TEXT,
                location TEXT,
                image_path TEXT
            )""")
        conn.commit()

        while not shutdown_event.is_set() or not db_queue.empty():
            try:
                name, timestamp, location, image_path = db_queue.get(timeout=1)
                
                # Debounce: check if we logged this person at this location recently
                last_time = self.last_detection.get((name, location))
                if last_time and (timestamp - last_time).total_seconds() < config.DEBOUNCE_SECONDS:
                    db_queue.task_done()
                    continue

                self.last_detection[(name, location)] = timestamp
                time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("INSERT INTO logs (name, time, location, image_path) VALUES (?, ?, ?, ?)",
                               (name, time_str, location, image_path))
                conn.commit()
                print(f"✅ [LOGGED] Found {name} at {location}")
                db_queue.task_done()

            except queue.Empty:
                continue

        conn.close()
        print("[INFO] Database writer thread stopped and connection closed.")


# --- Main Application Logic ---
def main():
    """Main function to start and manage all threads."""
    # Graceful shutdown handler for Ctrl+C
    def handle_exit(sig, frame):
        print("\n[INFO] Shutdown signal received. Closing all threads...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Prepare environment
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    known_faces = load_known_faces()
    print(f"[INFO] Loaded {len(known_faces)} known faces.")

    # Start all threads
    threads = [
        FaceProcessor(known_faces),
        DatabaseWriter()
    ]
    for url, loc in config.CAMERA_STREAMS.items():
        threads.append(CameraStreamer(url, loc))

    for t in threads:
        t.start()

    print("\n[INFO] Application running. Press 'q' in a camera window or Ctrl+C to exit.")
    
    # Keep the main thread alive to handle signals and wait for shutdown
    while not shutdown_event.is_set():
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            handle_exit(None, None)

    # Wait for all threads to complete their work
    for t in threads:
        t.join()

    # Final cleanup of any remaining OpenCV windows
    if config.SHOW_WINDOWS:
        cv2.destroyAllWindows()
        
    print("[INFO] Application shut down gracefully.")


if __name__ == "__main__":
    main()