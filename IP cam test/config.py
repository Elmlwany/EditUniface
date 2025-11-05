# config.py

# ... (all your existing settings) ...

# === Processing Configuration ===
# Set to True to process only a vertical slice of the frame
CROP_FRAME = True 
# Define the start and end of the slice as a percentage of the frame's width
# (0.33, 0.66) corresponds to the middle third.
CROP_HORIZONTAL_REGION_PERCENT = (0.33, 0.66)


# === System Configuration ===
# Set to False to run in "headless" mode without displaying camera windows
SHOW_WINDOWS = True 
# Model for face detection: 'hog' is faster, 'cnn' is more accurate but CPU-intensive
FACE_DETECTION_MODEL = "hog" 
# Process every Nth frame from the central queue to save CPU
PROCESS_EVERY_N_FRAMES = 5
# Time in seconds to wait before logging the same person again
DEBOUNCE_SECONDS = 10 
# Time in seconds for a camera thread to wait before attempting to reconnect
RECONNECT_DELAY_SECONDS = 5

# === File & Directory Paths ===
ENCODINGS_FILE = "face_encodings.pkl"
DB_FILE = "tracking.db"
EVIDENCE_DIR = "evidence"

# === Camera Configuration ===
# Dictionary mapping RTSP URLs to their location names
CAMERA_STREAMS = {
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/101": "Finance Room",
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/202": "AI Room",
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/302": "OUTDOOR",
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/402": "meeting room",
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/502": "cor-cam",
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/602": "Reception",
    "rtsp://admin:Admin123@192.168.0.103:554/Streaming/Channels/702": "OPS-ROOM"
}