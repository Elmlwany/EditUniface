# Rig Preparation Detection Configuration

# Camera Settings
CAMERA_INDEX = 1  # Change this if you want to use a different camera
CAMERA_WIDTH = 640  # Reduced for better performance
CAMERA_HEIGHT = 480  # Reduced for better performance
CAMERA_FPS = 30

# Detection Settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for person detection
MODEL_PATH = "yolov8n.pt"  # YOLO model file (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)

# Performance Settings
PERFORMANCE_MODE = "optimized"  # Options: "normal", "optimized", "ultra_fast"
FRAME_SKIP = 2  # Process every N frames (higher = faster, lower accuracy)
DETECTION_SCALE = 0.5  # Scale for detection (lower = faster)
MAX_FPS_TARGET = 15  # Target FPS for optimal performance

# Display Settings
ROI_COLOR = (0, 0, 255)  # Red color for ROI (BGR format)
PERSON_IN_ROI_COLOR = (0, 255, 0)  # Green color for person in ROI
ROI_TRANSPARENCY = 0.3  # Transparency of ROI overlay (0.0 to 1.0)

# Info Panel Settings
PANEL_HEIGHT = 120
PANEL_COLOR = (50, 50, 50)  # Dark gray
TEXT_COLOR = (255, 255, 255)  # White
COUNT_COLOR = (0, 255, 255)  # Yellow for people count
