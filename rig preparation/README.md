# Rig Preparation Detection System

This system uses camera index 1 to detect people only within a user-defined area of interest (ROI). The selected area is highlighted in red, and only people detected within this area are counted and marked.

## Features

- 🎥 **Camera Input**: Uses camera index 1 (configurable)
- 🖱️ **Mouse ROI Selection**: Click and drag to select area of interest
- 🔴 **Visual ROI Indicator**: Selected area highlighted in red with transparency
- 👤 **Person Detection**: Only detects people within the selected area
- 📊 **Real-time Stats**: Shows people count, FPS, and system status
- ⚙️ **Configurable Settings**: Easy configuration through config.py

## Quick Start

### Method 1: Use Launcher Scripts
- **Windows Batch**: Double-click `start_detection.bat`
- **PowerShell**: Right-click `start_detection.ps1` → "Run with PowerShell"

### Method 2: Manual Execution
```bash
# Activate Python environment
.\.venv\Scripts\Activate.ps1

# Run the detection system
python "rig preparation\rig_prep_detection_advanced.py"
```

## How to Use

1. **Start the Application**: Use one of the launcher scripts or run manually
2. **Select Area of Interest**: 
   - Click and drag with your mouse to draw a rectangle
   - The selected area will be highlighted in red
   - Only people within this red area will be detected
3. **Monitor Detection**: 
   - Green boxes show people detected in the ROI
   - Information panel shows real-time statistics
4. **Controls**:
   - `R` key: Reset the area selection
   - `Q` key: Quit the application

## Configuration

Edit `config.py` to customize settings:

```python
# Camera Settings
CAMERA_INDEX = 1  # Change camera (0, 1, 2, etc.)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Detection Settings
CONFIDENCE_THRESHOLD = 0.5  # Detection sensitivity
MODEL_PATH = "yolov8n.pt"  # YOLO model file

# Visual Settings
ROI_COLOR = (0, 0, 255)  # Red color for ROI
ROI_TRANSPARENCY = 0.3   # ROI overlay transparency
```

## File Structure

```
rig preparation/
├── rig_prep_detection_advanced.py  # Main detection application
├── config.py                       # Configuration file
├── start_detection.bat             # Windows batch launcher
├── start_detection.ps1             # PowerShell launcher
└── README.md                      # This file
```

## Requirements

- Python with OpenCV
- ultralytics (YOLO)
- Webcam/Camera at index 1
- yolov8n.pt model file (in parent directory)

## Troubleshooting

### Camera Issues
- If camera index 1 doesn't work, try changing `CAMERA_INDEX` in config.py
- Run the application to see available cameras listed in console

### Model Issues
- Ensure `yolov8n.pt` is in the parent directory
- Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

### Performance Issues
- Lower camera resolution in config.py
- Use yolov8n.pt for fastest performance
- Use yolov8s.pt or yolov8m.pt for better accuracy

## Use Cases

- **Rig Preparation Monitoring**: Monitor specific work areas
- **Safety Compliance**: Ensure personnel stay in designated areas
- **Area Access Control**: Count people in restricted zones
- **Workflow Optimization**: Analyze work area utilization

## Technical Details

- **Detection Model**: YOLOv8 (You Only Look Once)
- **Person Class ID**: 0 (COCO dataset standard)
- **ROI Method**: Center-point intersection
- **Framework**: OpenCV + ultralytics

---

**Note**: This system is designed for rig preparation monitoring and can be adapted for various area monitoring applications.
