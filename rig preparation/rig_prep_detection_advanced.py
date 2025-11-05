import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import sys

# Add current directory to path to import config
sys.path.append(os.path.dirname(__file__))
import config

class RigPreparationDetection:
    def __init__(self):
        self.camera_index = config.CAMERA_INDEX
        self.cap = None
        self.model = None
        self.roi_points = []
        self.roi_selected = False
        self.roi_mask = None
        self.drawing = False
        self.frame = None
        self.detection_active = True
        
        # Initialize YOLO model
        self.init_model()
        
        # Initialize camera
        self.init_camera()
        
    def init_model(self):
        """Initialize the YOLO model"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), "..", config.MODEL_PATH)
            if not os.path.exists(model_path):
                model_path = config.MODEL_PATH
            
            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Make sure yolov8n.pt is available in the parent directory")
            return
        
    def init_camera(self):
        """Initialize the camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                print("Available cameras:")
                # Try to find available cameras
                for i in range(5):
                    test_cap = cv2.VideoCapture(i)
                    if test_cap.isOpened():
                        print(f"  Camera {i}: Available")
                        test_cap.release()
                    else:
                        print(f"  Camera {i}: Not available")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            # Get actual properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera {self.camera_index} initialized:")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  FPS: {actual_fps}")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi_points = [(x, y)]
            self.roi_selected = False
            print("Starting ROI selection...")
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update the second point while dragging
                if len(self.roi_points) == 1:
                    self.roi_points.append((x, y))
                else:
                    self.roi_points[1] = (x, y)
                    
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.roi_points) == 2:
                self.roi_points[1] = (x, y)
                self.create_roi_mask()
                self.roi_selected = True
                width = abs(self.roi_points[1][0] - self.roi_points[0][0])
                height = abs(self.roi_points[1][1] - self.roi_points[0][1])
                print(f"ROI selected: {self.roi_points[0]} to {self.roi_points[1]} (Size: {width}x{height})")
    
    def create_roi_mask(self):
        """Create a mask for the selected ROI"""
        if len(self.roi_points) == 2 and self.frame is not None:
            h, w = self.frame.shape[:2]
            self.roi_mask = np.zeros((h, w), dtype=np.uint8)
            
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            
            # Ensure coordinates are in correct order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Create rectangular mask
            self.roi_mask[y1:y2, x1:x2] = 255
    
    def is_person_in_roi(self, bbox):
        """Check if a person bounding box intersects with ROI"""
        if not self.roi_selected or self.roi_mask is None:
            return False
        
        x1, y1, x2, y2 = bbox
        
        # Get center point of bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Check if center point is in ROI
        if (0 <= center_y < self.roi_mask.shape[0] and 
            0 <= center_x < self.roi_mask.shape[1]):
            return self.roi_mask[center_y, center_x] > 0
        
        return False
    
    def detect_people(self, frame):
        """Detect people in the frame using YOLO"""
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, verbose=False)
            people_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Class 0 is 'person' in COCO dataset
                        if int(box.cls[0]) == 0:  # person class
                            confidence = float(box.conf[0])
                            if confidence > config.CONFIDENCE_THRESHOLD:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                people_detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence
                                })
            
            return people_detections
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
    
    def draw_roi(self, frame):
        """Draw the ROI on the frame"""
        if len(self.roi_points) >= 2:
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            
            # Ensure coordinates are in correct order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Draw red rectangle for ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.ROI_COLOR, 3)
            
            # Add semi-transparent red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), config.ROI_COLOR, -1)
            cv2.addWeighted(frame, 1-config.ROI_TRANSPARENCY, overlay, config.ROI_TRANSPARENCY, 0, frame)
            
            # Add ROI label with background
            label = "AREA OF INTEREST"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x1, y1-text_height-20), (x1+text_width+10, y1-5), config.ROI_COLOR, -1)
            cv2.putText(frame, label, (x1+5, y1-10), font, font_scale, (255, 255, 255), thickness)
        
        elif self.drawing and len(self.roi_points) == 1:
            # Draw selection in progress
            cv2.circle(frame, self.roi_points[0], 5, config.ROI_COLOR, -1)
    
    def draw_detections(self, frame, detections):
        """Draw person detections on the frame"""
        people_in_roi = 0
        total_people = len(detections)
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            # Check if person is in ROI
            in_roi = self.is_person_in_roi(bbox)
            
            if in_roi:
                people_in_roi += 1
                # Green box for people in ROI
                color = config.PERSON_IN_ROI_COLOR
                label = f"PERSON IN ROI: {confidence:.2f}"
                thickness = 3
            else:
                # Skip drawing people outside ROI
                continue
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            # Background rectangle for text
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       font, font_scale, (255, 255, 255), text_thickness)
        
        return people_in_roi, total_people
    
    def add_info_panel(self, frame, people_in_roi, total_people, fps=0):
        """Add information panel to the frame"""
        h, w = frame.shape[:2]
        
        # Create info panel background
        panel = np.zeros((config.PANEL_HEIGHT, w, 3), dtype=np.uint8)
        panel[:] = config.PANEL_COLOR
        
        # Add border
        cv2.rectangle(panel, (0, 0), (w-1, config.PANEL_HEIGHT-1), (100, 100, 100), 2)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        info_texts = [
            ("RIG PREPARATION - PERSON DETECTION SYSTEM", config.TEXT_COLOR),
            (f"People in Area of Interest: {people_in_roi}", config.COUNT_COLOR),
            (f"Total People Detected: {total_people}", config.TEXT_COLOR),
            (f"FPS: {fps:.1f} | Camera: {self.camera_index} | ROI: {'Selected' if self.roi_selected else 'Not Selected'}", config.TEXT_COLOR),
            ("Controls: Click+Drag=Select ROI | R=Reset | Q=Quit", config.TEXT_COLOR)
        ]
        
        for i, (text, color) in enumerate(info_texts):
            y_pos = 20 + (i * 22)
            cv2.putText(panel, text, (10, y_pos), font, font_scale, color, thickness)
        
        # Combine frame and panel
        result = np.vstack([frame, panel])
        return result
    
    def run(self):
        """Main loop for the detection system"""
        if self.cap is None or not self.cap.isOpened():
            print("Camera not available. Exiting...")
            return
        
        # Create window and set mouse callback
        window_name = 'Rig Preparation Detection System'
        cv2.namedWindow(window_name, cv2.WINDOW_RESIZABLE)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("RIG PREPARATION DETECTION SYSTEM STARTED")
        print("="*60)
        print("Instructions:")
        print("• Click and drag with mouse to select area of interest")
        print("• Only people in the red area will be detected and counted")
        print("• Press 'R' to reset the area selection")
        print("• Press 'Q' to quit the application")
        print("="*60)
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break
            
            self.frame = frame.copy()
            
            # Detect people
            detections = self.detect_people(frame)
            
            # Draw ROI
            self.draw_roi(frame)
            
            # Draw detections (only people in ROI)
            people_in_roi, total_people = self.draw_detections(frame, detections)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end_time = time.time()
                current_fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            
            # Add info panel
            frame_with_info = self.add_info_panel(frame, people_in_roi, total_people, current_fps)
            
            # Display frame
            cv2.imshow(window_name, frame_with_info)
            
            # Print status every few seconds
            if fps_counter % 90 == 0:  # Every 3 seconds at 30 FPS
                status = f"ROI: {'✓' if self.roi_selected else '✗'} | People in ROI: {people_in_roi} | Total: {total_people} | FPS: {current_fps:.1f}"
                print(status)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Quitting application...")
                break
            elif key == ord('r') or key == ord('R'):
                # Reset ROI
                self.roi_points = []
                self.roi_selected = False
                self.roi_mask = None
                print("ROI reset - Please select new area of interest")
        
        # Cleanup
        print("Cleaning up resources...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")

def main():
    """Main function"""
    print("Initializing Rig Preparation Detection System...")
    detector = RigPreparationDetection()
    detector.run()

if __name__ == "__main__":
    main()
