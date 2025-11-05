import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

class RigPreparationDetection:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.cap = None
        self.model = None
        self.roi_points = []
        self.roi_selected = False
        self.roi_mask = None
        self.drawing = False
        self.frame = None
        self.detection_active = True
        
        # Initialize YOLO model
        try:
            self.model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, yolov8m.pt, etc. for better accuracy
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return
        
        # Initialize camera
        self.init_camera()
        
    def init_camera(self):
        """Initialize the camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera {self.camera_index} initialized successfully")
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
                print(f"ROI selected: {self.roi_points}")
    
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
                            if confidence > 0.5:  # Confidence threshold
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
            
            # Draw red rectangle for ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add semi-transparent red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
            
            # Add ROI label
            cv2.putText(frame, "AREA OF INTEREST", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def draw_detections(self, frame, detections):
        """Draw person detections on the frame"""
        people_in_roi = 0
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            # Check if person is in ROI
            in_roi = self.is_person_in_roi(bbox)
            
            if in_roi:
                people_in_roi += 1
                # Green box for people in ROI
                color = (0, 255, 0)
                label = f"Person in ROI: {confidence:.2f}"
            else:
                # Only draw if not in ROI mode or for debugging
                continue  # Skip drawing people outside ROI
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return people_in_roi
    
    def add_info_panel(self, frame, people_count):
        """Add information panel to the frame"""
        h, w = frame.shape[:2]
        
        # Create info panel background
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)  # Dark gray background
        
        # Add text information
        info_texts = [
            "RIG PREPARATION - PERSON DETECTION",
            f"People in ROI: {people_count}",
            "Instructions: Click and drag to select area of interest",
            "Press 'r' to reset ROI, 'q' to quit"
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = 25 + (i * 25)
            color = (0, 255, 255) if i == 1 else (255, 255, 255)  # Yellow for count
            cv2.putText(panel, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Combine frame and panel
        result = np.vstack([frame, panel])
        return result
    
    def run(self):
        """Main loop for the detection system"""
        if self.cap is None or not self.cap.isOpened():
            print("Camera not available")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow('Rig Preparation Detection', cv2.WINDOW_RESIZABLE)
        cv2.setMouseCallback('Rig Preparation Detection', self.mouse_callback)
        
        print("Starting Rig Preparation Detection System...")
        print("Instructions:")
        print("- Click and drag to select area of interest")
        print("- Press 'r' to reset ROI")
        print("- Press 'q' to quit")
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            self.frame = frame.copy()
            
            # Detect people
            detections = self.detect_people(frame)
            
            # Draw ROI
            self.draw_roi(frame)
            
            # Draw detections (only people in ROI)
            people_in_roi = self.draw_detections(frame, detections)
            
            # Add info panel
            frame_with_info = self.add_info_panel(frame, people_in_roi)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                print(f"FPS: {fps:.1f}, People in ROI: {people_in_roi}")
            
            # Display frame
            cv2.imshow('Rig Preparation Detection', frame_with_info)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset ROI
                self.roi_points = []
                self.roi_selected = False
                self.roi_mask = None
                print("ROI reset")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    # You can change camera_index if needed (0, 1, 2, etc.)
    detector = RigPreparationDetection(camera_index=1)
    detector.run()

if __name__ == "__main__":
    main()
