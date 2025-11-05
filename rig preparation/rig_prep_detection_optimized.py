"""
Optimized Rig Preparation Person Detection System
- High FPS performance optimizations
- Efficient person detection only in selected area
- Reduced processing overhead
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Optional

class OptimizedRigPrepDetector:
    def __init__(self, camera_index: int = 1):
        self.camera_index = camera_index
        self.cap = None
        self.roi_points = []
        self.roi_selected = False
        self.roi_mask = None
        
        # Performance optimizations
        self.frame_skip = 2  # Process every 2nd frame for detection
        self.frame_count = 0
        self.detection_scale = 0.5  # Scale down for detection (2x faster)
        self.display_scale = 1.0  # Keep display at full resolution
        
        # HOG detector with optimized parameters
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection parameters (optimized for speed)
        self.detection_params = {
            'winStride': (16, 16),  # Larger stride = faster but less accurate
            'padding': (8, 8),
            'scale': 1.05,  # Smaller scale step = faster
            'hitThreshold': 1.0,  # Correct parameter name
            'groupThreshold': 2  # Grouping threshold
        }
        
        print("Optimized Rig Preparation Detector initialized")
        print("Optimizations enabled:")
        print(f"  - Frame skipping: Process every {self.frame_skip} frames")
        print(f"  - Detection scale: {self.detection_scale}x")
        print(f"  - Optimized HOG parameters")

    def initialize_camera(self) -> bool:
        """Initialize camera with optimized settings"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better FPS
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {width}x{height} @ {fps} FPS")
        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points = [(x, y)]
            self.roi_selected = False
            print(f"ROI start point: ({x}, {y})")
            
        elif event == cv2.EVENT_LBUTTONUP and len(self.roi_points) == 1:
            self.roi_points.append((x, y))
            self.roi_selected = True
            self.create_roi_mask()
            print(f"ROI end point: ({x}, {y})")
            print("ROI selected! Person detection will be limited to this area.")

    def create_roi_mask(self):
        """Create mask for the selected ROI"""
        if len(self.roi_points) == 2:
            height, width = 480, 640  # Use camera resolution
            self.roi_mask = np.zeros((height, width), dtype=np.uint8)
            
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            
            # Create rectangular mask
            cv2.rectangle(self.roi_mask, (min(x1, x2), min(y1, y2)), 
                         (max(x1, x2), max(y1, y2)), 255, -1)

    def detect_persons_optimized(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Optimized person detection"""
        if not self.roi_selected:
            return []
        
        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return []
        
        # Scale down frame for detection
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, 
                                (int(width * self.detection_scale), 
                                 int(height * self.detection_scale)))
        
        # Apply ROI mask to small frame
        small_mask = cv2.resize(self.roi_mask, 
                               (int(width * self.detection_scale), 
                                int(height * self.detection_scale)))
        
        # Mask the frame
        masked_frame = cv2.bitwise_and(small_frame, small_frame, mask=small_mask)
        
        # Convert to grayscale for HOG
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect persons with optimized parameters
        rectangles, weights = self.hog.detectMultiScale(
            gray,
            winStride=self.detection_params['winStride'],
            padding=self.detection_params['padding'],
            scale=self.detection_params['scale'],
            hitThreshold=self.detection_params['hitThreshold'],
            groupThreshold=self.detection_params['groupThreshold']
        )
        
        # Scale rectangles back to original size
        scaled_rectangles = []
        for (x, y, w, h) in rectangles:
            x = int(x / self.detection_scale)
            y = int(y / self.detection_scale)
            w = int(w / self.detection_scale)
            h = int(h / self.detection_scale)
            scaled_rectangles.append((x, y, w, h))
        
        return scaled_rectangles

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def draw_interface(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw interface elements on frame"""
        display_frame = frame.copy()
        
        # Draw ROI
        if self.roi_selected and len(self.roi_points) == 2:
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            
            # Draw red rectangle for ROI
            cv2.rectangle(display_frame, (min(x1, x2), min(y1, y2)), 
                         (max(x1, x2), max(y1, y2)), (0, 0, 255), 2)
            
            # Fill ROI with semi-transparent red
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (min(x1, x2), min(y1, y2)), 
                         (max(x1, x2), max(y1, y2)), (0, 0, 255), -1)
            display_frame = cv2.addWeighted(display_frame, 0.9, overlay, 0.1, 0)
        
        # Draw person detections
        for (x, y, w, h) in detections:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, 'Person', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw status information
        status_y = 30
        
        # FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # ROI status
        roi_status = "ROI: Selected" if self.roi_selected else "ROI: Click and drag to select"
        cv2.putText(display_frame, roi_status, (10, status_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection count
        detection_text = f"Persons detected: {len(detections)}"
        cv2.putText(display_frame, detection_text, (10, status_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Performance info
        perf_text = f"Processing: {self.frame_skip}x skip, {self.detection_scale}x scale"
        cv2.putText(display_frame, perf_text, (10, status_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions
        if not self.roi_selected:
            instruction = "Instructions: Click and drag to select detection area"
            cv2.putText(display_frame, instruction, (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return display_frame

    def run(self):
        """Main execution loop"""
        if not self.initialize_camera():
            return
        
        cv2.namedWindow('Optimized Rig Preparation Detection', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Optimized Rig Preparation Detection', self.mouse_callback)
        
        print("\n=== OPTIMIZED RIG PREPARATION DETECTION ===")
        print("Instructions:")
        print("1. Click and drag to select the area of interest (ROI)")
        print("2. Person detection will only occur within the selected area")
        print("3. Press 'r' to reset ROI selection")
        print("4. Press 'q' to quit")
        print("5. Press 's' to toggle performance settings")
        print("\nOptimizations active for maximum FPS performance!")
        
        detections = []
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Update FPS
                self.update_fps()
                
                # Detect persons (optimized)
                current_detections = self.detect_persons_optimized(frame)
                if current_detections:  # Only update if we got new detections
                    detections = current_detections
                
                # Draw interface
                display_frame = self.draw_interface(frame, detections)
                
                # Show frame
                cv2.imshow('Optimized Rig Preparation Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.roi_points = []
                    self.roi_selected = False
                    self.roi_mask = None
                    detections = []
                    print("ROI reset")
                elif key == ord('s'):
                    self.toggle_performance_settings()
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

    def toggle_performance_settings(self):
        """Toggle between different performance settings"""
        if self.frame_skip == 2:
            self.frame_skip = 1
            self.detection_scale = 0.7
            print("Performance: Higher accuracy mode (slower)")
        elif self.frame_skip == 1:
            self.frame_skip = 3
            self.detection_scale = 0.3
            print("Performance: Maximum speed mode (lower accuracy)")
        else:
            self.frame_skip = 2
            self.detection_scale = 0.5
            print("Performance: Balanced mode (default)")

def main():
    """Main function"""
    print("Starting Optimized Rig Preparation Detection System...")
    
    detector = OptimizedRigPrepDetector(camera_index=1)
    detector.run()

if __name__ == "__main__":
    main()
