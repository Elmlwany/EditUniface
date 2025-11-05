"""
Ultra-Fast Rig Preparation Person Detection System
- Maximum FPS optimizations
- Minimal processing overhead
- Simplified but effective detection
"""

import cv2
import numpy as np
import time
from typing import Tuple, List

class UltraFastRigDetector:
    def __init__(self, camera_index: int = 1):
        self.camera_index = camera_index
        self.cap = None
        self.roi_points = []
        self.roi_selected = False
        
        # Ultra-fast settings
        self.process_every_n_frames = 5  # Only process every 5th frame
        self.frame_count = 0
        self.detection_size = (160, 120)  # Very small detection size
        self.last_detections = []
        
        # Simple background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=50)
        
        # FPS tracking
        self.fps_times = []
        self.current_fps = 0
        
        print("Ultra-Fast Detector initialized")
        print(f"Processing every {self.process_every_n_frames} frames")
        print(f"Detection resolution: {self.detection_size}")

    def initialize_camera(self) -> bool:
        """Initialize camera with maximum performance settings"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        
        # Optimize for speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)  # Try for higher FPS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Handle ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points = [(x, y)]
            self.roi_selected = False
            
        elif event == cv2.EVENT_LBUTTONUP and len(self.roi_points) == 1:
            self.roi_points.append((x, y))
            self.roi_selected = True
            print("ROI selected for ultra-fast detection!")

    def ultra_fast_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Ultra-fast person detection using motion + simple contours"""
        if not self.roi_selected:
            return self.last_detections
        
        # Only process every Nth frame
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_detections
        
        # Get ROI bounds
        x1, y1 = self.roi_points[0]
        x2, y2 = self.roi_points[1]
        roi_x1, roi_y1 = min(x1, x2), min(y1, y2)
        roi_x2, roi_y2 = max(x1, x2), max(y1, y2)
        
        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return []
        
        # Resize to very small size for ultra-fast processing
        small_roi = cv2.resize(roi, self.detection_size)
        
        # Background subtraction for motion detection
        fg_mask = self.bg_subtractor.apply(small_roi)
        
        # Simple morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area (person-like size)
            if 100 < area < 5000:  # Adjusted for small detection size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Scale back to original coordinates
                scale_x = (roi_x2 - roi_x1) / self.detection_size[0]
                scale_y = (roi_y2 - roi_y1) / self.detection_size[1]
                
                orig_x = int(x * scale_x) + roi_x1
                orig_y = int(y * scale_y) + roi_y1
                orig_w = int(w * scale_x)
                orig_h = int(h * scale_y)
                
                # Basic person aspect ratio check
                aspect_ratio = orig_h / orig_w if orig_w > 0 else 0
                if 1.2 < aspect_ratio < 3.0:  # Person-like aspect ratio
                    detections.append((orig_x, orig_y, orig_w, orig_h))
        
        self.last_detections = detections
        return detections

    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.fps_times.append(current_time)
        
        # Keep only last second of measurements
        self.fps_times = [t for t in self.fps_times if current_time - t < 1.0]
        self.current_fps = len(self.fps_times)

    def draw_ultra_fast_interface(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw minimal interface for maximum speed"""
        # Draw ROI
        if self.roi_selected and len(self.roi_points) == 2:
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            
            # Red ROI rectangle
            cv2.rectangle(frame, (min(x1, x2), min(y1, y2)), 
                         (max(x1, x2), max(y1, y2)), (0, 0, 255), 2)
        
        # Draw detections
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Minimal status display
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if not self.roi_selected:
            cv2.putText(frame, "Click and drag to select area", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"Detected: {len(detections)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def run(self):
        """Ultra-fast main loop"""
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return
        
        cv2.namedWindow('Ultra-Fast Rig Detection', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Ultra-Fast Rig Detection', self.mouse_callback)
        
        print("\n=== ULTRA-FAST RIG DETECTION ===")
        print("Optimized for maximum FPS!")
        print("Click and drag to select detection area")
        print("Press 'q' to quit, 'r' to reset ROI")
        
        detections = []
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Ultra-fast detection
                detections = self.ultra_fast_detect(frame)
                
                # Minimal drawing
                display_frame = self.draw_ultra_fast_interface(frame, detections)
                
                # Show frame
                cv2.imshow('Ultra-Fast Rig Detection', display_frame)
                
                # Update FPS
                self.update_fps()
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.roi_points = []
                    self.roi_selected = False
                    self.last_detections = []
                    print("ROI reset")
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"Final average FPS: {self.current_fps:.1f}")

def main():
    detector = UltraFastRigDetector(camera_index=1)
    detector.run()

if __name__ == "__main__":
    main()
