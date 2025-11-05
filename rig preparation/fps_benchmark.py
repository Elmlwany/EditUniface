"""
FPS Benchmark Test
Quick test to measure camera FPS without detection
"""

import cv2
import time

def benchmark_camera(camera_index=1, duration=10):
    """Benchmark camera FPS"""
    print(f"Benchmarking camera {camera_index} for {duration} seconds...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    start_time = time.time()
    
    print("Press 'q' to stop early, or wait for automatic completion...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed >= duration:
            break
        
        # Display frame with FPS counter
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Camera FPS Benchmark', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate final FPS
    end_time = time.time()
    total_time = end_time - start_time
    final_fps = frame_count / total_time if total_time > 0 else 0
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nBenchmark Results:")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {final_fps:.2f}")
    print(f"Camera baseline: {final_fps:.2f} FPS")
    
    return final_fps

if __name__ == "__main__":
    benchmark_camera(camera_index=1, duration=10)
