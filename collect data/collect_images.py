import cv2
import os
import time

# Create a folder to save images if it doesn't exist
output_folder = "face_recognition_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the 3 cameras (adjust indices based on your setup)
cam1 = cv2.VideoCapture(0)  # First camera
cam2 = cv2.VideoCapture(1)  # Second camera
cam3 = cv2.VideoCapture(2)  # Third camera

# Check if cameras opened successfully
if not (cam1.isOpened() and cam2.isOpened() and cam3.isOpened()):
    print("Error: One or more cameras failed to open.")
    exit()

# Number of captures to take (6 sets of 3 images = 18 total)
num_captures = 6
capture_count = 0

print("Starting image capture. Press 'q' to stop early if needed.")

while capture_count < num_captures:
    # Read frames from all three cameras
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    ret3, frame3 = cam3.read()

    if ret1 and ret2 and ret3:
        # Display the frames (optional, for real-time monitoring)
        cv2.imshow("Camera 1", frame1)
        cv2.imshow("Camera 2", frame2)
        cv2.imshow("Camera 3", frame3)

        # Save the frames to the folder
        capture_count += 1
        cv2.imwrite(f"{output_folder}/cam1_image{capture_count}.jpg", frame1)
        cv2.imwrite(f"{output_folder}/cam2_image{capture_count}.jpg", frame2)
        cv2.imwrite(f"{output_folder}/cam3_image{capture_count}.jpg", frame3)

        print(f"Captured set {capture_count} of {num_captures}")
        
        # Small delay between captures (e.g., 1 second)
        time.sleep(1)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close windows
cam1.release()
cam2.release()
cam3.release()
cv2.destroyAllWindows()

print(f"Done! {capture_count * 3} images saved to '{output_folder}' folder.")