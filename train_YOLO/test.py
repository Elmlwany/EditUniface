import cv2
import numpy as np
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import os
from glob import glob
import pickle

# Preprocessing for FaceNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_face(face):
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_tensor = preprocess(face_rgb).unsqueeze(0)
    return face_tensor

# Load models
yolo = YOLO("yolov8n.pt")  # Auto-downloads if not present
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Load or generate known embeddings
known_embeddings = {}
known_faces_dir = r"C:\SW\Camera attendence - 2\known_faces"  # Use raw string for Windows path
embeddings_file = "known_embeddings.pkl"

if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        known_embeddings = pickle.load(f)
else:
    if not os.path.exists(known_faces_dir):
        raise FileNotFoundError("Known faces directory not found at: " + known_faces_dir)
    for person_dir in os.listdir(known_faces_dir):
        person_path = os.path.join(known_faces_dir, person_dir)
        if os.path.isdir(person_path):
            embeddings = []
            for img_path in glob(os.path.join(person_path, "*.jpg")):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                img_tensor = preprocess_face(img)
                with torch.no_grad():
                    embedding = facenet(img_tensor).detach().numpy().flatten()
                embeddings.append(embedding)
            if embeddings:
                known_embeddings[person_dir] = np.mean(embeddings, axis=0)
        else:
            print(f"Warning: {person_path} is not a directory")
    if not known_embeddings:
        raise ValueError("No valid embeddings generated. Check known_faces directory for valid images.")
    with open(embeddings_file, "wb") as f:
        pickle.dump(known_embeddings, f)

def recognize_face(embedding, known_embeddings, threshold=0.8):
    min_dist = float("inf")
    identity = "Unknown"
    for name, known_emb in known_embeddings.items():
        dist = np.linalg.norm(embedding - known_emb)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            identity = name
    return identity, min_dist

# Test with a single image
img = cv2.imread("../test_images/abdelrahman.jpg")
if img is None:
    raise FileNotFoundError("Could not load test_image.jpg. Ensure it exists in the working directory.")

# Detect faces
results = yolo(img)
cropped_faces = []
boxes = []
for box in results[0].boxes:
    if int(box.cls) == 0:  # Person class
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            print(f"Warning: Empty face crop at box ({x1}, {y1}, {x2}, {y2})")
            continue
        cropped_faces.append(face)
        boxes.append((x1, y1, x2, y2))

# Save cropped faces for debugging
for i, face in enumerate(cropped_faces):
    cv2.imwrite(f"cropped_face_{i}.jpg", face)
    print(f"Saved cropped face to cropped_face_{i}.jpg")

# Recognize faces
for face, (x1, y1, x2, y2) in zip(cropped_faces, boxes):
    face_tensor = preprocess_face(face)
    face_tensor = face_tensor.to(torch.float32)  # Ensure correct dtype
    with torch.no_grad():
        embedding = facenet(face_tensor).detach().numpy().flatten()
    identity, dist = recognize_face(embedding, known_embeddings)
    print(f"Detected: {identity} (Distance: {dist:.2f})")
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{identity} ({dist:.2f})", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite("recognized_faces.jpg", img)
cv2.imshow("Face Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()