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
                    embedding = facenet(img_tensor).numpy().flatten()
                embeddings.append(embedding)
            if embeddings:
                known_embeddings[person_dir] = np.mean(embeddings, axis=0)
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

# Webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Ensure it's connected or try a video file.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Detect faces
    results = yolo(frame)
    cropped_faces = []
    boxes = []
    for box in results[0].boxes:
        if int(box.cls) == 0:  # Person class (fine-tune for faces if needed)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            cropped_faces.append(face)
            boxes.append((x1, y1, x2, y2))

    # Recognize faces
    for face, (x1, y1, x2, y2) in zip(cropped_faces, boxes):
        face_tensor = preprocess_face(face)
        with torch.no_grad():
            embedding = facenet(face_tensor).numpy().flatten()
        identity, dist = recognize_face(embedding, known_embeddings)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{identity} ({dist:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()