import torch
import cv2
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import numpy
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("Ultralytics YOLO:", YOLO)
print("FaceNet:", InceptionResnetV1)
print("NumPy:", numpy.__version__)