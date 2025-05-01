import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from dataset.models import InceptionResNetV2

def load_facenet_model(model_path=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate your architecture
    model = InceptionResNetV2(num_classes=1000)
    # Load trained weights if provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def detect_faces(image):
    # Use OpenCV's face detector as fallback
    # Ideally you'd use a more robust detector like MTCNN or DLib
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces_rects

def get_face_embedding(image, face_rect, model):
    x, y, w, h = face_rect
    face_img = image[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (160, 160))
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL and apply transformations
    face = Image.fromarray(face_rgb)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    tensor = transform(face).unsqueeze(0)
    device = next(model.parameters()).device  # Get model's device
    tensor = tensor.to(device)
    
    with torch.no_grad():
        embedding = model(tensor)
    
    return embedding.cpu().numpy()[0]
