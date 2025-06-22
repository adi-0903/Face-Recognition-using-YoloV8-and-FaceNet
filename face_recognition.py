import os
import cv2
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from deepface import DeepFace
import time

# Load YOLOv8 model
try:
    model = YOLO("detection/weights/best.pt")  
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")

# Load MTCNN and ResNet
try:
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    print("MTCNN and InceptionResnetV1 loaded.")
except Exception as e:
    print(f"Error loading MTCNN/InceptionResnetV1: {e}")

# Load known embeddings
def load_known_embeddings():
    try:
        with open('known_embeddings.pkl', 'rb') as f:
            known_embeddings = pickle.load(f)
            print("Known embeddings loaded.")
    except Exception as e:
        known_embeddings = {}
        print(f"Error loading embeddings: {e}")
    return known_embeddings

known_embeddings = load_known_embeddings()

# Compare embeddings
def compare_embeddings(embedding, known_embeddings, threshold=0.2):
    min_dist = float('inf')
    match = None
    for name, known_embedding in known_embeddings.items():
        dist = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
        if dist < min_dist:
            min_dist = dist
            match = name if dist < threshold else None
    return match

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

prev_time = 0
fps_limit = 5  # Limit to 5 FPS for smoother CPU usage

while True:
    current_time = time.time()
    if current_time - prev_time < 1 / fps_limit:
        continue
    prev_time = current_time

    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed.")
        break

    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        if box.conf[0] < 0.5:
            continue  # Skip low-confidence boxes

        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        face = frame[y1:y2, x1:x2]

        if face.shape[0] < 20 or face.shape[1] < 20:
            continue  # Skip too small faces

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        name = "Person"
        emotion = "N/A"

        try:
            face_resized = cv2.resize(face_rgb, (160, 160))
            mtcnn_box, _ = mtcnn.detect(face_resized)
            if mtcnn_box is not None:
                face_tensor = mtcnn(face_resized).squeeze().unsqueeze(0)
                embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
                match = compare_embeddings(embedding, known_embeddings)
                if match:
                    name = match
        except Exception as e:
            print(f"Embedding failed: {e}")

        try:
            face_bgr = cv2.resize(face, (224, 224))  # Required input size
            result = DeepFace.analyze(face_bgr, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except Exception as e:
            print(f"Emotion error: {e}")

        label = f"{name} | {emotion}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    cv2.imshow('Face & Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
