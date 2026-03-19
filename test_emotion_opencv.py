import cv2
import numpy as np
import time

print("Testing emotion with OpenCV DNN...")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load emotion model (if downloaded)
try:
    model = cv2.dnn.readNetFromTensorflow('emotion_model.h5')
    use_dnn = True
    print("✅ DNN model loaded")
except:
    use_dnn = False
    print("⚠️  No DNN model, using simple detection")

cap = cv2.VideoCapture(0)
time.sleep(3)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Camera error")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) > 0:
    print(f"✅ Face detected!")
    print("(For full emotion detection, need TensorFlow or ONNX model)")
else:
    print("No face detected")
