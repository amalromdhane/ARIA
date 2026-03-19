import cv2
import numpy as np
import time

print("Testing emotion detection with OpenCV...")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    print("✅ Face detected! (Emotion analysis needs CNN model)")
    print("Install FER properly for full emotion detection")
else:
    print("No face detected")
