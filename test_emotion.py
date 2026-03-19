from deepface import DeepFace
import cv2
import time

print("Testing emotion detection...")
cap = cv2.VideoCapture(0)
time.sleep(3)
ret, frame = cap.read()
cap.release()

if ret:
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    print("Your emotion:", result[0]['dominant_emotion'])
else:
    print("Camera error")
