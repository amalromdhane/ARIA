from deepface import DeepFace
import cv2
import time

cap = cv2.VideoCapture(0)
time.sleep(3)
ret, frame = cap.read()
cap.release()

result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
print("Emotion:", result[0]['dominant_emotion'])

