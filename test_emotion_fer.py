from fer import FER
import cv2
import time

print("Testing FER emotion detection...")

try:
    detector = FER(mtcnn=False)
    print("✅ Detector initialized")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

cap = cv2.VideoCapture(0)
time.sleep(3)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Camera error")
    exit()

print("Analyzing emotion...")

try:
    emotions = detector.detect_emotions(frame)
    
    if emotions:
        top_emotion, score = detector.top_emotion(frame)
        print(f"\n🎭 Dominant Emotion: {top_emotion.upper()}")
        print(f"📊 Confidence: {score:.2f}")
        
        print(f"\nAll emotions:")
        for emotion, confidence in emotions[0]['emotions'].items():
            bar = "█" * int(confidence / 5)
            print(f"  {emotion:12s}: {confidence:5.1f}% {bar}")
    else:
        print("No face detected in analysis")
        
except Exception as e:
    print(f"Error during detection: {e}")
    import traceback
    traceback.print_exc()
