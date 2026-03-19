"""
Sensor Node - Shares frames with Emotion Detection
"""

import cv2
import queue
import time

class SensorNode:
    def __init__(self, command_queue, use_camera=True, frame_queue=None):
        self.command_queue = command_queue
        self.frame_queue = frame_queue  # NEW: Share frames
        self.use_camera = use_camera
        self.running = False
        self.cap = None
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if use_camera:
            self._init_camera()
    
    def _init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            print("[SENSOR_NODE] ✓ Camera initialized")
        else:
            print("[SENSOR_NODE] ⚠️ Camera not available")
    
    def run(self):
        self.running = True
        if not self.cap or not self.cap.isOpened():
            print("[SENSOR_NODE] Camera not available")
            return
        
        print("[SENSOR_NODE] Starting camera...")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # NEW: Share frame with emotion detection
            if self.frame_queue is not None:
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                print(f"[SENSOR_NODE] 👤 Face detected ({len(faces)} face(s))")
                self.command_queue.put({
                    'type': 'FACE_DETECTED',
                    'count': len(faces)
                })
            
            time.sleep(0.1)
        
        if self.cap:
            self.cap.release()
    
    def shutdown(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print("[SENSOR_NODE] Shutting down...")
