"""
Sensor Node - Camera frames to frame_queue + face presence detection
No print spam, efficient frame sharing
"""

import cv2
import queue
import time


class SensorNode:
    def __init__(self, command_queue, use_camera=True, frame_queue=None):
        self.command_queue = command_queue
        self.frame_queue   = frame_queue
        self.use_camera    = use_camera
        self.running       = False
        self.cap           = None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Track face presence to avoid spamming command_queue
        self.face_was_present = False

        if use_camera:
            self._init_camera()

    def _init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("[SENSOR_NODE] ✓ Camera initialized")
        else:
            print("[SENSOR_NODE] ⚠️ Camera not available")

    def run(self):
        self.running = True

        if not self.cap or not self.cap.isOpened():
            print("[SENSOR_NODE] Camera not available — running without camera")
            while self.running:
                time.sleep(1)
            return

        print("[SENSOR_NODE] Starting camera...")
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1

            # ── Always push frame to queue (face_rec + emotion need it) ──
            if self.frame_queue is not None:
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest, push newest
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except Exception:
                        pass

            # ── Face presence check every 5 frames (not every frame) ──
            if frame_count % 5 == 0:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                face_now = len(faces) > 0

                # Only send command on state CHANGE — not every frame
                if face_now and not self.face_was_present:
                    self.command_queue.put({
                        'type': 'FACE_DETECTED',
                        'count': len(faces)
                    })
                elif not face_now and self.face_was_present:
                    self.command_queue.put({'type': 'FACE_LOST'})

                self.face_was_present = face_now

            time.sleep(0.033)  # ~30fps

        if self.cap:
            self.cap.release()

    def shutdown(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print("[SENSOR_NODE] Shutting down...")
