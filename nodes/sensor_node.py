"""
Sensor Node — uses FrameBroadcaster to send frames to all subscribers
"""
import cv2
import time
import threading


class SensorNode:
    def __init__(self, command_queue, use_camera=True, frame_queue=None):
        self.command_queue    = command_queue
        self.frame_queue      = frame_queue   # legacy fallback
        self.use_camera       = use_camera
        self.running          = False
        self.cap              = None
        self.gui_node         = None
        self.broadcaster      = None          # set by main.py
        self.face_was_present = False

        # Face overlays from face_recognition_node
        self._face_overlays  = []
        self._overlay_lock   = threading.Lock()
        self._last_overlay_t = 0.0

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if use_camera:
            self._init_camera()

    def _init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("[SENSOR_NODE] ✓ Camera initialized")
        else:
            print("[SENSOR_NODE] ⚠️ Camera not available")

    def set_face_overlays(self, overlays: list):
        with self._overlay_lock:
            self._face_overlays  = overlays
            self._last_overlay_t = time.time()

    def run(self):
        self.running = True
        if not self.cap or not self.cap.isOpened():
            print("[SENSOR_NODE] Camera not available")
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

            # ── Draw face rectangles on display frame ────────────────
            display = frame.copy()
            with self._overlay_lock:
                overlays   = list(self._face_overlays)
                overlay_age = time.time() - self._last_overlay_t

            if overlays and overlay_age < 3.0:
                for ov in overlays:
                    bbox  = ov.get('bbox')
                    name  = ov.get('name', '?')
                    known = ov.get('known', False)
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    color = (0, 220, 80) if known else (0, 80, 220)
                    cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(
                        name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    ly = max(0, y1-th-8)
                    cv2.rectangle(display, (x1,ly), (x1+tw+8, y1), color, -1)
                    cv2.putText(display, name, (x1+4, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255,255,255), 1, cv2.LINE_AA)
            elif frame_count % 3 == 0:
                # Haar fallback — yellow while recognizing
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 1.1, 5, minSize=(60,60))
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x,y), (x+w,y+h), (0,220,220), 2)
                    cv2.putText(display, 'Recognizing...',
                                (x, max(0,y-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0,220,220), 1, cv2.LINE_AA)

            # ── Send annotated frame to GUI ───────────────────────────
            if self.gui_node and hasattr(self.gui_node, 'update_camera_frame'):
                self.gui_node.update_camera_frame(display)

            # ── Broadcast raw frame to all subscribers ────────────────
            if self.broadcaster:
                self.broadcaster.publish(frame)
            elif self.frame_queue is not None:
                import queue as _q
                try:
                    self.frame_queue.put_nowait(frame)
                except _q.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except Exception:
                        pass

            # ── Face presence detection ───────────────────────────────
            if frame_count % 5 == 0:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 1.1, 5, minSize=(60,60))
                face_now = len(faces) > 0
                if face_now and not self.face_was_present:
                    self.command_queue.put({'type': 'FACE_DETECTED',
                                            'count': len(faces)})
                elif not face_now and self.face_was_present:
                    self.command_queue.put({'type': 'FACE_LOST'})
                self.face_was_present = face_now

            time.sleep(0.033)

        if self.cap:
            self.cap.release()

    def shutdown(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print("[SENSOR_NODE] Shutting down...")