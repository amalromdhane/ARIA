"""
Face Recognition Node
- Responds quickly when face detected (low threshold)
- Greets ONCE per session, never repeats
- Proper DB management
- Goodbye when person leaves
"""

import time
import os
import pickle
import queue

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[FACE_REC] Run: pip3 install face-recognition")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class FaceRecognitionNode:
    def __init__(self, frame_queue, sound_emotion_queue, state_manager, main_system=None):
        self.frame_queue         = frame_queue
        self.sound_emotion_queue = sound_emotion_queue
        self.state_manager       = state_manager
        self.main_system         = main_system
        self.running             = False

        # Database
        self.db_path  = 'visitor_database.pkl'
        self.visitors = self._load_db()

        # Frontal cascade for quick frontal check
        self.frontal_cascade = None
        if CV2_AVAILABLE:
            path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.frontal_cascade = cv2.CascadeClassifier(path)

        self.frame_counter = 0

        # Session state
        self.current_visitor_id   = None
        self.current_visitor_name = None
        self.greeted_this_session = False
        self.waiting_for_name     = False

        # Frontal confirmation — require 3 consecutive frontal frames (fast!)
        self.frontal_count     = 0
        self.frontal_needed    = 3

        # Absence tracking
        self.last_seen_face    = time.time()
        self.absence_threshold = 8.0
        self.goodbye_sent      = False
        self.face_present      = False

        # Recognition rate limit
        self.last_recognition     = 0
        self.recognition_interval = 2.0
        self.match_threshold      = 0.52

        print(f"[FACE_REC] Initialized — {len(self.visitors)} known visitors")

    # ── Database ──────────────────────────────────────────────────────

    def _load_db(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    db = pickle.load(f)
                print(f"[FACE_REC] Loaded {len(db)} visitors from DB")
                return db
        except Exception as e:
            print(f"[FACE_REC] DB load error: {e}")
        return {}

    def _save_db(self):
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.visitors, f)
        except Exception as e:
            print(f"[FACE_REC] DB save error: {e}")

    def set_visitor_name(self, name):
        if self.current_visitor_id and self.current_visitor_id in self.visitors:
            self.visitors[self.current_visitor_id]['name'] = name
            self.current_visitor_name = name
            self._save_db()
            print(f"[FACE_REC] Name saved: '{name}'")

            gui = getattr(self.state_manager, 'gui_node', None)
            if gui:
                v = self.visitors[self.current_visitor_id].get('visits', 1)
                gui.update_visitor_info(name=name, visits=f'Visit #{v}')

            ai = getattr(self.state_manager, 'ai_agent_node', None)
            if ai:
                ai.set_context(name=name)

        self.waiting_for_name = False

    def update_visitor_name(self, visitor_id, name):
        if visitor_id in self.visitors:
            self.visitors[visitor_id]['name'] = name
            self._save_db()

    # ── Frontal detection ─────────────────────────────────────────────

    def _is_frontal(self, frame):
        if not CV2_AVAILABLE or self.frontal_cascade is None:
            return True
        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.frontal_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(70, 70))
            return len(faces) > 0
        except Exception:
            return False

    # ── Main processing ───────────────────────────────────────────────

    def process_frame(self, frame):
        if not FACE_REC_AVAILABLE or not CV2_AVAILABLE:
            return

        self.frame_counter += 1

        # Process every 5th frame for responsiveness
        if self.frame_counter % 5 != 0:
            return

        try:
            frontal = self._is_frontal(frame)

            if not frontal:
                self.frontal_count = max(0, self.frontal_count - 1)

                # Check departure
                now = time.time()
                if (self.face_present
                        and not self.goodbye_sent
                        and self.current_visitor_id
                        and now - self.last_seen_face > self.absence_threshold):
                    self.face_present = False
                    self._send_goodbye()
                return

            # Frontal face present
            self.face_present   = True
            self.last_seen_face = time.time()
            self.goodbye_sent   = False

            # Already greeted — nothing more to do until goodbye
            if self.greeted_this_session:
                return

            # Accumulate frontal count
            self.frontal_count += 1
            if self.frontal_count < self.frontal_needed:
                print(f"[FACE_REC] Confirming face... {self.frontal_count}/{self.frontal_needed}")
                return

            # Rate limit full recognition
            now = time.time()
            if now - self.last_recognition < self.recognition_interval:
                return
            self.last_recognition = now

            # Full recognition
            small     = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb       = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model='hog')
            encodings = face_recognition.face_encodings(rgb, locations)

            if not encodings:
                print("[FACE_REC] Face seen but no encoding — retrying")
                return

            encoding   = encodings[0]
            matched_id = self._find_match(encoding)

            if matched_id:
                self._greet_returning(matched_id, encoding)
            else:
                self._greet_new(encoding)

            # Lock: no more greetings until goodbye
            self.greeted_this_session = True
            self.frontal_count        = 0

            # Notify emotion detection node
            self.state_manager.command_queue.put({'type': 'VISITOR_ARRIVED'})

        except Exception as e:
            print(f"[FACE_REC] Error: {e}")

    def _find_match(self, encoding):
        if not self.visitors:
            return None
        try:
            known_encs = [v['encoding'] for v in self.visitors.values()
                          if 'encoding' in v and v['encoding'] is not None]
            known_ids  = [vid for vid, v in self.visitors.items()
                          if 'encoding' in v and v['encoding'] is not None]

            if not known_encs:
                return None

            distances = face_recognition.face_distance(known_encs, encoding)
            best_idx  = int(distances.argmin())
            best_dist = distances[best_idx]
            print(f"[FACE_REC] Best match: {best_dist:.3f} (threshold {self.match_threshold})")

            return known_ids[best_idx] if best_dist < self.match_threshold else None
        except Exception as e:
            print(f"[FACE_REC] Match error: {e}")
            return None

    # ── Greetings ─────────────────────────────────────────────────────

    def _greet_new(self, encoding):
        visitor_id = f"Visitor_{int(time.time())}"
        self.visitors[visitor_id] = {
            'encoding':   encoding,
            'name':       visitor_id,
            'visits':     1,
            'first_seen': time.strftime('%Y-%m-%d %H:%M'),
            'last_seen':  time.strftime('%Y-%m-%d %H:%M'),
        }
        self._save_db()

        self.current_visitor_id   = visitor_id
        self.current_visitor_name = None
        self.waiting_for_name     = True

        msg = "Hello! Welcome. I don't think we've met before — what's your name?"
        self._speak(msg)
        self._chat(msg)

        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.update_visitor_info(name='New visitor', visits='1st visit')

        ai = getattr(self.state_manager, 'ai_agent_node', None)
        if ai:
            ai.set_context(name=None, visits=1)

        self.state_manager.change_state('GREETING')
        print(f"[FACE_REC] New visitor: {visitor_id}")

    def _greet_returning(self, visitor_id, new_encoding=None):
        visitor = self.visitors[visitor_id]
        visitor['visits']   += 1
        visitor['last_seen'] = time.strftime('%Y-%m-%d %H:%M')
        if new_encoding is not None:
            visitor['encoding'] = new_encoding
        self._save_db()

        self.current_visitor_id   = visitor_id
        name    = visitor.get('name', visitor_id)
        visits  = visitor['visits']
        display = name if not name.startswith('Visitor_') else 'there'

        if visits == 2:
            msg = f"Welcome back, {display}! Great to see you again."
        elif visits == 3:
            msg = f"Hello again, {display}! Your third visit already."
        else:
            msg = f"Welcome back, {display}! Visit number {visits}."

        self._speak(msg)
        self._chat(msg)

        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.update_visitor_info(name=display, visits=f'Visit #{visits}')

        ai = getattr(self.state_manager, 'ai_agent_node', None)
        if ai:
            ai.set_context(name=display, visits=visits)

        self.state_manager.change_state('GREETING')
        print(f"[FACE_REC] Returning: {display} (#{visits})")

    def _send_goodbye(self):
        self.goodbye_sent = True

        name = self.current_visitor_name or ''
        if not name or name.startswith('Visitor_'):
            name = 'there'

        msg = f"Goodbye, {name}! Have a wonderful day."
        self._speak(msg)
        self._chat(msg)

        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.update_visitor_info(name='—', visits='—', mood='—')

        # Full reset for next visitor
        self.current_visitor_id   = None
        self.current_visitor_name = None
        self.greeted_this_session = False
        self.waiting_for_name     = False
        self.frontal_count        = 0

        ai = getattr(self.state_manager, 'ai_agent_node', None)
        if ai:
            ai.set_context(name=None, visits=0)
            ai.history = []

        self.state_manager.command_queue.put({'type': 'VISITOR_LEFT'})
        self.state_manager.change_state('IDLE')
        print("[FACE_REC] Session reset — ready for next visitor")

    def _speak(self, text):
        try:
            self.sound_emotion_queue.put({'type': 'SPEAK', 'text': text})
        except Exception:
            pass

    def _chat(self, text):
        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.add_chat_message('ROBOT', text)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def run(self):
        self.running = True
        print("[FACE_REC] Running...")
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    self.process_frame(frame)
            except Exception:
                pass
            time.sleep(0.02)

    def stop(self):
        self.running = False
        print("[FACE_REC] Stopped")
