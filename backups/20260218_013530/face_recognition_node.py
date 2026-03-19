"""
Face Recognition Node
- Greets ONCE per visit session
- No repeat greetings while same person is in frame
- Goodbye when person leaves, then fully resets
"""

import time
import os
import pickle
import queue


try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[FACE_REC] face_recognition not installed")

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

        self.db_path  = 'visitor_database.pkl'
        self.visitors = self._load_db()

        self.frame_counter = 0

        # ── Per-session state ─────────────────────────────────────────
        # Once we greet someone, we NEVER greet again until they leave
        self.current_visitor_id   = None
        self.current_visitor_name = None
        self.greeted_this_session = False   # KEY: blocks repeat greetings
        self.waiting_for_name     = False

        # Absence / departure tracking
        self.last_seen_face     = time.time()
        self.absence_threshold  = 10.0   # seconds without face = left
        self.goodbye_sent       = False
        self.face_present       = False

        # Recognition cooldown — only re-check identity every N seconds
        self.last_recognition   = 0
        self.recognition_interval = 5.0

        print(f"[FACE_REC] Node initialized")
        print(f"[FACE_REC] Known visitors: {len(self.visitors)}")

    def _load_db(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
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
        """Called when visitor types/says their name"""
        if self.current_visitor_id and self.current_visitor_id in self.visitors:
            self.visitors[self.current_visitor_id]['name'] = name
            self.current_visitor_name = name
            self._save_db()
            print(f"[FACE_REC] Name saved: {name}")
            gui = getattr(self.state_manager, 'gui_node', None)
            if gui:
                visits = self.visitors[self.current_visitor_id].get('visits', 1)
                gui.update_visitor_info(name=name, visits=f'Visit #{visits}')
        self.waiting_for_name = False

    def update_visitor_name(self, visitor_id, name):
        if visitor_id in self.visitors:
            self.visitors[visitor_id]['name'] = name
            self._save_db()

    def process_frame(self, frame):
        if not FACE_REC_AVAILABLE:
            return

        self.frame_counter += 1
        # Only process every 30 frames
        if self.frame_counter % 30 != 0:
            return

        try:
            import cv2
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            if not encodings:
                # No face in frame
                self.face_present = False
                now = time.time()
                if (not self.goodbye_sent
                        and self.current_visitor_id
                        and now - self.last_seen_face > self.absence_threshold):
                    self._send_goodbye()
                return

            # Face present
            self.face_present   = True
            self.last_seen_face = time.time()
            self.goodbye_sent   = False

            # ── Already greeted this person — do nothing ──────────────
            if self.greeted_this_session:
                return

            # ── Rate-limit recognition attempts ──────────────────────
            now = time.time()
            if now - self.last_recognition < self.recognition_interval:
                return
            self.last_recognition = now

            encoding = encodings[0]

            # Try to match known visitor
            matched_id = None
            if self.visitors:
                known_encodings = [v['encoding'] for v in self.visitors.values()]
                known_ids       = list(self.visitors.keys())
                distances       = face_recognition.face_distance(known_encodings, encoding)
                best_idx        = int(distances.argmin())
                if distances[best_idx] < 0.55:
                    matched_id = known_ids[best_idx]

            if matched_id:
                self._greet_returning(matched_id)
            else:
                self._greet_new(encoding)

            # Block any further greetings until this session ends
            self.greeted_this_session = True

        except Exception as e:
            print(f"[FACE_REC] Error: {e}")

    def _greet_new(self, encoding):
        visitor_id = f"Visitor_{len(self.visitors)+1:03d}"
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

        msg = "Hello! Welcome. I don't think we've met before. What's your name?"
        self._speak(msg)

        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.add_chat_message('ROBOT', msg)
            gui.update_visitor_info(name='New visitor', visits='1st visit')

        ai = getattr(self.state_manager, 'ai_agent_node', None)
        if ai:
            ai.set_context(name=None, visits=1)

        print(f"[FACE_REC] New visitor: {visitor_id}")

    def _greet_returning(self, visitor_id):
        visitor = self.visitors[visitor_id]
        visitor['visits']   += 1
        visitor['last_seen'] = time.strftime('%Y-%m-%d %H:%M')
        self._save_db()

        self.current_visitor_id   = visitor_id
        name   = visitor.get('name', visitor_id)
        visits = visitor['visits']

        # Use friendly name — skip raw IDs
        display = name if not name.startswith('Visitor_') else 'there'

        if visits == 2:
            msg = f"Welcome back, {display}! Great to see you again."
        elif visits == 3:
            msg = f"Hello again, {display}! This is your third visit."
        else:
            msg = f"Welcome back, {display}! Visit number {visits}."

        self._speak(msg)

        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.add_chat_message('ROBOT', msg)
            gui.update_visitor_info(name=display, visits=f'Visit #{visits}')

        ai = getattr(self.state_manager, 'ai_agent_node', None)
        if ai:
            ai.set_context(name=display, visits=visits)

        print(f"[FACE_REC] Returning: {display} (visit #{visits})")

    def _send_goodbye(self):
        self.goodbye_sent = True

        name = self.current_visitor_name or 'there'
        if name.startswith('Visitor_'):
            name = 'there'

        msg = f"Goodbye, {name}! Have a wonderful day."
        self._speak(msg)

        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.add_chat_message('ROBOT', msg)
            gui.update_visitor_info(name='—', visits='—', mood='—')

        # ── Full session reset ────────────────────────────────────────
        self.current_visitor_id   = None
        self.current_visitor_name = None
        self.greeted_this_session = False   # Ready for next person
        self.waiting_for_name     = False

        ai = getattr(self.state_manager, 'ai_agent_node', None)
        if ai:
            ai.set_context(name=None, visits=0)
            ai.history = []

        self.state_manager.change_state('IDLE')
        print("[FACE_REC] Session ended — ready for next visitor")

    def _speak(self, text):
        try:
            self.sound_emotion_queue.put({'type': 'SPEAK', 'text': text})
        except Exception:
            pass

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
            time.sleep(0.05)

    def stop(self):
        self.running = False
        print("[FACE_REC] Stopped")
