"""
State Manager — Fixed version v3
=================================
FIXES (on top of previous version):
1. Added _normalize_preference() — maps partial/truncated STT results like
   'Hist', 'sci', 'tech' to canonical preference names so set_visitor_preference()
   always receives a valid keyword and returns True.
2. Added early-exit guard in USER_MESSAGE preference path — if the text arrives
   while TTS is still playing (time < _preference_tts_done_at), it is discarded
   immediately without even checking _expecting_preference, preventing Google's
   partial 'Hist' result from being consumed before the question finishes.
3. SET_NAME now reads name_display (romanized English) from the command and uses
   it for all spoken greetings, AI context, and GUI updates — while keeping the
   original Arabic name for face DB storage and matching.
"""

import queue
import time


# ── Preference keyword normalization map ──────────────────────────────────────
PREF_KEYWORDS = {
    'science': ['science', 'sci', 'scien', 'Sayın'],
    'sport':   ['sport', 'sports', 'spor'],
    'history': ['history', 'hist', 'historic', 'historical', 'histo'],
    'tech':    ['tech', 'technology', 'technologie', 'techn'],
    'art':     ['art', 'arts'],
}


class StateManagerNode:
    def __init__(self, state_queue, emotion_queue, command_queue, sound_emotion_queue=None):
        self.state_queue          = state_queue
        self.emotion_queue        = emotion_queue
        self.command_queue        = command_queue
        self.sound_emotion_queue  = sound_emotion_queue
        self.current_state        = 'IDLE'
        self.running              = False

        self.face_recognition_node  = None
        self.ai_agent_node          = None
        self.gui_node               = None
        self.emotion_detection_node = None
        self.voice_node             = None
        self.sound_node             = None

        self.in_conversation       = False
        self.last_user_interaction = 0
        self.conversation_timeout  = 45

        self._expecting_name       = False
        self._expecting_preference = False
        self._preference_vid       = None
        self._preference_asked_at  = 0
        self._preference_tts_done_at = 0

        self.state_emotions = {
            'IDLE':      'NEUTRAL',
            'GREETING':  'HAPPY',
            'LISTENING': 'ATTENTIVE',
            'THINKING':  'CONFUSED',
            'HELPING':   'EXCITED',
            'FAREWELL':  'HAPPY',
            'ERROR':     'SAD',
            'BUSY':      'TIRED',
            'SURPRISED': 'SURPRISED',
            'ANGRY':     'ANGRY',
        }

        print("[STATE_MANAGER] Initialized")

    # ── helpers ───────────────────────────────────────────────────────

    def change_state(self, new_state):
        if new_state not in self.state_emotions:
            return
        self.current_state = new_state
        emotion = self.state_emotions[new_state]
        try: self.state_queue.put_nowait(new_state)
        except queue.Full: pass
        try: self.emotion_queue.put_nowait(emotion)
        except queue.Full: pass
        print(f"[STATE_MANAGER] → {new_state} ({emotion})")

    def _mark_conversation_active(self):
        self.in_conversation       = True
        self.last_user_interaction = time.time()
        if self.emotion_detection_node:
            self.emotion_detection_node.set_conversation_active(True)

    def _check_conversation_timeout(self):
        if self.in_conversation:
            if time.time() - self.last_user_interaction > self.conversation_timeout:
                self.in_conversation = False
                if self.emotion_detection_node:
                    self.emotion_detection_node.set_conversation_active(False)
                print("[STATE_MANAGER] Conversation timed out")

    def _get_current_visitor_id(self):
        if self.face_recognition_node:
            return getattr(self.face_recognition_node, 'current_visitor_id', None)
        return None

    def _save_visitor_name(self, name: str) -> bool:
        fr = self.face_recognition_node
        if fr is None:
            return False
        if hasattr(fr, 'is_waiting_for_name'):
            result = fr.set_visitor_name(name)
            if result:
                print(f"[STATE_MANAGER] ✓ Name saved: '{name}'")
            else:
                print(f"[STATE_MANAGER] ✗ No pending visitor for name: '{name}'")
            return result
        visitor_id = self._get_current_visitor_id()
        if visitor_id:
            fr.set_visitor_name(visitor_id, name)
            print(f"[STATE_MANAGER] ✓ Name saved: '{name}' → {visitor_id}")
            return True
        print(f"[STATE_MANAGER] ✗ No pending visitor for name: '{name}'")
        return False

    def _is_unknown_visitor_pending(self):
        if self._expecting_name:
            return True
        fr = self.face_recognition_node
        if fr is None:
            return False
        if hasattr(fr, 'is_waiting_for_name'):
            return fr.is_waiting_for_name()
        visitor_id = getattr(fr, 'current_visitor_id', None)
        if visitor_id is None:
            return False
        name = fr.visitors.get(visitor_id, {}).get('name', '')
        return name.startswith('Visitor_')

    def _speak(self, text: str, is_preference_question: bool = False):
        """Speak text and notify voice_node to mute mic for the duration."""
        if not text or not text.strip():
            return
        if self.sound_emotion_queue:
            self.sound_emotion_queue.put({'type': 'SPEAK', 'text': text})
        if self.gui_node:
            self.gui_node.add_chat_message('ROBOT', text)
        word_count = len(text.split())
        estimated_duration = 0.5 + word_count * 0.45 + 1.0
        done_at = time.time() + estimated_duration
        vn = self.voice_node
        if vn and hasattr(vn, '_robot_speaking_until'):
            vn._robot_speaking_until = done_at
        if is_preference_question:
            self._preference_tts_done_at = done_at
            print(f"[STATE_MANAGER] Preference TTS estimated done in {estimated_duration:.1f}s")

    # ── FIX 1 — preference keyword normalizer ─────────────────────────
    def _normalize_preference(self, text: str) -> str:
        t = text.lower().strip()
        for canonical, variants in PREF_KEYWORDS.items():
            for variant in variants:
                if t.startswith(variant) or variant.startswith(t):
                    print(f"[STATE_MANAGER] Preference normalized: '{text}' → '{canonical}'")
                    return canonical
        return t

    # ── main loop ─────────────────────────────────────────────────────

    def process_commands(self):
        while self.running:
            try:
                self._check_conversation_timeout()

                if not self.command_queue.empty():
                    cmd      = self.command_queue.get_nowait()
                    cmd_type = cmd.get('type', '')

                    if cmd_type == 'CHANGE_STATE':
                        self.change_state(cmd.get('state', 'IDLE'))

                    elif cmd_type == 'WAKE_WORD_DETECTED':
                        self.change_state('LISTENING')
                        if self.voice_node:
                            self.voice_node.activate_wake_word()

                    elif cmd_type == 'VISITOR_ARRIVED':
                        fr = self.face_recognition_node
                        if fr and hasattr(fr, 'is_waiting_for_name') and fr.is_waiting_for_name():
                            self._expecting_name = True
                            print("[STATE_MANAGER] Expecting name input")
                        if self.emotion_detection_node:
                            self.emotion_detection_node.notify_visitor_arrived()

                    elif cmd_type == 'SET_NAME':
                        name = cmd.get('name', '').strip()
                        if not name:
                            continue

                        # FIX 3: read romanized display name — fallback to original
                        # if voice_node didn't supply it (e.g. typed Latin name)
                        name_display = cmd.get('name_display', name).strip()

                        self._expecting_name = False

                        if self.voice_node and hasattr(self.voice_node, 'cancel_pending_name'):
                            self.voice_node.cancel_pending_name()

                        if self.sound_node and hasattr(self.sound_node, 'cancel_pending_tts'):
                            self.sound_node.cancel_pending_tts()

                        self._mark_conversation_active()
                        self.change_state('GREETING')

                        # Save Arabic original to face DB for recognition matching
                        self._save_visitor_name(name)

                        # AI context and GUI always use the English display name
                        if self.ai_agent_node:
                            self.ai_agent_node.set_context(name=name_display)
                        if self.gui_node:
                            self.gui_node.update_visitor_info(name=name_display)

                        fr = self.face_recognition_node
                        will_ask_preference = False
                        preference_vid = None
                        if fr:
                            for v, data in fr.visitors.items():
                                # Match on Arabic original stored in face DB
                                if data.get('name', '').lower() == name.lower():
                                    if data.get('visits', 1) == 1 and not data.get('preference'):
                                        will_ask_preference = True
                                        preference_vid = v
                                    break

                        if will_ask_preference:
                            self._expecting_preference = True
                            self._preference_vid       = preference_vid
                            self._preference_asked_at  = time.time()
                            # Spoken greeting uses English display name
                            msg = (
                                f"Nice to meet you, {name_display}! "
                                f"what's your favorite topic — science, sport, history, tech, or art?"
                            )
                            if self.voice_node and hasattr(self.voice_node, 'set_preference_mode'):
                                self.voice_node.set_preference_mode(msg)
                            self._speak(msg, is_preference_question=True)
                            print(f"[STATE_MANAGER] Preference mode ON — waiting for topic from {name_display}")

                        elif self.ai_agent_node and self.ai_agent_node.enabled:
                            self.ai_agent_node.ask(
                                f"The visitor just told me their name is {name_display}. "
                                f"Greet them warmly by name in one sentence."
                            )
                        else:
                            msg = f"Nice to meet you, {name_display}! How may I assist you?"
                            self._speak(msg)

                    elif cmd_type == 'ASK_PREFERENCE':
                        if not self._expecting_preference:
                            name = cmd.get('name', 'you')
                            vid  = cmd.get('vid')
                            self._expecting_preference = True
                            self._preference_vid       = vid
                            self._preference_asked_at  = time.time()
                            msg = (
                                f"By the way {name}, what topic interests you most — "
                                f"science, sport, history, tech, or art?"
                            )
                            if self.voice_node and hasattr(self.voice_node, 'set_preference_mode'):
                                self.voice_node.set_preference_mode(msg)
                            self._speak(msg, is_preference_question=True)

                    elif cmd_type == 'SET_PREFERENCE':
                        raw_pref = cmd.get('preference', '').strip()
                        # FIX 1: normalize before passing to face_recognition
                        pref = self._normalize_preference(raw_pref)
                        self._expecting_preference = False
                        fr   = self.face_recognition_node
                        vid  = self._preference_vid
                        self._preference_vid = None

                        if fr and hasattr(fr, 'set_visitor_preference') and pref:
                            saved = fr.set_visitor_preference(pref, vid=vid)
                            if saved:
                                msg = f"Perfect! I'll share fun facts about {pref} on your next visit!"
                            else:
                                msg = "Got it! I'll find interesting facts for you next time."
                            self._speak(msg)

                    elif cmd_type == 'USER_MESSAGE':
                        text = cmd.get('text', '').strip()
                        if not text:
                            continue

                        # FIX 2: discard ANYTHING that arrives while TTS is still
                        # playing — catches Google's early partial 'Hist' result
                        now = time.time()
                        if now < self._preference_tts_done_at + 0.5:
                            remaining = (self._preference_tts_done_at + 0.5) - now
                            print(f"[STATE_MANAGER] TTS still playing "
                                  f"({remaining:.1f}s left), discarding: '{text}'")
                            continue

                        # Priority 1: preference answer
                        if self._expecting_preference:
                            print(f"[STATE_MANAGER] Preference answer received: '{text}'")
                            self._expecting_preference = False
                            if self.voice_node and hasattr(self.voice_node, '_preference_mode'):
                                self.voice_node._preference_mode = False
                            self.command_queue.put({
                                'type':       'SET_PREFERENCE',
                                'preference': text,
                                'vid':        self._preference_vid,
                            })
                            continue

                        # Priority 2: name answer (typed in GUI)
                        if self._is_unknown_visitor_pending():
                            print(f"[STATE_MANAGER] Name received via GUI: '{text}'")
                            self._expecting_name = False
                            if self.voice_node and hasattr(self.voice_node, 'cancel_pending_name'):
                                self.voice_node.cancel_pending_name()
                            self._mark_conversation_active()
                            self.change_state('GREETING')
                            # Typed name is always Latin — both fields are the same
                            self.command_queue.put({
                                'type':         'SET_NAME',
                                'name':         text,
                                'name_display': text,
                            })
                            continue

                        # Priority 3: normal conversation → AI
                        self._mark_conversation_active()
                        self.change_state('THINKING')
                        if self.ai_agent_node and self.ai_agent_node.enabled:
                            self.ai_agent_node.ask(text)
                        else:
                            fallback = self._fallback(text)
                            self._speak(fallback)
                            self.change_state('GREETING')

                    elif cmd_type == 'VISITOR_LEFT':
                        self._expecting_name         = False
                        self._expecting_preference   = False
                        self._preference_vid         = None
                        self._preference_asked_at    = 0
                        self._preference_tts_done_at = 0
                        self.in_conversation         = False
                        if self.voice_node and hasattr(self.voice_node, '_preference_mode'):
                            self.voice_node._preference_mode = False
                        if self.emotion_detection_node:
                            self.emotion_detection_node.notify_visitor_left()
                            self.emotion_detection_node.set_conversation_active(False)

                    elif cmd_type == 'ROBOT_SPEAK':
                        text = cmd.get('text', '')
                        if text:
                            self._speak(text)
                            text_low = text.lower()
                            if any(p in text_low for p in [
                                "what's your name", "your name", "introduce yourself",
                                "introduce yourselves",
                            ]):
                                self._expecting_name = True
                                print("[STATE_MANAGER] Expecting name input")

                    elif cmd_type == 'SHUTDOWN':
                        self.running = False

            except queue.Empty:
                pass
            except Exception as e:
                print(f"[STATE_MANAGER] Error: {e}")

            time.sleep(0.05)

    def _fallback(self, text):
        t = text.lower()
        if any(w in t for w in ['hello', 'hi', 'hey']):
            return "Hello! Welcome. How may I assist you?"
        if any(w in t for w in ['help', 'assist']):
            return "I am here to help. What do you need?"
        if any(w in t for w in ['bye', 'goodbye']):
            return "Goodbye! Have a great day."
        if any(w in t for w in ['thank']):
            return "You are very welcome!"
        if any(w in t for w in ['where', 'direction', 'room', 'floor']):
            return "Please speak to the front desk for directions."
        return "How may I assist you?"

    def run(self):
        self.running = True
        print("[STATE_MANAGER] Running...")
        self.process_commands()

    def shutdown(self):
        self.running = False
        print("[STATE_MANAGER] Stopped")