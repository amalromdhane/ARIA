"""
State Manager — Full priority system
Fixed: USER_MESSAGE checks current_visitor_id to save name
"""

import queue
import time


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

        self.in_conversation       = False
        self.last_user_interaction = 0
        self.conversation_timeout  = 45

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
        """Helper — récupère current_visitor_id depuis face_recognition_node."""
        if self.face_recognition_node:
            return getattr(self.face_recognition_node, 'current_visitor_id', None)
        return None

    def _save_visitor_name(self, name: str) -> bool:
        """
        Sauvegarde le nom du visiteur courant.
        Retourne True si sauvegardé, False sinon.
        """
        visitor_id = self._get_current_visitor_id()
        if visitor_id and self.face_recognition_node:
            self.face_recognition_node.set_visitor_name(visitor_id, name)
            print(f"[STATE_MANAGER] ✓ Name saved: '{name}' → {visitor_id}")
            return True
        print(f"[STATE_MANAGER] ✗ No pending visitor for name: '{name}'")
        return False

    def process_commands(self):
        while self.running:
            try:
                self._check_conversation_timeout()

                if not self.command_queue.empty():
                    cmd      = self.command_queue.get_nowait()
                    cmd_type = cmd.get('type', '')

                    # ── State change ──────────────────────────────────
                    if cmd_type == 'CHANGE_STATE':
                        self.change_state(cmd.get('state', 'IDLE'))

                    # ── Wake word ─────────────────────────────────────
                    elif cmd_type == 'WAKE_WORD_DETECTED':
                        self.change_state('LISTENING')
                        if self.voice_node:
                            self.voice_node.activate_wake_word()

                    # ── SET_NAME (from web GUI "Your Name" field) ─────
                    elif cmd_type == 'SET_NAME':
                        name = cmd.get('name', '').strip()
                        if not name:
                            continue
                        self._mark_conversation_active()
                        self.change_state('GREETING')
                        saved = self._save_visitor_name(name)
                        if self.ai_agent_node:
                            self.ai_agent_node.set_context(name=name)
                        if self.gui_node:
                            self.gui_node.update_visitor_info(name=name)
                        if self.ai_agent_node and self.ai_agent_node.enabled:
                            self.ai_agent_node.ask(
                                f"The visitor just told me their name is {name}. "
                                f"Greet them warmly by name in one sentence."
                            )
                        else:
                            msg = f"Nice to meet you, {name}! How may I assist you?"
                            if self.sound_emotion_queue:
                                self.sound_emotion_queue.put({'type': 'SPEAK', 'text': msg})
                            if self.gui_node:
                                self.gui_node.add_chat_message('ROBOT', msg)

                    # ── USER_MESSAGE (from GUI text input) ───────────
                    elif cmd_type == 'USER_MESSAGE':
                        text = cmd.get('text', '').strip()
                        if not text:
                            continue

                        # KEY FIX: si un visiteur inconnu est en attente de nom
                        # → traiter ce message comme son nom
                        visitor_id = self._get_current_visitor_id()
                        if visitor_id:
                            print(f"[STATE_MANAGER] Treating message as name: '{text}'")
                            self._mark_conversation_active()
                            self.change_state('GREETING')
                            self._save_visitor_name(text)
                            if self.ai_agent_node:
                                self.ai_agent_node.set_context(name=text)
                            if self.gui_node:
                                self.gui_node.update_visitor_info(name=text)
                            if self.ai_agent_node and self.ai_agent_node.enabled:
                                self.ai_agent_node.ask(
                                    f"The visitor just told me their name is {text}. "
                                    f"Greet them warmly by name in one sentence."
                                )
                            else:
                                msg = f"Nice to meet you, {text}! How may I assist you?"
                                if self.sound_emotion_queue:
                                    self.sound_emotion_queue.put({'type': 'SPEAK', 'text': msg})
                                if self.gui_node:
                                    self.gui_node.add_chat_message('ROBOT', msg)
                            continue

                        # Message normal — passer à l'AI
                        self._mark_conversation_active()
                        self.change_state('THINKING')
                        if self.ai_agent_node and self.ai_agent_node.enabled:
                            self.ai_agent_node.ask(text)
                        else:
                            fallback = self._fallback(text)
                            if self.sound_emotion_queue:
                                self.sound_emotion_queue.put({'type': 'SPEAK', 'text': fallback})
                            if self.gui_node:
                                self.gui_node.add_chat_message('ROBOT', fallback)
                            self.change_state('GREETING')

                    # ── Visitor arrived ───────────────────────────────
                    elif cmd_type == 'VISITOR_ARRIVED':
                        if self.emotion_detection_node:
                            self.emotion_detection_node.notify_visitor_arrived()

                    # ── Visitor left ──────────────────────────────────
                    elif cmd_type == 'VISITOR_LEFT':
                        self.in_conversation = False
                        if self.emotion_detection_node:
                            self.emotion_detection_node.notify_visitor_left()
                            self.emotion_detection_node.set_conversation_active(False)

                    # ── Shutdown ──────────────────────────────────────
                    elif cmd_type == 'SHUTDOWN':
                        self.running = False

                    # ── Unknown commands — log but don't crash ────────
                    else:
                        pass   # Ignore silently (FACE_DETECTED etc.)

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