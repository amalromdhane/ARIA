"""
State Manager Node
Priority system: conversation > face detection > emotion comments
"""

import queue
import time


class StateManagerNode:
    def __init__(self, state_queue, emotion_queue, command_queue, sound_emotion_queue=None):
        self.state_queue         = state_queue
        self.emotion_queue       = emotion_queue
        self.command_queue       = command_queue
        self.sound_emotion_queue = sound_emotion_queue
        self.current_state       = 'IDLE'
        self.running             = False

        # Connected by main.py after init
        self.face_recognition_node = None
        self.ai_agent_node         = None
        self.gui_node              = None
        self.emotion_detection_node = None

        # ── Priority tracking ─────────────────────────────────────────
        # When user is actively conversing, block emotion interruptions
        self.in_conversation       = False
        self.last_user_interaction = 0
        self.conversation_timeout  = 45  # seconds of inactivity = conversation over

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

        print("[STATE_MANAGER] Node initialized")

    def change_state(self, new_state):
        if new_state not in self.state_emotions:
            return
        self.current_state = new_state
        emotion = self.state_emotions[new_state]
        try:
            self.state_queue.put_nowait(new_state)
        except queue.Full:
            pass
        try:
            self.emotion_queue.put_nowait(emotion)
        except queue.Full:
            pass
        print(f"[STATE_MANAGER] → {new_state} ({emotion})")

    def _mark_conversation_active(self):
        """Mark that user is actively conversing — blocks emotion interruptions"""
        self.in_conversation       = True
        self.last_user_interaction = time.time()
        # Tell emotion node to stay silent
        if self.emotion_detection_node:
            self.emotion_detection_node.set_conversation_active(True)

    def _check_conversation_timeout(self):
        """Check if conversation has gone idle"""
        if self.in_conversation:
            elapsed = time.time() - self.last_user_interaction
            if elapsed > self.conversation_timeout:
                self.in_conversation = False
                if self.emotion_detection_node:
                    self.emotion_detection_node.set_conversation_active(False)
                print("[STATE_MANAGER] Conversation timed out — emotion detection resumed")

    def process_commands(self):
        while self.running:
            try:
                # Check if conversation has timed out
                self._check_conversation_timeout()

                if not self.command_queue.empty():
                    cmd      = self.command_queue.get_nowait()
                    cmd_type = cmd.get('type', '')

                    # ── State change ──────────────────────────────────
                    if cmd_type == 'CHANGE_STATE':
                        self.change_state(cmd.get('state', 'IDLE'))

                    # ── Name from GUI or voice ────────────────────────
                    elif cmd_type == 'SET_NAME':
                        name = cmd.get('name', '').strip()
                        if not name:
                            continue

                        self._mark_conversation_active()
                        print(f"[STATE_MANAGER] Name: {name}")
                        self.change_state('GREETING')

                        # Save to face recognition DB
                        if self.face_recognition_node:
                            try:
                                self.face_recognition_node.set_visitor_name(name)
                            except Exception as e:
                                print(f"[STATE_MANAGER] Face rec error: {e}")

                        # Update AI context
                        if self.ai_agent_node:
                            self.ai_agent_node.set_context(name=name)

                        # Update GUI panel
                        if self.gui_node:
                            self.gui_node.update_visitor_info(name=name)

                        # Greet via Mistral
                        if self.ai_agent_node and self.ai_agent_node.enabled:
                            self.ai_agent_node.ask(
                                f"The visitor just introduced themselves as {name}. "
                                f"Greet them warmly by name in one sentence."
                            )
                        else:
                            msg = f"Nice to meet you, {name}!"
                            if self.sound_emotion_queue:
                                self.sound_emotion_queue.put({'type': 'SPEAK', 'text': msg})
                            if self.gui_node:
                                self.gui_node.add_chat_message('ROBOT', msg)

                    # ── User message (typed or voice) ─────────────────
                    elif cmd_type == 'USER_MESSAGE':
                        text = cmd.get('text', '').strip()
                        if not text:
                            continue

                        self._mark_conversation_active()
                        print(f"[STATE_MANAGER] Message: {text}")
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

                    # ── Shutdown ──────────────────────────────────────
                    elif cmd_type == 'SHUTDOWN':
                        self.running = False

                    else:
                        print(f"[STATE_MANAGER] Unknown: {cmd_type}")

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
