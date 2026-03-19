"""
AI Agent Node - Mistral AI (Free Plan)
Handles ALL conversations: typed AND voice
"""

import threading
import time
import queue
import os


try:
    from mistralai import Mistral
    MISTRAL_OK = True
except ImportError:
    MISTRAL_OK = False
    print("[AI_AGENT] Run: pip3 install mistralai")


class AIAgentNode:
    def __init__(self, sound_emotion_queue, face_recognition_node=None):
        self.sound_queue           = sound_emotion_queue
        self.face_recognition_node = face_recognition_node
        self.gui_node              = None   # set by main.py
        self.state_manager         = None   # set by main.py
        self.running               = False

        self.req_queue       = queue.Queue()
        self.history         = []
        self.current_name    = None
        self.current_emotion = 'neutral'
        self.visit_count     = 0
        self.processing      = False  # prevent overlapping responses

        self.api_key = self._load_key()
        self.enabled = MISTRAL_OK and bool(self.api_key)

        if self.enabled:
            self.client = Mistral(api_key=self.api_key)
            print("[AI_AGENT] ✓ Mistral ready")
        else:
            self.client = None
            if not self.api_key:
                print("[AI_AGENT] No API key — create config/mistral_key.txt")
            else:
                print("[AI_AGENT] Mistral not installed")

        self.system_prompt = (
            "You are ARIA, a professional reception assistant robot at a corporate office. "
            "You are warm, helpful, and very concise. "
            "IMPORTANT RULES:\n"
            "1. Keep ALL responses to 1-2 sentences maximum.\n"
            "2. NEVER comment on the visitor's mood or emotion unless they bring it up.\n"
            "3. Focus ONLY on what the visitor is asking or saying.\n"
            "4. You can help with: directions, appointments, welcoming, general questions.\n"
            "5. Never say you are an AI or language model. You are ARIA.\n"
            "6. If asked your name, say 'I am ARIA, the reception assistant.'"
        )
        print("[AI_AGENT] Initialized")

    def _load_key(self):
        key = os.getenv('MISTRAL_API_KEY')
        if key:
            return key
        for path in ['config/mistral_key.txt', 'config/.env', '.env']:
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('MISTRAL_API_KEY='):
                            return line.split('=', 1)[1].strip()
                        if line and not line.startswith('#') and len(line) > 10:
                            return line
            except FileNotFoundError:
                continue
        return None

    def set_context(self, name=None, emotion=None, visits=None):
        if name is not None:    self.current_name    = name
        if emotion is not None: self.current_emotion = emotion
        if visits is not None:  self.visit_count     = visits
        # If name cleared, reset history for new visitor
        if name is None and visits == 0:
            self.history = []
            print("[AI_AGENT] Context reset for new visitor")

    def get_ai_response(self, user_input):
        """Get Mistral response synchronously"""
        if not self.enabled:
            resp = self._fallback(user_input)
            self._deliver(resp)
            return resp

        try:
            # Build context — but NEVER mention emotion unless asked
            ctx = []
            if self.current_name:
                ctx.append(f"Visitor name: {self.current_name}")
            if self.visit_count > 1:
                ctx.append(f"Visit number: {self.visit_count}")

            msgs = [{"role": "system", "content": self.system_prompt}]
            if ctx:
                msgs.append({"role": "system",
                             "content": "Context — " + ", ".join(ctx)})

            # Add conversation history
            for turn in self.history[-8:]:
                msgs.append({"role": "user",      "content": turn['u']})
                msgs.append({"role": "assistant",  "content": turn['a']})

            msgs.append({"role": "user", "content": user_input})

            print(f"[AI_AGENT] → Mistral: '{user_input[:60]}'")

            resp = self.client.chat.complete(
                model="mistral-small-latest",
                messages=msgs,
                max_tokens=150,
                temperature=0.7,
            )
            answer = resp.choices[0].message.content.strip()
            print(f"[AI_AGENT] ← Mistral: '{answer[:80]}'")

            self.history.append({'u': user_input, 'a': answer})
            if len(self.history) > 12:
                self.history.pop(0)

            self._deliver(answer)

            # Return to GREETING state after answering
            if self.state_manager:
                self.state_manager.change_state('GREETING')

            return answer

        except Exception as e:
            print(f"[AI_AGENT] Mistral error: {e}")
            err = "I'm having a brief issue. Please try again."
            self._deliver(err)
            return err

    def _deliver(self, text):
        """Speak + show in chat"""
        self.sound_queue.put({'type': 'SPEAK', 'text': text})
        if self.gui_node:
            self.gui_node.add_chat_message('ROBOT', text)

    def _fallback(self, text):
        t = text.lower()
        n = f", {self.current_name}" if self.current_name else ""
        if any(w in t for w in ['hello', 'hi', 'hey']):
            return f"Hello{n}! Welcome. How may I assist you today?"
        if any(w in t for w in ['help', 'assist']):
            return "I am here to help. What do you need?"
        if any(w in t for w in ['bye', 'goodbye']):
            return f"Goodbye{n}! Have a great day."
        if any(w in t for w in ['thank']):
            return "You are very welcome!"
        if any(w in t for w in ['where', 'direction', 'room', 'floor']):
            return "Please ask at the front desk for directions."
        if any(w in t for w in ['name', 'who are you', 'what are you']):
            return "I am ARIA, your reception assistant. How may I help?"
        return f"How may I assist you{n}?"

    def ask(self, text, context=None):
        """Queue a request — works for BOTH voice and typing"""
        if context:
            self.set_context(**context)
        self.req_queue.put(text)

    def _loop(self):
        """Process requests one at a time"""
        while self.running:
            try:
                text = self.req_queue.get(timeout=0.5)
                self.processing = True
                self.get_ai_response(text)
                self.processing = False
            except queue.Empty:
                pass
            except Exception as e:
                self.processing = False
                print(f"[AI_AGENT] Loop error: {e}")

    def run(self):
        self.running = True
        print("[AI_AGENT] Running...")
        self._loop()

    def stop(self):
        self.running = False
        print("[AI_AGENT] Stopped")
