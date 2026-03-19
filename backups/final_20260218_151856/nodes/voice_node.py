"""
Voice Node — Improved speech recognition, routes to Mistral AI
Listens continuously, sends voice commands to state manager
"""

import threading
import time
import queue

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[VOICE_NODE] speech_recognition not installed. Run: pip3 install SpeechRecognition")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[VOICE_NODE] pyaudio not installed. Run: pip3 install pyaudio")


class VoiceNode:
    def __init__(self, command_queue, face_recognition_node=None):
        self.command_queue         = command_queue
        self.face_recognition_node = face_recognition_node
        self.running               = False

        # Tuned for better recognition
        self.recognizer            = sr.Recognizer() if SR_AVAILABLE else None
        self.microphone            = None
        self.enabled               = SR_AVAILABLE and PYAUDIO_AVAILABLE

        # Sensitivity tuning
        if self.recognizer:
            self.recognizer.energy_threshold        = 300    # lower = more sensitive
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold          = 0.8   # shorter pause needed
            self.recognizer.phrase_threshold         = 0.3
            self.recognizer.non_speaking_duration    = 0.5

        # Wake word — optional, set to None to always listen
        self.wake_word   = None
        self.listening   = True
        self.last_spoken = 0
        self.cooldown    = 1.5  # seconds between processing

        print(f"[VOICE_NODE] Initialized — enabled: {self.enabled}")

    def _init_mic(self):
        """Initialize microphone with best available device"""
        if not self.enabled:
            return False
        try:
            # Try to find best mic
            mic_list = sr.Microphone.list_microphone_names()
            print(f"[VOICE_NODE] Available mics: {len(mic_list)}")

            # Prefer USB or built-in mic
            preferred_idx = None
            for i, name in enumerate(mic_list):
                n = name.lower()
                if any(w in n for w in ['usb', 'headset', 'built-in', 'default']):
                    preferred_idx = i
                    print(f"[VOICE_NODE] Preferred mic: [{i}] {name}")
                    break

            if preferred_idx is not None:
                self.microphone = sr.Microphone(device_index=preferred_idx)
            else:
                self.microphone = sr.Microphone()

            # Calibrate noise
            print("[VOICE_NODE] Calibrating for ambient noise (2s)...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"[VOICE_NODE] ✓ Calibrated. Energy threshold: {self.recognizer.energy_threshold:.0f}")
            return True

        except Exception as e:
            print(f"[VOICE_NODE] Mic init error: {e}")
            self.enabled = False
            return False

    def _listen_once(self):
        """Listen for one phrase and return text or None"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )

            # Try Google first (best accuracy), fall back to Sphinx
            try:
                text = self.recognizer.recognize_google(audio, language='en-US')
                print(f"[VOICE_NODE] Heard: '{text}'")
                return text.strip()
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"[VOICE_NODE] Google SR error: {e} — trying offline")
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    print(f"[VOICE_NODE] Sphinx heard: '{text}'")
                    return text.strip()
                except Exception:
                    return None

        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"[VOICE_NODE] Listen error: {e}")
            return None

    def _process_text(self, text):
        """Route recognized text to appropriate handler"""
        if not text:
            return

        now = time.time()
        if now - self.last_spoken < self.cooldown:
            return
        self.last_spoken = now

        text_lower = text.lower().strip()

        # Check for name introduction
        name_triggers = ['my name is', "i'm", 'i am', 'call me']
        for trigger in name_triggers:
            if trigger in text_lower:
                parts = text_lower.split(trigger, 1)
                if len(parts) > 1:
                    name = parts[1].strip().split()[0].capitalize()
                    if len(name) > 1:
                        print(f"[VOICE_NODE] Name detected: {name}")
                        self.command_queue.put({'type': 'SET_NAME', 'name': name})
                        return

        # Everything else goes to Mistral as USER_MESSAGE
        print(f"[VOICE_NODE] Routing to AI: '{text}'")
        self.command_queue.put({'type': 'USER_MESSAGE', 'text': text})

    def _listen_loop(self):
        """Continuous listening loop"""
        print("[VOICE_NODE] Listening continuously...")
        consecutive_errors = 0

        while self.running:
            if not self.listening:
                time.sleep(0.5)
                continue

            try:
                text = self._listen_once()
                if text:
                    consecutive_errors = 0
                    self._process_text(text)
                else:
                    time.sleep(0.1)

            except Exception as e:
                consecutive_errors += 1
                print(f"[VOICE_NODE] Error #{consecutive_errors}: {e}")
                if consecutive_errors > 10:
                    print("[VOICE_NODE] Too many errors, pausing 10s...")
                    time.sleep(10)
                    consecutive_errors = 0
                else:
                    time.sleep(1)

    def run(self):
        self.running = True
        print("[VOICE_NODE] Starting...")

        if not self.enabled:
            print("[VOICE_NODE] Disabled — no speech recognition")
            print("[VOICE_NODE] Install: pip3 install SpeechRecognition pyaudio")
            while self.running:
                time.sleep(1)
            return

        if not self._init_mic():
            print("[VOICE_NODE] Could not initialize microphone")
            while self.running:
                time.sleep(1)
            return

        self._listen_loop()

    def shutdown(self):
        self.running  = False
        self.listening = False
        print("[VOICE_NODE] Stopped")
