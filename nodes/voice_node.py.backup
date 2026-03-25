"""
Voice Node — Speech-to-text that fills GUI message input
Uses default microphone (no complex testing)
"""

import threading
import time
import queue

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[VOICE_NODE] Run: pip3 install SpeechRecognition")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[VOICE_NODE] Run: pip3 install pyaudio")


class VoiceNode:
    def __init__(self, command_queue, face_recognition_node=None, wake_word_mode=False):
        self.command_queue         = command_queue
        self.face_recognition_node = face_recognition_node
        self.wake_word_mode        = wake_word_mode
        self.running               = False

        self.recognizer  = sr.Recognizer() if SR_AVAILABLE else None
        self.microphone  = None
        self.enabled     = SR_AVAILABLE and PYAUDIO_AVAILABLE

        self._active       = not wake_word_mode
        self._active_until = 0
        self._active_secs  = 10
        self._active_lock  = threading.Lock()

        if self.recognizer:
            self.recognizer.energy_threshold         = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold          = 0.8
            self.recognizer.phrase_threshold         = 0.3
            self.recognizer.non_speaking_duration    = 0.5

        self.last_spoken = 0
        self.cooldown    = 1.2
        self.gui_node = None

        mode = "wake-word" if wake_word_mode else "continuous"
        print(f"[VOICE_NODE] Initialized — {mode} mode | enabled: {self.enabled}")

    def activate_wake_word(self):
        with self._active_lock:
            self._active       = True
            self._active_until = time.time() + self._active_secs
        print(f"[VOICE_NODE] 🎤 Activated for {self._active_secs}s")

    def is_active(self):
        with self._active_lock:
            if not self.wake_word_mode:
                return True
            if self._active and time.time() < self._active_until:
                return True
            if self._active and time.time() >= self._active_until:
                self._active = False
                print("[VOICE_NODE] ⏹ Deactivated")
            return False

    def _init_mic(self):
        if not self.enabled:
            return False
        try:
            print("[VOICE_NODE] Using default microphone")
            self.microphone = sr.Microphone()
            
            print("[VOICE_NODE] Calibrating (2s, stay quiet)...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
            threshold = self.recognizer.energy_threshold
            print(f"[VOICE_NODE] ✓ Ready — threshold: {threshold:.0f}")
            return True

        except Exception as e:
            print(f"[VOICE_NODE] Mic init error: {e}")
            self.enabled = False
            return False

    def _listen_once(self):
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=4, phrase_time_limit=12)
            
            try:
                text = self.recognizer.recognize_google(audio, language='en-US')
                print(f"[VOICE_NODE] Heard: '{text}'")
                return text.strip()
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"[VOICE_NODE] Google error: {e}")
                return None

        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"[VOICE_NODE] Listen error: {e}")
            return None

    def _process(self, text):
        if not text:
            return
        
        now = time.time()
        if now - self.last_spoken < self.cooldown:
            return
        self.last_spoken = now

        low = text.lower().strip()
        if low in ('hey aria', 'aria', 'hey', 'area'):
            return

        if self.gui_node:
            print(f"[VOICE_NODE] → GUI: '{text}'")
            self.gui_node.set_message_input(text)
        else:
            print(f"[VOICE_NODE] → Direct: '{text}'")
            self.command_queue.put({'type': 'USER_MESSAGE', 'text': text})

    def _loop(self):
        errors = 0
        
        while self.running:
            if not self.is_active():
                time.sleep(0.2)
                continue
            
            try:
                text = self._listen_once()
                if text:
                    errors = 0
                    self._process(text)
                else:
                    time.sleep(0.05)
                    
            except Exception as e:
                errors += 1
                print(f"[VOICE_NODE] Error #{errors}: {e}")
                if errors > 10:
                    print("[VOICE_NODE] Too many errors, pausing 10s...")
                    time.sleep(10)
                    errors = 0
                else:
                    time.sleep(1)

    def run(self):
        self.running = True
        print("[VOICE_NODE] Starting...")

        if not self.enabled:
            print("[VOICE_NODE] Disabled")
            while self.running:
                time.sleep(1)
            return

        if not self._init_mic():
            print("[VOICE_NODE] Mic failed")
            while self.running:
                time.sleep(1)
            return

        if self.wake_word_mode:
            print("[VOICE_NODE] Waiting for wake word...")
        else:
            print("[VOICE_NODE] 🎤 Listening continuously — speak clearly!")

        self._loop()

    def shutdown(self):
        self.running = False
        self._active = False
        print("[VOICE_NODE] Stopped")
