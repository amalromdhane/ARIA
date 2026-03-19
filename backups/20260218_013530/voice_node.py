# voice_node_fixed.py - Replace your current voice_node.py with this
"""
Voice Node — Wake Word Detection + Speech Recognition
Single microphone access point to avoid conflicts
"""

import threading
import time
import queue
import os
import sys
import ctypes
import subprocess

# Suppress ALSA at C level
try:
    _ALSA_ERR = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                  ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    ctypes.cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(
        _ALSA_ERR(lambda *a: None))
except Exception:
    pass

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


class VoiceNode:
    def __init__(self, command_queue, face_recognition_node=None, 
                 wake_word="hey aria", use_wake_word=True):
        self.command_queue = command_queue
        self.face_recognition_node = face_recognition_node
        self.wake_word = wake_word.lower()
        self.use_wake_word = use_wake_word
        self.running = False

        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None
        self.microphone = None
        self.enabled = SR_AVAILABLE and PYAUDIO_AVAILABLE

        # State
        self.is_active = not use_wake_word  # If no wake word, always active
        self.active_until = 0
        self.active_duration = 30  # Stay active for 30s after wake word

        # Settings
        if self.recognizer:
            self.recognizer.energy_threshold = 400
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3

        self.last_spoken = 0
        self.cooldown = 1.0

        mode = "wake-word" if use_wake_word else "continuous"
        print(f"[VOICE_NODE] Initialized — {mode} mode | Enabled: {self.enabled}")

    def _find_working_mic(self):
        """Test microphones and find first working one"""
        if not self.enabled:
            return None

        try:
            mic_list = sr.Microphone.list_microphone_names()
            print(f"[VOICE_NODE] Testing {len(mic_list)} mic(s)...")

            for i, name in enumerate(mic_list):
                n = name.lower()
                # Skip virtual/broken devices
                if any(bad in n for bad in ['front', 'rear', 'surround', 'iec958',
                                             'hdmi', 'modem', 'phoneline', 'dmix',
                                             'dsnoop', 'default']):
                    continue

                # Quick test
                try:
                    test_mic = sr.Microphone(device_index=i)
                    with test_mic as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print(f"[VOICE_NODE] ✓ Working mic [{i}]: {name}")
                    return i
                except Exception as e:
                    print(f"[VOICE_NODE] ✗ Mic [{i}] failed: {e}")
                    continue

            print("[VOICE_NODE] No working mic found")
            return None

        except Exception as e:
            print(f"[VOICE_NODE] Mic scan error: {e}")
            return None

    def _init_mic(self):
        """Initialize microphone"""
        if not self.enabled:
            return False

        mic_idx = self._find_working_mic()
        if mic_idx is None:
            self.enabled = False
            return False

        try:
            self.microphone = sr.Microphone(device_index=mic_idx)
            print("[VOICE_NODE] Calibrating (2s)...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"[VOICE_NODE] ✓ Ready — threshold: {self.recognizer.energy_threshold:.0f}")
            return True
        except Exception as e:
            print(f"[VOICE_NODE] Mic init error: {e}")
            self.enabled = False
            return False

    def activate(self):
        """Activate voice recognition (call this when face detected)"""
        self.is_active = True
        self.active_until = time.time() + self.active_duration
        print(f"[VOICE_NODE] 🎤 Activated for {self.active_duration}s")

    def _play_chime(self):
        """Play activation sound"""
        try:
            import pygame
            import numpy as np
            
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            sample_rate = 22050
            duration = 0.2
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            wave = 0.3 * np.sin(2 * np.pi * 523.25 * t)  # C5
            wave += 0.2 * np.sin(2 * np.pi * 659.25 * t)  # E5
            wave *= np.linspace(1, 0, len(wave))
            
            sound = pygame.sndarray.make_sound((wave * 32767).astype(np.int16))
            sound.play()
            time.sleep(0.3)  # Let sound finish
        except:
            pass

    def _check_wake_word(self, text):
        """Check if text contains wake word"""
        if not self.use_wake_word:
            return True
        t = text.lower()
        return (self.wake_word in t or 
                "aria" in t or 
                "hey" in t and "area" in t)

    def _listen_once(self, timeout=1.0, phrase_limit=5):
        """Listen for one utterance"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_limit
                )
            
            text = self.recognizer.recognize_google(audio, language='en-US')
            return text.strip()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[VOICE_NODE] API error: {e}")
            return None
        except Exception as e:
            print(f"[VOICE_NODE] Listen error: {e}")
            return None

    def _process_command(self, text):
        """Process recognized speech"""
        if not text:
            return

        now = time.time()
        if now - self.last_spoken < self.cooldown:
            return
        self.last_spoken = now

        low = text.lower().strip()
        print(f"[VOICE_NODE] Processing: '{text}'")

        # Check for wake word if not active
        if not self.is_active:
            if self._check_wake_word(low):
                print(f"[VOICE_NODE] ✅ Wake word detected!")
                self.activate()
                self._play_chime()
                # Don't process the wake word itself as a command
                return
            else:
                # Not active and no wake word — ignore
                return

        # We're active — extend timeout
        self.active_until = time.time() + self.active_duration

        # Ignore wake word echoes
        if low in ('hey aria', 'aria', 'hey', 'hey area', 'area'):
            return

        # Detect name
        for trigger in ['my name is', "i'm ", 'i am ', 'call me ']:
            if trigger in low:
                parts = low.split(trigger, 1)
                if len(parts) > 1:
                    name = parts[1].strip().split()[0].capitalize()
                    if len(name) > 1:
                        print(f"[VOICE_NODE] Name detected: {name}")
                        self.command_queue.put({'type': 'SET_NAME', 'name': name})
                        return

        # Regular message
        print(f"[VOICE_NODE] → Command: '{text}'")
        self.command_queue.put({'type': 'USER_MESSAGE', 'text': text})

    def _check_timeout(self):
        """Check if active period expired"""
        if self.use_wake_word and self.is_active:
            if time.time() > self.active_until:
                self.is_active = False
                print("[VOICE_NODE] ⏹ Deactivated (timeout)")

    def run(self):
        """Main loop"""
        self.running = True
        print("[VOICE_NODE] Starting...")

        if not self.enabled:
            print("[VOICE_NODE] Disabled — no speech recognition")
            while self.running:
                time.sleep(1)
            return

        if not self._init_mic():
            print("[VOICE_NODE] No mic available")
            while self.running:
                time.sleep(1)
            return

        if self.use_wake_word:
            print(f"\n[VOICE_NODE] 🎤 Say '{self.wake_word}' to activate")
        else:
            print("[VOICE_NODE] 🎤 Listening continuously")

        # Main loop
        while self.running:
            try:
                self._check_timeout()

                if self.use_wake_word and not self.is_active:
                    # Short listen for wake word only
                    text = self._listen_once(timeout=1.0, phrase_limit=3)
                    self._process_command(text)
                else:
                    # Active mode — longer listen for commands
                    text = self._listen_once(timeout=2.0, phrase_limit=8)
                    self._process_command(text)

            except Exception as e:
                print(f"[VOICE_NODE] Error: {e}")
                time.sleep(0.5)

    def shutdown(self):
        self.running = False
        print("[VOICE_NODE] Stopped")

