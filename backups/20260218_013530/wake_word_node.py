"""
Wake Word Node - Simple offline wake word detection
Uses speech recognition + keyword matching (no API keys needed)
"""

import threading
import time
import queue

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[WAKE_WORD] speech_recognition not installed")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class WakeWordNode:
    """
    Simple wake word detection using speech recognition
    Listens for keyword, then activates voice node
    """
    
    def __init__(self, command_queue, wake_word="hey aria"):
        self.command_queue = command_queue
        self.wake_word = wake_word.lower()
        self.running = False
        self.voice_node = None  # Will be wired by main.py
        
        # Recognition settings
        self.energy_threshold = 400  # Higher = less sensitive to background
        self.pause_threshold = 0.5
        self.phrase_threshold = 0.3
        
        # Timing
        self.listen_timeout = 1.0  # Seconds to listen for wake word
        self.phrase_time_limit = 3.0  # Max phrase length
        
        self.enabled = SR_AVAILABLE and PYAUDIO_AVAILABLE
        
        if self.enabled:
            self.recognizer = sr.Recognizer()
            self.microphone = None
        else:
            self.recognizer = None
            
        print(f"[WAKE_WORD] Initialized (simple mode) - wake word: '{self.wake_word}'")
        print(f"[WAKE_WORD] Enabled: {self.enabled}")

    def _init_mic(self):
        """Initialize microphone"""
        if not self.enabled:
            return False
            
        try:
            # Find best microphone
            mic_list = sr.Microphone.list_microphone_names()
            preferred_idx = None
            
            for i, name in enumerate(mic_list):
                n = name.lower()
                if any(x in n for x in ['usb', 'headset', 'default', 'built-in']):
                    preferred_idx = i
                    print(f"[WAKE_WORD] Using mic: {name}")
                    break
            
            if preferred_idx is not None:
                self.microphone = sr.Microphone(device_index=preferred_idx)
            else:
                self.microphone = sr.Microphone()
            
            # Calibrate for ambient noise
            print("[WAKE_WORD] Calibrating... (stay quiet)")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, self.energy_threshold)
                self.recognizer.pause_threshold = self.pause_threshold
                self.recognizer.phrase_threshold = self.phrase_threshold
            
            print(f"[WAKE_WORD] Calibrated. Energy threshold: {self.recognizer.energy_threshold}")
            return True
            
        except Exception as e:
            print(f"[WAKE_WORD] Mic init error: {e}")
            return False

    def _listen_for_wake_word(self):
        """Listen for wake word phrase"""
        if not self.microphone:
            return False
            
        try:
            with self.microphone as source:
                # Short listen for wake word
                audio = self.recognizer.listen(
                    source, 
                    timeout=self.listen_timeout,
                    phrase_time_limit=self.phrase_time_limit
                )
            
            # Try to recognize
            try:
                text = self.recognizer.recognize_google(audio, language='en-US').lower()
                print(f"[WAKE_WORD] Heard: '{text}'")
                
                # Check if wake word is in phrase
                if self.wake_word in text or "aria" in text or "hey" in text:
                    print(f"\n[WAKE_WORD] ✅ '{self.wake_word.upper()}' DETECTED!")
                    return True
                    
                # Also check for similar words
                wake_parts = self.wake_word.split()
                if any(part in text for part in wake_parts):
                    print(f"\n[WAKE_WORD] ✅ (partial match) DETECTED!")
                    return True
                    
            except sr.UnknownValueError:
                pass  # Didn't understand audio
            except sr.RequestError as e:
                print(f"[WAKE_WORD] API error: {e}")
                
        except sr.WaitTimeoutError:
            pass  # No speech detected
        except Exception as e:
            print(f"[WAKE_WORD] Listen error: {e}")
            
        return False

    def _activate_voice(self):
        """Activate voice node and play sound"""
        # Notify voice node
        if self.voice_node:
            self.voice_node.activate_wake_word()
        
        # Notify command queue (for state manager)
        self.command_queue.put({
            'type': 'WAKE_WORD_DETECTED',
            'wake_word': self.wake_word
        })
        
        # Play activation sound
        self._play_sound()

    def _play_sound(self):
        """Play confirmation sound"""
        try:
            import pygame
            import numpy as np
            
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            
            # Pleasant chime (C5 + E5)
            sample_rate = 22050
            duration = 0.2
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            wave = 0.3 * np.sin(2 * np.pi * 523.25 * t)  # C5
            wave += 0.2 * np.sin(2 * np.pi * 659.25 * t)  # E5
            
            # Fade out
            wave *= np.linspace(1, 0, len(wave))
            
            sound = pygame.sndarray.make_sound((wave * 32767).astype(np.int16))
            sound.play()
            
        except Exception as e:
            # Silent fallback
            pass

    def run(self):
        """Main loop"""
        self.running = True
        
        if not self.enabled:
            print("[WAKE_WORD] Disabled - running in passive mode")
            while self.running:
                time.sleep(1)
            return
            
        if not self._init_mic():
            print("[WAKE_WORD] Could not initialize microphone")
            while self.running:
                time.sleep(1)
            return
        
        print(f"\n[WAKE_WORD] 🎤 Listening for '{self.wake_word}'...")
        print("[WAKE_WORD] (Say 'Hey Aria' to activate voice commands)\n")
        
        while self.running:
            try:
                if self._listen_for_wake_word():
                    self._activate_voice()
                    
                    # Wait for voice node to finish (10 sec timeout)
                    time.sleep(10)
                    
                    print(f"\n[WAKE_WORD] 🎤 Listening for '{self.wake_word}'...")
                    
            except Exception as e:
                print(f"[WAKE_WORD] Error: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False
        print("[WAKE_WORD] Stopped")


# Alias for compatibility
SimpleWakeWordNode = WakeWordNode
