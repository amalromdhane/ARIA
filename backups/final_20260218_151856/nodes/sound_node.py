"""
Sound Node - Plays sounds for emotions
"""

import threading
import queue
import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[SOUND_NODE] pygame not installed - sounds disabled")


class SoundNode:
    def __init__(self, emotion_queue):
        """
        Initialize Sound Node
        
        Args:
            emotion_queue: Queue to receive emotion updates
        """
        self.sound_emotion_queue = emotion_queue
        self.running = True
        
        # Initialize pygame mixer for sound effects
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                print("[SOUND_NODE] pygame mixer initialized")
            except:
                print("[SOUND_NODE] Could not initialize pygame mixer")
        
        # Sound phrases for each emotion
        self.emotion_sounds = {
            'HAPPY': "Hello! How can I help you today?",
            'SAD': "Oh no, I'm sorry about that.",
            'ANGRY': "This is not acceptable!",
            'SURPRISED': "Wow! I didn't expect that!",
            'CONFUSED': "Hmm, let me think about this...",
            'TIRED': "I'm a bit tired right now...",
            'EXCITED': "This is amazing! I'm so excited!",
            'ATTENTIVE': "I'm listening carefully.",
            'NEUTRAL': "Standing by."
        }
        
        print("[SOUND_NODE] Node initialized")
    
    def speak(self, text):
        """Speak text using espeak"""
        try:
            import os
            # Simple blocking call
            os.system(f'espeak "{text}" 2>/dev/null &')
            print(f"[SOUND_NODE] Speaking: {text}")
        except Exception as e:
            print(f"[SOUND_NODE] Speech error: {e}")
    
    def play_beep(self, frequency=440, duration=0.1):
        """Play a simple beep sound"""
        if PYGAME_AVAILABLE:
            try:
                # Generate a simple beep
                sample_rate = 22050
                import numpy as np
                
                # Generate sine wave
                frames = int(duration * sample_rate)
                arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
                
                # Convert to 16-bit data
                sound = np.array(arr * 32767, dtype=np.int16)
                sound = np.repeat(sound.reshape(frames, 1), 2, axis=1)
                
                # Play sound
                sound_obj = pygame.sndarray.make_sound(sound)
                sound_obj.play()
            except Exception as e:
                print(f"[SOUND_NODE] Beep error: {e}")
    
    def process_emotions(self):
        """Process incoming emotion updates and play sounds"""
        while self.running:
            try:
                if not self.sound_emotion_queue.empty():
                    emotion_data = self.sound_emotion_queue.get_nowait()
                    
                    # Handle dict (custom message) or string (emotion)
                    if isinstance(emotion_data, dict):
                        # Custom speech command
                        if emotion_data.get('type') == 'SPEAK':
                            text = emotion_data.get('text')
                            if text:
                                print(f"[SOUND_NODE] Custom speech: {text}")
                                self.speak(text)
                                # Skip default emotion sound when custom message sent
                                continue
                    else:
                        # Regular emotion
                        emotion = emotion_data
                        print(f"[SOUND_NODE] Playing sound for: {emotion}")
                        
                        # Play beep
                        if emotion == 'HAPPY':
                            self.play_beep(523, 0.1)
                            time.sleep(0.05)
                            self.play_beep(659, 0.1)
                        elif emotion == 'SAD':
                            self.play_beep(392, 0.2)
                        elif emotion == 'SURPRISED':
                            self.play_beep(880, 0.05)
                            time.sleep(0.02)
                            self.play_beep(1046, 0.05)
                        else:
                            self.play_beep(440, 0.1)
                        
                        # DON'T speak default phrase automatically
                        # Only beep for state changes
                        # Custom messages (emotion/face rec) handle speech
                
                time.sleep(0.1)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[SOUND_NODE] Error: {e}")
    
    def run(self):
        """Start the sound node"""
        print("[SOUND_NODE] Node running...")
        self.process_emotions()
    
    def shutdown(self):
        """Shutdown the sound node"""
        self.running = False
        print("[SOUND_NODE] Shutting down...")
