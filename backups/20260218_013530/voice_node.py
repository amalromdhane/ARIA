"""
Voice Node — Speech-to-text that fills GUI message input
Uses default microphone with VAD, Whisper transcription, and phonetic matching
"""

import threading
import time
import queue
import numpy as np
import webrtcvad
import wave
import io
import json
from datetime import datetime

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

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[VOICE_NODE] Run: pip3 install openai-whisper")

try:
    from rapidfuzz import fuzz, utils
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("[VOICE_NODE] Run: pip3 install rapidfuzz")


class VoiceNode:
    def __init__(self, command_queue, face_recognition_node=None, wake_word_mode=False):
        self.command_queue         = command_queue
        self.face_recognition_node = face_recognition_node
        self.wake_word_mode        = wake_word_mode
        self.running               = False

        self.recognizer  = sr.Recognizer() if SR_AVAILABLE else None
        self.microphone  = None
        self.enabled     = SR_AVAILABLE and PYAUDIO_AVAILABLE and WHISPER_AVAILABLE

        # Initialize Whisper model
        self.whisper_model = None
        if WHISPER_AVAILABLE and self.enabled:
            print("[VOICE_NODE] Loading Whisper small model...")
            try:
                self.whisper_model = whisper.load_model("small")
                print("[VOICE_NODE] Whisper model loaded")
            except Exception as e:
                print(f"[VOICE_NODE] Whisper load error: {e}")
                self.enabled = False

        # Initialize VAD
        self.vad = webrtcvad.Vad(2) if webrtcvad else None  # Aggressiveness level 2

        # Known names database (could be loaded from file)
        self.known_names = self._load_known_names()

        self._active       = not wake_word_mode
        self._active_until = 0
        self._active_secs  = 10
        self._active_lock  = threading.Lock()

        self.last_spoken = 0
        self.cooldown    = 1.2
        self.gui_node = None
        
        # Audio buffer for VAD
        self.audio_buffer = []
        self.recording = False
        self.silence_frames = 0
        self.MAX_SILENCE_FRAMES = 30  # ~0.6 seconds at 50ms frames
        
        mode = "wake-word" if wake_word_mode else "continuous"
        print(f"[VOICE_NODE] Initialized — {mode} mode | enabled: {self.enabled}")
        print(f"[VOICE_NODE] Known names: {len(self.known_names)} entries")

    def _load_known_names(self):
        """Load known names from database or JSON file"""
        # Example database - would load from actual DB in production
        return {
            "chahed": {"name": "Chahed", "variations": ["chahed", "chahid", "shahed", "shahid"]},
            "mohamed": {"name": "Mohamed", "variations": ["mohamed", "mohammed", "muhammed"]},
            "sarah": {"name": "Sarah", "variations": ["sarah", "sara"]},
            "ahmed": {"name": "Ahmed", "variations": ["ahmed", "ahmad", "achmed"]},
        }

    def _phonetic_match(self, text, threshold=0.85):
        """
        Match transcribed text against known names using phonetic similarity
        Returns matched name and confidence score
        """
        if not RAPIDFUZZ_AVAILABLE:
            return None, 0.0
            
        text_lower = text.lower().strip()
        
        best_match = None
        best_score = 0.0
        
        for key, data in self.known_names.items():
            # Check exact name
            score = fuzz.ratio(text_lower, key) / 100.0
            if score > best_score:
                best_score = score
                best_match = data["name"]
            
            # Check variations
            for variation in data["variations"]:
                score = fuzz.ratio(text_lower, variation) / 100.0
                if score > best_score:
                    best_score = score
                    best_match = data["name"]
        
        if best_score >= threshold:
            return best_match, best_score
        return None, best_score

    def _confirm_name(self, suggested_name, original_text):
        """Ask for confirmation when name confidence is low"""
        print(f"[VOICE_NODE] Low confidence match: '{original_text}' -> '{suggested_name}' (score: {suggested_name[1]:.2f})")
        
        # Could implement confirmation through GUI
        # For now, return the matched name if confidence > 0.7, otherwise None
        if suggested_name[1] >= 0.7:
            return suggested_name[0]
        
        # In production, would push to GUI for confirmation
        self.command_queue.put({
            'type': 'NAME_CONFIRMATION',
            'suggested': suggested_name[0],
            'original': original_text,
            'score': suggested_name[1]
        })
        
        # Return None to indicate needs confirmation
        return None

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

    def _vad_listen(self):
        """Listen with VAD to detect speech activity"""
        try:
            # Use pyaudio directly for VAD
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                           channels=1,
                           rate=16000,
                           input=True,
                           frames_per_buffer=960)  # 30ms frames at 16kHz
            
            frames = []
            speech_detected = False
            silence_frames = 0
            
            print("[VOICE_NODE] Listening for speech...")
            
            while self.running:
                if not self.is_active():
                    time.sleep(0.1)
                    continue
                    
                frame = stream.read(960, exception_on_overflow=False)
                is_speech = self.vad.is_speech(frame, 16000)
                
                if is_speech:
                    frames.append(frame)
                    speech_detected = True
                    silence_frames = 0
                elif speech_detected:
                    silence_frames += 1
                    if silence_frames > self.MAX_SILENCE_FRAMES:
                        # End of speech
                        break
                    frames.append(frame)
                else:
                    # Still waiting for speech
                    time.sleep(0.01)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if frames:
                # Convert frames to audio data
                audio_data = b''.join(frames)
                return audio_data
            
        except Exception as e:
            print(f"[VOICE_NODE] VAD listen error: {e}")
            
        return None

    def _transcribe_with_whisper(self, audio_data):
        """Transcribe audio using Whisper"""
        if not self.whisper_model:
            return None
            
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio_array,
                language="fr",  # French + Arabic support
                task="transcribe",
                fp16=False  # Use FP32 for compatibility
            )
            
            text = result["text"].strip()
            if text:
                print(f"[VOICE_NODE] Whisper: '{text}'")
                return text
            
        except Exception as e:
            print(f"[VOICE_NODE] Whisper error: {e}")
            
        return None

    def _process_text(self, text):
        """Process transcribed text with name matching"""
        if not text:
            return None
            
        # Check if text contains a name
        words = text.split()
        
        for word in words:
            matched_name, score = self._phonetic_match(word)
            
            if matched_name:
                print(f"[VOICE_NODE] Name matched: '{word}' -> '{matched_name}' (score: {score:.2f})")
                
                if score >= 0.85:
                    # High confidence - accept directly
                    print(f"[VOICE_NODE] ✓ High confidence, accepting '{matched_name}'")
                    return matched_name
                elif score >= 0.7:
                    # Medium confidence - could ask for confirmation
                    print(f"[VOICE_NODE] ⚠ Medium confidence: '{matched_name}'")
                    # In production, would push to GUI
                    self.command_queue.put({
                        'type': 'NAME_CONFIRMATION_NEEDED',
                        'name': matched_name,
                        'original': word,
                        'score': score
                    })
                    return matched_name  # Return for now, would wait for confirmation in real system
                else:
                    print(f"[VOICE_NODE] ✗ Low confidence: '{word}' (score: {score:.2f})")
                    return None
        
        return None

    def _listen_once(self):
        """Complete listening pipeline: VAD -> Whisper -> Name matching"""
        try:
            # Step 1: VAD detection
            audio_data = self._vad_listen()
            
            if not audio_data:
                return None
            
            # Step 2: Whisper transcription
            text = self._transcribe_with_whisper(audio_data)
            
            if not text:
                return None
            
            # Step 3: Name matching
            matched_name = self._process_text(text)
            
            return {
                'original_text': text,
                'matched_name': matched_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"[VOICE_NODE] Listen error: {e}")
            return None

    def _process(self, result):
        """Process the final result"""
        if not result:
            return
        
        now = time.time()
        if now - self.last_spoken < self.cooldown:
            return
        self.last_spoken = now
        
        # Skip wake words
        low = result['original_text'].lower().strip()
        if low in ('hey aria', 'aria', 'hey', 'area'):
            return
        
        # Format output
        if result['matched_name']:
            output_text = f"J'ai entendu {result['matched_name']} — C'est correct ? Sinon tapez"
            print(f"[VOICE_NODE] → Name detection: '{result['matched_name']}'")
            
            # Save to database (in production)
            self._save_to_db(result['matched_name'], result['original_text'], result['timestamp'])
        else:
            output_text = result['original_text']
            print(f"[VOICE_NODE] → Text: '{output_text}'")
        
        # Send to GUI
        if self.gui_node:
            print(f"[VOICE_NODE] → GUI: '{output_text}'")
            self.gui_node.set_message_input(output_text)
        else:
            print(f"[VOICE_NODE] → Direct: '{output_text}'")
            self.command_queue.put({'type': 'USER_MESSAGE', 'text': output_text})

    def _save_to_db(self, name, original_text, timestamp):
        """Save recognized name to database"""
        # In production, this would save to actual database
        print(f"[VOICE_NODE] 💾 Saving to DB: {name} (from: '{original_text}') at {timestamp}")
        
        # Example database save
        # self.db.save_name_recognition(name, original_text, timestamp)
        
        # Update known names with new variations
        if original_text not in self.known_names.get(name.lower(), {}).get('variations', []):
            print(f"[VOICE_NODE] 📝 Adding new variation: '{original_text}' for {name}")
            # In production, would save to DB and reload known names

    def _loop(self):
        errors = 0
        
        while self.running:
            if not self.is_active():
                time.sleep(0.2)
                continue
            
            try:
                result = self._listen_once()
                if result:
                    errors = 0
                    self._process(result)
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
            print("[VOICE_NODE] Disabled (missing dependencies)")
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
            print("[VOICE_NODE] 🎤 Listening with VAD and Whisper — speak clearly!")

        self._loop()

    def shutdown(self):
        self.running = False
        self._active = False
        print("[VOICE_NODE] Stopped")