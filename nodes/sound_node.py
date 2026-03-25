"""
Sound Node — Fixed version
=================================
FIXES:
1. ENGLISH ONLY for robot-generated sentences. The _detect_lang() and
   _split_by_script() functions are kept for correct pronunciation of
   names that may contain Arabic characters, but all robot sentences
   are now English so no French detection is needed in practice.
2. French keyword list stripped down — French words in robot output
   should no longer appear since all messages are English now.
3. No logic changes to TTS engine itself — gTTS + pygame path unchanged.
"""

import threading
import queue
import time
import os
import tempfile

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[SOUND_NODE] pip3 install pygame")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("[SOUND_NODE] pip3 install gtts  (falling back to espeak)")


# ── Language detection ────────────────────────────────────────────────────────

def _detect_lang(text):
    """
    Detect TTS language for a segment of text.
    Returns 'ar' only when text is MOSTLY Arabic (>70% of alpha chars).
    Everything else is English ('en').
    French detection removed — all robot output is now English.
    """
    if not text:
        return 'en'

    arabic_chars = sum(1 for c in text
                       if '\u0600' <= c <= '\u06FF'
                       or '\u0750' <= c <= '\u077F'
                       or '\u08A0' <= c <= '\u08FF')
    latin_chars  = sum(1 for c in text if c.isascii() and c.isalpha())

    total = arabic_chars + latin_chars
    if total > 0 and arabic_chars / total > 0.7:
        return 'ar'

    return 'en'


# ── Speech functions ──────────────────────────────────────────────────────────

def _split_by_script(text):
    """
    Split text into (lang, segment) pairs by dominant script.
    'Welcome back أمال!' → [('en','Welcome back '), ('ar','أمال!')]
    Each segment is spoken with the correct gTTS voice.
    Base language for Latin segments is always 'en' (English only).
    """
    segments = []
    current_script = None
    current = []

    for char in text:
        if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
            script = 'ar'
        elif char.isalpha():
            script = 'latin'
        else:
            script = current_script or 'latin'

        if char.isalpha() and script != current_script:
            if current:
                segments.append((current_script, ''.join(current)))
            current = [char]
            current_script = script
        else:
            current.append(char)
            if current_script is None:
                current_script = script

    if current:
        segments.append((current_script, ''.join(current)))

    result = []
    for script, seg in segments:
        if not seg.strip():
            continue
        # FIX — base language always English; Arabic names still get 'ar'
        lang = 'ar' if script == 'ar' else 'en'
        result.append((lang, seg))
    return result


def _speak_gtts(text):
    """
    Speak text using gTTS + pygame.
    Splits mixed-script text so Arabic names are pronounced correctly
    within otherwise English sentences.
    Returns True on success, False on failure.
    """
    if not GTTS_AVAILABLE or not PYGAME_AVAILABLE:
        return False
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        segments = _split_by_script(text)

        for lang, segment in segments:
            if not segment.strip():
                continue
            tts = gTTS(text=segment, lang=lang, slow=False)

            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                tmp_path = f.name

            tts.save(tmp_path)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

            pygame.mixer.music.unload()
            os.unlink(tmp_path)

        print(f"[SOUND_NODE] gTTS: {text}")
        return True

    except Exception as e:
        print(f"[SOUND_NODE] gTTS error: {e} — falling back to espeak")
        return False


def _speak_espeak(text):
    """
    Fallback: espeak with majority-script detection.
    Arabic names in English sentences stay English (majority wins).
    """
    try:
        import subprocess
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        latin_chars  = sum(1 for c in text if c.isascii() and c.isalpha())
        total        = arabic_chars + latin_chars
        is_arabic    = total > 0 and arabic_chars / total > 0.7
        voice        = 'ar' if is_arabic else 'en-us'
        subprocess.run(['espeak', '-v', voice, text],
                       stderr=subprocess.DEVNULL, check=False)
        print(f"[SOUND_NODE] espeak [{voice}]: {text}")
    except FileNotFoundError:
        print("[SOUND_NODE] espeak not found")
    except Exception as e:
        print(f"[SOUND_NODE] espeak error: {e}")


def _speak(text):
    """Speak text: try gTTS first, fall back to espeak."""
    if not text or not text.strip():
        return
    success = _speak_gtts(text)
    if not success:
        _speak_espeak(text)


# ── SoundNode class ───────────────────────────────────────────────────────────

class SoundNode:
    def __init__(self, emotion_queue):
        self.sound_emotion_queue = emotion_queue
        self.running             = True
        self.voice_node          = None
        self.is_speaking         = False
        self.speaking_lock       = threading.Lock()

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                print("[SOUND_NODE] pygame mixer initialized")
            except Exception as e:
                print(f"[SOUND_NODE] pygame mixer init failed: {e}")

        print(f"[SOUND_NODE] TTS engine: {'gTTS' if GTTS_AVAILABLE else 'espeak (fallback)'}")
        print("[SOUND_NODE] Node initialized")

    def set_voice_node(self, voice_node):
        self.voice_node = voice_node
        print("[SOUND_NODE] Voice node connected")

    def is_currently_speaking(self):
        with self.speaking_lock:
            return self.is_speaking

    def cancel_pending_tts(self):
        """
        Drain all pending SPEAK commands from the queue.
        Called when a GUI name correction arrives — prevents the stale
        "I heard Sarah…" TTS from playing after "Farah" was accepted.
        Does NOT interrupt audio already playing (that's handled by
        the caller stopping the confirmation window).
        """
        cancelled = 0
        while not self.sound_emotion_queue.empty():
            try:
                item = self.sound_emotion_queue.get_nowait()
                if isinstance(item, dict) and item.get('type') == 'SPEAK':
                    cancelled += 1
                    print(f"[SOUND_NODE] Cancelled queued TTS: {item.get('text','')[:50]!r}")
                else:
                    # Put non-SPEAK items back (emotion beeps etc.)
                    try:
                        self.sound_emotion_queue.put_nowait(item)
                    except Exception:
                        pass
            except Exception:
                break
        if cancelled:
            print(f"[SOUND_NODE] Cancelled {cancelled} pending TTS command(s)")

    def _notify_voice_node_speaking(self, text):
        """
        Mute microphone for estimated speech duration.
        gTTS fetch (~0.5s) + playback + mic ringdown = 0.5s/word + 2.5s buffer.
        """
        if self.voice_node:
            word_count = len(text.split())
            duration   = max(3.5, word_count * 0.5 + 2.5)
            self.voice_node._robot_speaking           = True
            self.voice_node._robot_speaking_until     = time.time() + duration + 0.5
            self.voice_node._last_robot_response      = text.lower().strip()
            self.voice_node._last_robot_response_time = time.time()
            print(f"[SOUND_NODE] Muting mic for {duration:.1f}s")

    def speak(self, text):
        """Public speak method — notifies voice node then speaks."""
        if not text:
            return

        self._notify_voice_node_speaking(text)

        with self.speaking_lock:
            self.is_speaking = True

        try:
            _speak(text)
            time.sleep(0.1)
        except Exception as e:
            print(f"[SOUND_NODE] speak() error: {e}")
        finally:
            with self.speaking_lock:
                self.is_speaking = False
            if self.voice_node:
                self.voice_node._robot_speaking        = False
                self.voice_node._last_robot_speech_end = time.time()

    def play_beep(self, frequency=440, duration=0.1):
        if not PYGAME_AVAILABLE:
            return
        try:
            import numpy as np
            sample_rate = 22050
            frames      = int(duration * sample_rate)
            arr         = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
            sound       = np.array(arr * 32767, dtype=np.int16)
            sound       = np.repeat(sound.reshape(frames, 1), 2, axis=1)
            pygame.sndarray.make_sound(sound).play()
        except Exception as e:
            print(f"[SOUND_NODE] Beep error: {e}")

    def process_emotions(self):
        while self.running:
            try:
                if not self.sound_emotion_queue.empty():
                    emotion_data = self.sound_emotion_queue.get_nowait()
                    if isinstance(emotion_data, dict):
                        if emotion_data.get('type') == 'SPEAK':
                            text = emotion_data.get('text')
                            if text:
                                self.speak(text)
                                continue
                    else:
                        emotion = emotion_data
                        print(f"[SOUND_NODE] Emotion: {emotion}")
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
                time.sleep(0.1)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[SOUND_NODE] Error: {e}")

    def run(self):
        print("[SOUND_NODE] Node running...")
        self.process_emotions()

    def shutdown(self):
        self.running = False
        print("[SOUND_NODE] Shutting down...")