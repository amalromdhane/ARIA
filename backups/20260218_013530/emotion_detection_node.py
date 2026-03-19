"""
Emotion Detection Node - with conversation priority
Does NOT interrupt when user is in active conversation
"""

import time
import threading
import queue

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("[EMOTION_NODE] FER not installed")


class EmotionDetectionNode:
    def __init__(self, state_manager, frame_queue, sound_emotion_queue, main_system=None):
        self.state_manager       = state_manager
        self.frame_queue         = frame_queue
        self.sound_emotion_queue = sound_emotion_queue
        self.main_system         = main_system
        self.running             = False

        self.detector       = None
        self.frame_counter  = 0
        self.last_emotion   = None
        self.cooldown       = 10.0   # seconds between emotion announcements
        self.last_spoken    = 0

        # Priority lock — when True, emotion detection stays silent
        self.conversation_active = False
        self.last_conversation   = 0
        self.conversation_silence = 30  # seconds of silence after conversation

        if FER_AVAILABLE:
            print("[EMOTION_NODE] Initializing FER...")
            try:
                self.detector = FER(mtcnn=False)
                print("[EMOTION_NODE] ✓ FER ready")
            except Exception as e:
                print(f"[EMOTION_NODE] FER init error: {e}")
        else:
            print("[EMOTION_NODE] FER not available")

        print("[EMOTION_NODE] Ready")

    def set_conversation_active(self, active):
        """Called by state manager when user is talking"""
        self.conversation_active = active
        if active:
            self.last_conversation = time.time()

    def _should_speak(self):
        """Return True only if it's appropriate to comment on emotion"""
        now = time.time()

        # Silence if conversation happened recently
        if now - self.last_conversation < self.conversation_silence:
            return False

        # Silence if we spoke recently
        if now - self.last_spoken < self.cooldown:
            return False

        return True

    def process_frame(self, frame):
        """Analyze frame for emotions"""
        if not self.detector:
            return

        self.frame_counter += 1
        if self.frame_counter % 30 != 0:
            return

        try:
            result = self.detector.detect_emotions(frame)
            if not result:
                return

            emotions   = result[0]['emotions']
            top_emotion = max(emotions, key=emotions.get)
            score       = emotions[top_emotion]

            if score < 0.45:
                return

            emotion_map = {
                'happy':    'HAPPY',
                'sad':      'SAD',
                'angry':    'ANGRY',
                'fear':     'SURPRISED',
                'surprise': 'SURPRISED',
                'neutral':  'NEUTRAL',
                'disgust':  'SAD',
            }
            mapped = emotion_map.get(top_emotion, 'NEUTRAL')

            # Always update the face
            if mapped != self.last_emotion:
                self.last_emotion = mapped
                self.state_manager.change_state(
                    self._emotion_to_state(mapped))

                # Update GUI visitor info
                if hasattr(self.state_manager, 'gui_node') and self.state_manager.gui_node:
                    self.state_manager.gui_node.update_visitor_info(
                        mood=top_emotion)

            # Only speak if not in conversation
            if self._should_speak() and mapped not in ('NEUTRAL',):
                self.last_spoken = time.time()
                responses = {
                    'HAPPY':     "You look happy today! Welcome.",
                    'SAD':       "You seem a bit down. I'm here to help.",
                    'ANGRY':     "You seem frustrated. Let me assist you.",
                    'SURPRISED': "You look surprised! How can I help?",
                }
                msg = responses.get(mapped)
                if msg:
                    self.sound_emotion_queue.put({'type': 'SPEAK', 'text': msg})
                    if hasattr(self.state_manager, 'gui_node') and self.state_manager.gui_node:
                        self.state_manager.gui_node.add_chat_message('ROBOT', msg)

        except Exception as e:
            pass

    def _emotion_to_state(self, emotion):
        return {
            'HAPPY':     'GREETING',
            'SAD':       'HELPING',
            'ANGRY':     'HELPING',
            'SURPRISED': 'SURPRISED',
            'NEUTRAL':   'IDLE',
        }.get(emotion, 'IDLE')

    def run(self):
        self.running = True
        print("[EMOTION_NODE] Running...")
        import cv2
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    self.process_frame(frame)
            except Exception:
                pass
            time.sleep(0.05)

    def stop(self):
        self.running = False
        print("[EMOTION_NODE] Stopped")
