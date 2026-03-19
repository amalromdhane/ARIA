"""
Emotion Detection Node — Smart Priority System

Rules:
1. NEVER comment on emotion during active conversation
2. On first detection → note emotion once at greeting (e.g. "you seem happy today")
3. If robot FAILS to help → show SAD face, acknowledge it
4. If response went well (user smiles after) → acknowledge positively
5. After a joke → react to laughter
6. Long silence + sad face → check in
7. Cooldown between any emotion comments: 60 seconds minimum
"""

import time
import threading
import queue


try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("[EMOTION_NODE] FER not available")


class EmotionDetectionNode:
    def __init__(self, state_manager, frame_queue, sound_emotion_queue, main_system=None):
        self.state_manager       = state_manager
        self.frame_queue         = frame_queue
        self.sound_emotion_queue = sound_emotion_queue
        self.main_system         = main_system
        self.running             = False

        self.detector      = None
        self.frame_counter = 0

        # Current detected emotion
        self.current_emotion  = 'neutral'
        self.emotion_score    = 0.0
        self.last_emotion     = None

        # Priority / conversation state
        self.conversation_active  = False
        self.last_conversation    = 0
        self.conversation_silence = 30   # seconds after last msg before emotion can speak

        # Emotion comment throttle
        self.last_emotion_comment = 0
        self.comment_cooldown     = 60   # minimum seconds between emotion comments

        # Session state flags — set by state_manager
        self.visitor_present      = False
        self.first_greeting_done  = False  # first emotion noted at greeting
        self.last_robot_action    = None   # 'helped', 'failed', 'joke'
        self.last_action_time     = 0

        # Sustained emotion tracking (for "still sad after 30s" check)
        self.sad_sustained_since  = None
        self.sad_sustained_threshold = 30  # seconds of continuous sad

        if FER_AVAILABLE:
            print("[EMOTION_NODE] Loading FER...")
            try:
                self.detector = FER(mtcnn=False)
                print("[EMOTION_NODE] ✓ FER ready")
            except Exception as e:
                print(f"[EMOTION_NODE] FER error: {e}")

        print("[EMOTION_NODE] Ready")

    # ── Called by state_manager ────────────────────────────────────────

    def set_conversation_active(self, active):
        self.conversation_active = active
        if active:
            self.last_conversation = time.time()

    def notify_visitor_arrived(self):
        """Called when face_recognition first greets someone"""
        self.visitor_present     = True
        self.first_greeting_done = False
        self.sad_sustained_since = None
        self.last_robot_action   = None

    def notify_visitor_left(self):
        self.visitor_present     = False
        self.first_greeting_done = False
        self.current_emotion     = 'neutral'

    def notify_robot_action(self, action):
        """
        action: 'helped' | 'failed' | 'joke'
        Called by ai_agent_node after delivering a response
        """
        self.last_robot_action = action
        self.last_action_time  = time.time()
        print(f"[EMOTION_NODE] Robot action noted: {action}")

    # ── Core processing ────────────────────────────────────────────────

    def _should_comment(self):
        """Return True only if emotion comment is appropriate now"""
        now = time.time()

        # Never during active conversation
        if self.conversation_active:
            return False

        # Wait for conversation silence period
        if now - self.last_conversation < self.conversation_silence:
            return False

        # Global cooldown
        if now - self.last_emotion_comment < self.comment_cooldown:
            return False

        # Need a visitor
        if not self.visitor_present:
            return False

        return True

    def process_frame(self, frame):
        if not self.detector:
            return

        self.frame_counter += 1
        # Analyse every 20 frames
        if self.frame_counter % 20 != 0:
            return

        try:
            result = self.detector.detect_emotions(frame)
            if not result:
                return

            emotions    = result[0]['emotions']
            top_emotion = max(emotions, key=emotions.get)
            score       = emotions[top_emotion]

            if score < 0.40:
                return

            self.current_emotion = top_emotion
            self.emotion_score   = score

            mapped = self._map(top_emotion)

            # Always update face display
            if mapped != self.last_emotion:
                self.last_emotion = mapped
                try:
                    self.state_manager.emotion_queue.put_nowait(mapped)
                except Exception:
                    pass

                # Update GUI mood label
                gui = getattr(self.state_manager, 'gui_node', None)
                if gui:
                    gui.update_visitor_info(mood=top_emotion)

            # Track sustained sadness
            if top_emotion == 'sad':
                if self.sad_sustained_since is None:
                    self.sad_sustained_since = time.time()
            else:
                self.sad_sustained_since = None

            # Smart comment logic
            self._consider_comment(top_emotion, score, mapped)

        except Exception as e:
            pass

    def _consider_comment(self, emotion, score, mapped):
        now = time.time()

        # ── Case 1: First detection at greeting ───────────────────────
        if (self.visitor_present
                and not self.first_greeting_done
                and not self.conversation_active
                and now - self.last_conversation > 5):

            self.first_greeting_done  = True
            self.last_emotion_comment = now

            msg = None
            if emotion == 'happy' and score > 0.55:
                msg = "You look happy today — that's great to see!"
            elif emotion == 'sad' and score > 0.50:
                msg = "You seem a little down. I hope I can help brighten your day."
            elif emotion == 'angry' and score > 0.50:
                msg = "I can see you might be frustrated. I'll do my best to help."
            # neutral / surprised get no comment — don't be annoying

            if msg:
                self._speak_and_show(msg, mapped)
            return

        # ── Case 2: After robot action — react to emotion response ────
        if self.last_robot_action and now - self.last_action_time < 20:
            action = self.last_robot_action
            self.last_robot_action = None  # consume it

            if action == 'helped' and emotion in ('happy', 'surprise'):
                if self._should_comment():
                    msg = "I'm glad that helped! Anything else I can do for you?"
                    self.last_emotion_comment = now
                    self._speak_and_show(msg, 'HAPPY')
                    return

            elif action == 'failed' and emotion in ('sad', 'angry', 'disgust'):
                if self._should_comment():
                    msg = "I'm sorry I couldn't help better. Let me try a different approach."
                    self.last_emotion_comment = now
                    self._speak_and_show(msg, 'SAD')
                    return

            elif action == 'joke' and emotion in ('happy', 'surprise'):
                if self._should_comment():
                    msg = "Ha, glad that landed! I've been working on my delivery."
                    self.last_emotion_comment = now
                    self._speak_and_show(msg, 'HAPPY')
                    return

        # ── Case 3: Sustained sadness (>30s, no conversation) ─────────
        if (self.sad_sustained_since
                and now - self.sad_sustained_since > self.sad_sustained_threshold
                and self._should_comment()):

            self.sad_sustained_since  = None
            self.last_emotion_comment = now
            msg = "You've seemed a bit down for a while. Is there anything I can help with?"
            self._speak_and_show(msg, 'SAD')
            return

    def _map(self, emotion):
        return {
            'happy':    'HAPPY',
            'sad':      'SAD',
            'angry':    'ANGRY',
            'fear':     'SURPRISED',
            'surprise': 'SURPRISED',
            'neutral':  'NEUTRAL',
            'disgust':  'SAD',
        }.get(emotion, 'NEUTRAL')

    def _speak_and_show(self, text, emotion):
        try:
            self.sound_emotion_queue.put({'type': 'SPEAK', 'text': text})
        except Exception:
            pass
        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.add_chat_message('ROBOT', text)
        print(f"[EMOTION_NODE] Comment: '{text}'")

    def run(self):
        self.running = True
        print("[EMOTION_NODE] Running...")
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    self.process_frame(frame)
            except Exception:
                pass
            time.sleep(0.04)

    def stop(self):
        self.running = False
        print("[EMOTION_NODE] Stopped")
