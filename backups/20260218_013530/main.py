#!/usr/bin/env python3
"""
Main — ARIA Reception Robot (Fixed Version)
"""

import sys
import os
import queue
import threading
import time
import ctypes

# ── Suppress ALSA errors ─────────────────────────────────────────────────────
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                           ctypes.c_char_p, ctypes.c_int,
                                           ctypes.c_char_p)
    def _alsa_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(_alsa_error_handler)
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass

os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Import nodes ─────────────────────────────────────────────────────────────
print("Loading modules...")

from nodes.state_manager_node import StateManagerNode
from nodes.sensor_node import SensorNode
from nodes.sound_node import SoundNode
from nodes.emotion_detection_node import EmotionDetectionNode
from nodes.face_recognition_node import FaceRecognitionNode
from nodes.voice_node import VoiceNode
from nodes.ai_agent_node import AIAgentNode

# GUI last
from nodes.gui_node import GUINode

print("All modules loaded.\n")


class ReceptionRobotSystem:
    def __init__(self):
        print("=" * 60)
        print("  ARIA — Reception Assistant  |  Mistral AI")
        print("=" * 60)

        self.threads = []
        self.keyboard_running = True
        self.gui_node = None  # Will be created in run()

        # Shared queues
        self.state_queue = queue.Queue()
        self.emotion_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.sound_emotion_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=5)

        # Initialize all nodes EXCEPT GUI
        print("\n[SYSTEM] Initializing nodes...")

        self.state_manager = StateManagerNode(
            self.state_queue, self.emotion_queue,
            self.command_queue, self.sound_emotion_queue
        )

        self.sensor_node = SensorNode(
            self.command_queue, use_camera=True, frame_queue=self.frame_queue
        )

        self.sound_node = SoundNode(self.sound_emotion_queue)

        self.emotion_detection_node = EmotionDetectionNode(
            self.state_manager, self.frame_queue, self.sound_emotion_queue, self
        )

        self.face_recognition_node = FaceRecognitionNode(
            self.frame_queue, self.sound_emotion_queue, self.state_manager, self
        )

        self.ai_agent_node = AIAgentNode(
            self.sound_emotion_queue, self.face_recognition_node
        )

        # Voice (no wake word — simpler)
        try:
            self.voice_node = VoiceNode(
                self.command_queue,
                self.face_recognition_node,
                wake_word_mode=False  # Always active
            )
        except Exception as e:
            print(f"[SYSTEM] Voice init failed: {e}")
            self.voice_node = _DummyVoiceNode()

        print("[SYSTEM] Nodes initialized ✓")

    def run(self):
        """Start all threads, then GUI in main thread"""
        
        # Wire nodes together
        self.state_manager.face_recognition_node = self.face_recognition_node
        self.state_manager.ai_agent_node = self.ai_agent_node
        self.state_manager.emotion_detection_node = self.emotion_detection_node
        self.state_manager.voice_node = self.voice_node

        self.ai_agent_node.state_manager = self.state_manager

        # NOW create GUI (in main thread)
        print("[SYSTEM] Creating GUI...")
        self.gui_node = GUINode(
            self.state_queue, self.emotion_queue, self.command_queue
        )

        # Wire GUI reference back
        self.state_manager.gui_node = self.gui_node
        self.ai_agent_node.gui_node = self.gui_node

        # Start background threads (daemon = won't block exit)
        threads = [
            ("StateManager", self.state_manager.run),
            ("Sensor", self.sensor_node.run),
            ("Sound", self.sound_node.run),
            ("EmotionDetect", self.emotion_detection_node.run),
            ("FaceRecognition", self.face_recognition_node.run),
            ("AIAgent", self.ai_agent_node.run),
            ("Voice", self.voice_node.run),
        ]

        for name, target in threads:
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self.threads.append(t)
            print(f"[SYSTEM] ▶ {name}")

        # Keyboard thread
        kb = threading.Thread(target=self._keyboard, daemon=True, name="Keyboard")
        kb.start()

        print("\n" + "=" * 60)
        print("  ✅  ARIA IS READY")
        print("=" * 60)
        print("  🎤  Voice: listening")
        print("  📷  Camera: active")
        print("  💬  Type in GUI or speak")
        print("  ⌨️   Type 'quit' to exit")
        print("=" * 60 + "\n")

        # GUI runs in MAIN THREAD (blocks here)
        try:
            self.gui_node.run()
        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted")
        finally:
            self.shutdown()

    def _keyboard(self):
        while self.keyboard_running:
            try:
                raw = input().strip()
                if not raw:
                    continue
                low = raw.lower()

                if low == 'quit':
                    self.shutdown()
                    break
                elif low.startswith('ask '):
                    self.ai_agent_node.ask(raw[4:].strip())
                elif low.startswith('say '):
                    text = raw[4:].strip()
                    self.sound_emotion_queue.put({'type': 'SPEAK', 'text': text})
                    if self.gui_node:
                        self.gui_node.add_chat_message('ROBOT', text)
                elif low == 'reset':
                    self.ai_agent_node.history = []
                    self.ai_agent_node.current_name = None
                    if self.gui_node:
                        self.gui_node.update_visitor_info(name='—', visits='—', mood='—')
                    print("Context reset")
                else:
                    self.ai_agent_node.ask(raw)
            except EOFError:
                break
            except Exception as e:
                print(f"[KB] {e}")

    def shutdown(self):
        print("\n[SYSTEM] Shutting down...")
        self.keyboard_running = False

        try:
            self.command_queue.put_nowait({'type': 'SHUTDOWN'})
        except:
            pass

        for node, method in [
            (self.sensor_node, 'shutdown'),
            (self.sound_node, 'shutdown'),
            (self.emotion_detection_node, 'stop'),
            (self.face_recognition_node, 'stop'),
            (self.ai_agent_node, 'stop'),
            (self.voice_node, 'shutdown'),
            (self.state_manager, 'shutdown'),
            (self.gui_node, 'shutdown'),
        ]:
            try:
                if node:
                    getattr(node, method)()
            except Exception as e:
                print(f"[SYSTEM] {type(node).__name__}: {e}")

        print("[SYSTEM] Done.")


class _DummyVoiceNode:
    def activate_wake_word(self): 
        pass
    def run(self):
        while True:
            time.sleep(1)
    def shutdown(self): 
        pass


if __name__ == "__main__":
    try:
        system = ReceptionRobotSystem()
        system.run()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
