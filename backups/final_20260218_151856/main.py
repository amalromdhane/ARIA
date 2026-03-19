#!/usr/bin/env python3
"""
Main — ARIA Reception Robot with Mistral AI
"""

import sys
import os
import queue
import threading
import time

os.environ['PYTHONWARNINGS']                            = 'ignore'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']               = 'hide'
os.environ['TF_ENABLE_ONEDNN_OPTS']                    = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']                     = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']   = 'python'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodes.gui_node                 import GUINode
from nodes.state_manager_node       import StateManagerNode
from nodes.sensor_node              import SensorNode
from nodes.logger_node              import LoggerNode
from nodes.sound_node               import SoundNode
from nodes.emotion_detection_node   import EmotionDetectionNode
from nodes.face_recognition_node    import FaceRecognitionNode
from nodes.voice_node               import VoiceNode
from nodes.ai_agent_node            import AIAgentNode


class ReceptionRobotSystem:
    def __init__(self):
        print("=" * 60)
        print("  ARIA — Reception Assistant  |  Mistral AI")
        print("=" * 60)

        # Shared queues
        self.state_queue         = queue.Queue()
        self.emotion_queue       = queue.Queue()
        self.command_queue       = queue.Queue()
        self.log_queue           = queue.Queue()
        self.sound_emotion_queue = queue.Queue()
        self.frame_queue         = queue.Queue(maxsize=5)

        self.threads          = []
        self.keyboard_running = True

        print("\n[SYSTEM] Initializing nodes...")

        # GUI
        self.gui_node = GUINode(
            self.state_queue,
            self.emotion_queue,
            self.command_queue
        )

        # State Manager
        self.state_manager = StateManagerNode(
            self.state_queue,
            self.emotion_queue,
            self.command_queue,
            self.sound_emotion_queue
        )

        # Sensor
        self.sensor_node = SensorNode(
            self.command_queue,
            use_camera=True,
            frame_queue=self.frame_queue
        )

        # Logger
        self.logger_node = LoggerNode(self.state_queue, 'logs/robot.log')

        # Sound
        self.sound_node = SoundNode(self.sound_emotion_queue)

        # Emotion Detection
        self.emotion_detection_node = EmotionDetectionNode(
            self.state_manager,
            self.frame_queue,
            self.sound_emotion_queue,
            self
        )

        # Face Recognition
        self.face_recognition_node = FaceRecognitionNode(
            self.frame_queue,
            self.sound_emotion_queue,
            self.state_manager,
            self
        )

        # AI Agent (Mistral)
        self.ai_agent_node = AIAgentNode(
            self.sound_emotion_queue,
            self.face_recognition_node
        )

        # Voice
        self.voice_node = VoiceNode(
            self.command_queue,
            self.face_recognition_node,
        )

        # ── Wire everything together ──────────────────────────────────
        self.state_manager.face_recognition_node  = self.face_recognition_node
        self.state_manager.ai_agent_node          = self.ai_agent_node
        self.state_manager.gui_node               = self.gui_node
        self.state_manager.emotion_detection_node = self.emotion_detection_node

        self.ai_agent_node.gui_node               = self.gui_node
        self.ai_agent_node.state_manager          = self.state_manager

        print("[SYSTEM] ✓ All nodes initialized and wired")

    def start(self):
        print("\n[SYSTEM] Starting threads...\n")

        threads = [
            ("StateManager",    self.state_manager.run),
            ("Sensor",          self.sensor_node.run),
            ("Logger",          self.logger_node.run),
            ("Sound",           self.sound_node.run),
            ("EmotionDetect",   self.emotion_detection_node.run),
            ("FaceRecognition", self.face_recognition_node.run),
            ("AIAgent",         self.ai_agent_node.run),
            ("Voice",           self.voice_node.run),
        ]

        for name, target in threads:
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self.threads.append(t)
            print(f"[SYSTEM] ▶ {name}")

        kb = threading.Thread(target=self._keyboard, daemon=True, name="Keyboard")
        kb.start()

        print("\n" + "=" * 60)
        print("  ✅  ARIA IS READY")
        print("=" * 60)
        print("  📷  Camera: face recognition + emotion detection")
        print("  🎤  Voice: speak naturally — no wake word needed")
        print("  💬  Type: use the GUI input boxes")
        print("  🧠  AI: Mistral handles all conversations")
        print("  🔇  Emotion comments paused during conversations")
        print("=" * 60 + "\n")

        try:
            self.gui_node.run()
        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted")
        finally:
            self.shutdown()

    def _keyboard(self):
        """Debug terminal"""
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
                    self.gui_node.add_chat_message('ROBOT', text)
                elif low.startswith('state '):
                    self.command_queue.put({'type': 'CHANGE_STATE',
                                           'state': raw[6:].strip().upper()})
                elif low == 'list':
                    visitors = getattr(self.face_recognition_node, 'visitors', {})
                    if visitors:
                        print("\nKnown visitors:")
                        for vid, d in visitors.items():
                            print(f"  {vid}: {d.get('name','?')} — {d.get('visits',0)} visits")
                    else:
                        print("No visitors yet")
                elif low == 'reset':
                    self.ai_agent_node.history = []
                    self.ai_agent_node.current_name = None
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
        except Exception:
            pass

        for node, method in [
            (self.gui_node,                'shutdown'),
            (self.sensor_node,             'shutdown'),
            (self.logger_node,             'shutdown'),
            (self.sound_node,              'shutdown'),
            (self.emotion_detection_node,  'stop'),
            (self.face_recognition_node,   'stop'),
            (self.ai_agent_node,           'stop'),
            (self.voice_node,              'shutdown'),
            (self.state_manager,           'shutdown'),
        ]:
            try:
                getattr(node, method)()
            except Exception as e:
                print(f"[SYSTEM] {node.__class__.__name__}: {e}")

        print("[SYSTEM] Done.")


if __name__ == "__main__":
    try:
        ReceptionRobotSystem().start()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
