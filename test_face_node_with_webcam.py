#!/usr/bin/env python3
import queue
import time
import cv2
import threading

# ── Paste your two classes here ───────────────────────────────────────────────
# PersonSession
# FaceRecognitionNode

# (copy-paste the complete PersonSession class and FaceRecognitionNode class from your code)

# ── Dummy replacements for ROS parts ──────────────────────────────────────────
class DummyStateManager:
    def __init__(self):
        self.command_queue = queue.Queue()
        self.gui_node = type('DummyGUI', (), {
            'add_chat_message': lambda s, who, txt: print(f"  [GUI] {who}: {txt}"),
            'update_visitor_info': lambda *a, **k: None
        })()
        self.ai_agent_node = None

    def change_state(self, new_state):
        print(f"  [STATE] → {new_state}")

    def command_queue_put(self, item):
        self.command_queue.put(item)

# ── Main test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Simple Face Recognition Standalone Test (webcam) ===\n")

    frame_queue = queue.Queue(maxsize=5)
    sound_queue = queue.Queue()           # not really used here
    state_manager = DummyStateManager()

    # Create the node (use your real db path)
    node = FaceRecognitionNode(
        frame_queue=frame_queue,
        sound_emotion_queue=sound_queue,
        state_manager=state_manager,
        db_path="/home/amal/ros2_face_recognition/data/face_database.db",  # ← your path
        match_threshold=0.55,          # feel free to tune
        absence_threshold=5.0,         # smaller = faster goodbye during test
        recognition_interval=1.5       # faster recognition during test
    )

    # Start the node's processing loop in background
    node_thread = threading.Thread(target=node.run, daemon=True)
    node_thread.start()

    print("Opening webcam... (press 'q' in the OpenCV window to stop)\n")

    cap = cv2.VideoCapture(0)  # 0 = default webcam, try 1 or 2 if needed
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        node.stop()
        exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Optional: show what the node sees
            cv2.imshow("Webcam (node input)", frame)

            # Put frame into the queue (drop old if full)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except:
                    pass
            frame_queue.put(frame.copy())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.033)  # ~30 fps

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        node.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Test finished.")