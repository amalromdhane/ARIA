"""
Logger Node
Logs all system events and state changes
"""

import threading
import time
import queue
from datetime import datetime
import os

class LoggerNode:
    def __init__(self, state_queue_copy, log_file='logs/robot.log'):
        """
        Initialize Logger Node
        
        Args:
            state_queue_copy: Queue to receive state updates for logging
            log_file: Path to log file
        """
        self.state_queue = state_queue_copy
        self.log_file = log_file
        self.running = True
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        print("[LOGGER_NODE] Node initialized")
        self.log("System started")
    
    def log(self, message, level='INFO'):
        """
        Log a message
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # Print to console
        print(f"[LOGGER] {log_entry}")
        
        # Write to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"[LOGGER] ERROR: Could not write to log file: {e}")
    
    def process_logs(self):
        """Process incoming log messages"""
        while self.running:
            try:
                if not self.state_queue.empty():
                    state = self.state_queue.get_nowait()
                    self.log(f"State changed to: {state}")
                
                time.sleep(0.1)
            except queue.Empty:
                pass
    
    def run(self):
        """Start the logger node"""
        print("[LOGGER_NODE] Node running...")
        self.process_logs()
    
    def shutdown(self):
        """Shutdown the logger node"""
        self.running = False
        self.log("System shutdown")
        print("[LOGGER_NODE] Shutting down...")
