"""
Control Interface Node
Provides UI controls for testing and simulation
"""

import tkinter as tk
from tkinter import ttk
import queue

class ControlInterfaceNode:
    def __init__(self, command_queue):
        """
        Initialize Control Interface Node
        
        Args:
            command_queue: Queue to send commands
        """
        self.command_queue = command_queue
        
        self.root = tk.Tk()
        self.root.title("🎮 Control Interface")
        self.root.geometry("600x500")
        self.root.configure(bg='#2C3E50')
        
        self.create_ui()
        
        print("[CONTROL_INTERFACE] Node initialized")
    
    def create_ui(self):
        """Create control interface UI"""
        # Title
        title_frame = tk.Frame(self.root, bg='#34495E', height=50)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="Control Interface",
            font=('Arial', 20, 'bold'),
            bg='#34495E',
            fg='#ECF0F1'
        ).pack(pady=10)
        
        # Main content
        content_frame = tk.Frame(self.root, bg='#2C3E50')
        content_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Instructions
        tk.Label(
            content_frame,
            text="Click buttons to change robot state:",
            font=('Arial', 14),
            bg='#2C3E50',
            fg='#ECF0F1'
        ).pack(pady=10)
        
        # State control buttons
        button_frame = tk.Frame(content_frame, bg='#2C3E50')
        button_frame.pack(pady=20)
        
        states_buttons = [
            ('👋 Greeting', 'GREETING', '#27AE60'),
            ('👂 Listening', 'LISTENING', '#3498DB'),
            ('🤔 Thinking', 'THINKING', '#9B59B6'),
            ('💡 Helping', 'HELPING', '#F39C12'),
            ('😢 Error', 'ERROR', '#E74C3C'),
            ('😴 Busy', 'BUSY', '#95A5A6'),
            ('😲 Surprised', 'SURPRISED', '#E67E22'),
            ('😠 Angry', 'ANGRY', '#C0392B'),
            ('👋 Farewell', 'FAREWELL', '#16A085'),
            ('😐 Idle', 'IDLE', '#7F8C8D')
        ]
        
        row = 0
        col = 0
        for text, state, color in states_buttons:
            btn = tk.Button(
                button_frame,
                text=text,
                font=('Arial', 11),
                bg=color,
                fg='white',
                width=15,
                height=2,
                command=lambda s=state: self.send_state_command(s),
                cursor='hand2',
                relief='raised',
                bd=2
            )
            btn.grid(row=row, column=col, padx=5, pady=5)
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        # Separator
        ttk.Separator(content_frame, orient='horizontal').pack(fill='x', pady=20)
        
        # System controls
        system_frame = tk.Frame(content_frame, bg='#2C3E50')
        system_frame.pack(pady=10)
        
        tk.Button(
            system_frame,
            text="📊 Get Current State",
            font=('Arial', 11),
            bg='#3498DB',
            fg='white',
            width=20,
            command=self.get_state,
            cursor='hand2'
        ).pack(pady=5)
        
        tk.Button(
            system_frame,
            text="🔄 Reset to Idle",
            font=('Arial', 11),
            bg='#95A5A6',
            fg='white',
            width=20,
            command=lambda: self.send_state_command('IDLE'),
            cursor='hand2'
        ).pack(pady=5)
    
    def send_state_command(self, state):
        """Send state change command"""
        command = {
            'type': 'CHANGE_STATE',
            'state': state
        }
        self.command_queue.put(command)
        print(f"[CONTROL_INTERFACE] Sent command: {state}")
    
    def get_state(self):
        """Request current state"""
        command = {
            'type': 'GET_STATE'
        }
        self.command_queue.put(command)
    
    def run(self):
        """Start the control interface"""
        print("[CONTROL_INTERFACE] Node running...")
        self.root.mainloop()
    
    def shutdown(self):
        """Shutdown the control interface"""
        self.root.quit()
