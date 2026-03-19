"""
Web GUI Node - Flask + Socket.IO replacement for Tkinter
Accessible from any device on the network
"""

import threading
import time
import queue
import os
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit
import logging


# Suppress Flask logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class WebGUINode:
    def __init__(self, state_queue, emotion_queue, command_queue, port=5000):
        self.state_queue = state_queue
        self.emotion_queue = emotion_queue
        self.command_queue = command_queue
        self.port = port
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'aria-secret-key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        self.current_emotion = 'NEUTRAL'
        self.visitor_info = {
            'name': '—',
            'visits': '—',
            'mood': '—'
        }
        self.messages = []
        self.running = False
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Define Flask routes and Socket.IO events"""
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @self.socketio.on('connect')
        def handle_connect():
            """Send current state to new client"""
            emit('emotion_update', {'emotion': self.current_emotion})
            emit('visitor_update', self.visitor_info)
            emit('chat_history', {'messages': self.messages})
        
        @self.socketio.on('set_name')
        def handle_set_name(data):
            name = data.get('name', '').strip()
            if name:
                self.command_queue.put({'type': 'SET_NAME', 'name': name})
                self.add_chat_message('USER', name)
        
        @self.socketio.on('send_message')
        def handle_message(data):
            text = data.get('text', '').strip()
            if text:
                self.command_queue.put({'type': 'USER_MESSAGE', 'text': text})
                self.add_chat_message('USER', text)
        
        @self.socketio.on('request_state')
        def handle_state_request():
            emit('emotion_update', {'emotion': self.current_emotion})
            emit('visitor_update', self.visitor_info)
    
    def add_chat_message(self, sender, msg):
        """Add message to chat history and broadcast"""
        self.messages.append({
            'sender': sender,
            'text': msg,
            'time': time.strftime('%H:%M')
        })
        # Keep last 50 messages
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]
        
        self.socketio.emit('new_message', {
            'sender': sender,
            'text': msg,
            'time': time.strftime('%H:%M')
        })
    
    def update_visitor_info(self, name=None, visits=None, mood=None):
        """Update visitor panel and broadcast"""
        if name is not None:
            self.visitor_info['name'] = name
        if visits is not None:
            self.visitor_info['visits'] = str(visits)
        if mood is not None:
            self.visitor_info['mood'] = mood.capitalize() if mood != '—' else '—'
        
        self.socketio.emit('visitor_update', self.visitor_info)
    
    def _poll_queues(self):
        """Background thread: poll queues for updates"""
        while self.running:
            try:
                # Check emotion queue
                if not self.emotion_queue.empty():
                    emotion = self.emotion_queue.get_nowait()
                    self.current_emotion = emotion
                    self.socketio.emit('emotion_update', {'emotion': emotion})
            except:
                pass
            
            time.sleep(0.05)
    
    def run(self):
        """Start Flask server"""
        self.running = True
        
        # Start queue polling thread
        poll_thread = threading.Thread(target=self._poll_queues, daemon=True)
        poll_thread.start()
        
        print(f"[WEB_GUI] Starting server on http://0.0.0.0:{self.port}")
        print(f"[WEB_GUI] Open browser to: http://localhost:{self.port}")
        
        self.socketio.run(
            self.app, 
            host='0.0.0.0', 
            port=self.port, 
            debug=False,
            use_reloader=False,
            log_output=False
        )
    
    def shutdown(self):
        self.running = False


# HTML Template based on preview.html with live updates
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARIA — Reception Assistant</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        :root {
            --bg: #F0F2F5;
            --surface: #FFFFFF;
            --border: #E2E5EA;
            --ink: #0F1923;
            --ink2: #4B5563;
            --ink3: #9CA3AF;
            --blue: #1A56DB;
            --blue-bg: #EBF0FC;
            --green: #0B7E57;
            --green-bg: #E6F4EF;
            --face-bg: #F7F8FB;
            --shadow: 0 1px 3px rgba(0,0,0,.08), 0 4px 16px rgba(0,0,0,.06);
            
            /* Emotion colors */
            --c-neutral: #2563EB;
            --c-happy: #059669;
            --c-sad: #3B82F6;
            --c-angry: #EF4444;
            --c-surprised: #8B5CF6;
            --c-confused: #F59E0B;
            --c-tired: #6B7280;
            --c-excited: #10B981;
            --c-attentive: #2563EB;
        }
        
        body {
            font-family: 'DM Sans', sans-serif;
            background: var(--bg);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        /* Top Bar */
        .topbar {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            height: 56px;
            display: flex;
            align-items: center;
            padding: 0 28px;
            gap: 12px;
            flex-shrink: 0;
        }
        .topbar-logo {
            width: 32px; height: 32px;
            background: var(--blue);
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            color: white; font-size: 16px; font-family: 'DM Serif Display', serif;
        }
        .topbar-title {
            font-family: 'DM Serif Display', serif;
            font-size: 18px;
            color: var(--ink);
            letter-spacing: -0.3px;
        }
        .topbar-divider { width: 1px; height: 20px; background: var(--border); margin: 0 8px; }
        .topbar-sub { font-size: 13px; color: var(--ink3); }
        .topbar-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
        .badge-green {
            background: var(--green-bg);
            color: var(--green);
            font-size: 11px; font-weight: 600;
            padding: 4px 10px; border-radius: 99px;
            display: flex; align-items: center; gap: 5px;
        }
        .badge-green::before {
            content: '';
            width: 6px; height: 6px;
            background: var(--green);
            border-radius: 50%;
        }
        .clock { font-size: 13px; color: var(--ink3); font-weight: 500; }
        
        /* Main Grid */
        .main {
            flex: 1;
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 20px;
            padding: 20px 28px;
            overflow: hidden;
        }
        
        /* Left Column */
        .left { display: flex; flex-direction: column; gap: 16px; }
        
        .face-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: var(--shadow);
            flex: 1;
        }
        
        .face-wrap {
            width: 280px; height: 280px;
            background: var(--face-bg);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            margin: 8px 0 20px;
            position: relative;
            border: 2px solid var(--border);
            transition: all 0.3s ease;
        }
        
        .emotion-pill {
            background: var(--blue-bg);
            color: var(--blue);
            font-size: 11px; font-weight: 600;
            padding: 4px 14px; border-radius: 99px;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }
        
        .face-caption {
            font-size: 13px;
            color: var(--ink3);
            text-align: center;
            line-height: 1.5;
        }
        
        /* Visitor Info */
        .info-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
        }
        .info-card-title {
            font-size: 11px; font-weight: 600;
            color: var(--ink3);
            letter-spacing: 0.6px;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        .info-row {
            display: flex; align-items: center;
            gap: 10px; margin-bottom: 10px;
        }
        .info-icon {
            width: 32px; height: 32px;
            background: var(--blue-bg);
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 15px;
            flex-shrink: 0;
        }
        .info-label { font-size: 11px; color: var(--ink3); }
        .info-value { font-size: 13px; font-weight: 500; color: var(--ink); }
        
        /* Right Column */
        .right { display: flex; flex-direction: column; gap: 16px; overflow: hidden; }
        
        .chat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: var(--shadow);
            display: flex; flex-direction: column;
            flex: 1; overflow: hidden;
        }
        .chat-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; justify-content: space-between;
        }
        .chat-title {
            font-size: 14px; font-weight: 600; color: var(--ink);
        }
        .chat-count {
            font-size: 11px; color: var(--ink3);
            background: var(--bg); padding: 3px 10px; border-radius: 99px;
        }
        .chat-body {
            flex: 1; overflow-y: auto; padding: 16px 20px;
            display: flex; flex-direction: column; gap: 14px;
        }
        .chat-body::-webkit-scrollbar { width: 4px; }
        .chat-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
        
        .msg { display: flex; flex-direction: column; gap: 4px; }
        .msg-sender {
            font-size: 11px; font-weight: 600; color: var(--ink3);
            letter-spacing: 0.3px;
        }
        .msg-sender.robot { color: var(--blue); }
        .msg-bubble {
            background: var(--bg);
            border-radius: 0 12px 12px 12px;
            padding: 10px 14px;
            font-size: 13px; color: var(--ink);
            line-height: 1.55;
            max-width: 85%;
            border: 1px solid var(--border);
        }
        .msg-bubble.robot {
            background: var(--blue-bg);
            border-color: #C7D6F7;
            border-radius: 12px 12px 12px 0;
        }
        .msg.user { align-items: flex-end; }
        .msg.user .msg-bubble {
            border-radius: 12px 0 12px 12px;
        }
        
        /* Input Row */
        .input-row {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 16px;
            flex-shrink: 0;
        }
        
        .input-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: var(--shadow);
        }
        .input-label {
            font-size: 11px; font-weight: 600;
            color: var(--ink3); letter-spacing: 0.5px;
            text-transform: uppercase; margin-bottom: 8px;
        }
        .input-field-wrap {
            display: flex; gap: 8px;
        }
        .input-field {
            flex: 1;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 9px 14px;
            font-family: 'DM Sans', sans-serif;
            font-size: 13px; color: var(--ink);
            outline: none;
            transition: border-color .15s;
        }
        .input-field:focus { border-color: var(--blue); }
        .input-field::placeholder { color: var(--ink3); }
        .btn {
            background: var(--blue);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 9px 18px;
            font-family: 'DM Sans', sans-serif;
            font-size: 13px; font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
            transition: background .15s;
        }
        .btn:hover { background: #1446B8; }
        .btn.secondary {
            background: transparent;
            color: var(--blue);
            border: 1px solid var(--blue);
        }
        
        /* Typing indicator */
        .typing {
            display: none;
            align-items: center;
            gap: 4px;
            padding: 10px 14px;
            background: var(--blue-bg);
            border-radius: 12px 12px 12px 0;
            border: 1px solid #C7D6F7;
            width: fit-content;
        }
        .typing.active { display: flex; }
        .typing-dot {
            width: 6px; height: 6px;
            background: var(--blue);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        /* Mobile responsive */
        @media (max-width: 900px) {
            .main { grid-template-columns: 1fr; }
            .left { display: none; } /* Hide face on mobile, show chat only */
        }
    </style>
</head>
<body>
    <!-- Top Bar -->
    <div class="topbar">
        <div class="topbar-logo">A</div>
        <span class="topbar-title">ARIA</span>
        <div class="topbar-divider"></div>
        <span class="topbar-sub">Reception Assistant</span>
        <div class="topbar-right">
            <span class="badge-green">System Online</span>
            <span class="clock" id="clock">09:41</span>
        </div>
    </div>

    <!-- Main -->
    <div class="main">
        <!-- Left -->
        <div class="left">
            <div class="face-card">
                <div class="emotion-pill" id="emotion-pill">STANDBY</div>
                
                <div class="face-wrap" id="face-wrap">
                    <svg class="face-svg" width="200" height="200" viewBox="0 0 140 140" fill="none" xmlns="http://www.w3.org/2000/svg" id="face-svg">
                        <!-- Head -->
                        <circle cx="70" cy="70" r="56" fill="#EBF0FC" stroke="#1A56DB" stroke-width="2.5" id="head-circle"/>
                        <!-- Ears -->
                        <rect x="6" y="52" width="14" height="26" rx="7" fill="#EBF0FC" stroke="#1A56DB" stroke-width="2" class="ears"/>
                        <rect x="120" y="52" width="14" height="26" rx="7" fill="#EBF0FC" stroke="#1A56DB" stroke-width="2" class="ears"/>
                        <!-- Eyes -->
                        <circle cx="48" cy="60" r="13" fill="white" stroke="#1A56DB" stroke-width="2" class="eyes"/>
                        <circle cx="92" cy="60" r="13" fill="white" stroke="#1A56DB" stroke-width="2" class="eyes"/>
                        <circle cx="48" cy="60" r="6" fill="#1A56DB" class="pupils"/>
                        <circle cx="92" cy="60" r="6" fill="#1A56DB" class="pupils"/>
                        <circle cx="45" cy="57" r="2.5" fill="white" class="highlights"/>
                        <circle cx="89" cy="57" r="2.5" fill="white" class="highlights"/>
                        <!-- Smile -->
                        <path d="M44 90 Q70 110 96 90" stroke="#1A56DB" stroke-width="3" stroke-linecap="round" fill="none" id="mouth"/>
                        <!-- Nose dot -->
                        <circle cx="70" cy="78" r="3" fill="#1A56DB" opacity="0.4"/>
                    </svg>
                </div>
                
                <div class="face-caption" id="face-caption">Awaiting visitor</div>
            </div>

            <!-- Visitor Info -->
            <div class="info-card">
                <div class="info-card-title">Visitor Info</div>
                <div class="info-row">
                    <div class="info-icon">👤</div>
                    <div>
                        <div class="info-label">Name</div>
                        <div class="info-value" id="info-name">—</div>
                    </div>
                </div>
                <div class="info-row">
                    <div class="info-icon">📅</div>
                    <div>
                        <div class="info-label">Visit Count</div>
                        <div class="info-value" id="info-visits">—</div>
                    </div>
                </div>
                <div class="info-row">
                    <div class="info-icon">😊</div>
                    <div>
                        <div class="info-label">Detected Mood</div>
                        <div class="info-value" id="info-mood">—</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right -->
        <div class="right">
            <div class="chat-card">
                <div class="chat-header">
                    <span class="chat-title">Conversation</span>
                    <span class="chat-count" id="msg-count">0 messages</span>
                </div>
                <div class="chat-body" id="chat-body">
                    <!-- Messages appear here -->
                    
                    <!-- Typing indicator -->
                    <div class="msg" id="typing-indicator">
                        <span class="msg-sender robot">ARIA</span>
                        <div class="typing">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Input Row -->
            <div class="input-row">
                <div class="input-card">
                    <div class="input-label">Your Name</div>
                    <div class="input-field-wrap">
                        <input class="input-field" type="text" id="name-input" placeholder="e.g. Malak">
                        <button class="btn secondary" onclick="setName()">Save</button>
                    </div>
                </div>

                <div class="input-card">
                    <div class="input-label">Send a Message</div>
                    <div class="input-field-wrap">
                        <input class="input-field" type="text" id="msg-input" placeholder="Ask a question or type a command…">
                        <button class="btn" onclick="sendMessage()">Send →</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let msgCount = 0;
        
        // Emotion configurations
        const emotions = {
            'NEUTRAL':   { color: '#2563EB', pill: 'STANDBY', caption: 'Awaiting visitor' },
            'HAPPY':     { color: '#059669', pill: 'HAPPY', caption: 'Glad to see you!' },
            'SAD':       { color: '#3B82F6', pill: 'CONCERNED', caption: 'Here to help.' },
            'ANGRY':     { color: '#EF4444', pill: 'ALERT', caption: 'Let me resolve this.' },
            'SURPRISED': { color: '#8B5CF6', pill: 'SURPRISED', caption: 'That is unexpected!' },
            'CONFUSED':  { color: '#F59E0B', pill: 'THINKING', caption: 'Processing your request…' },
            'TIRED':     { color: '#6B7280', pill: 'RESTING', caption: 'Brief pause.' },
            'EXCITED':   { color: '#10B981', pill: 'ENGAGED', caption: 'Happy to assist!' },
            'ATTENTIVE': { color: '#2563EB', pill: 'LISTENING', caption: 'I am listening.' }
        };
        
        // Clock
        function updateClock() {
            const now = new Date();
            document.getElementById('clock').textContent = 
                now.getHours().toString().padStart(2,'0') + ':' + 
                now.getMinutes().toString().padStart(2,'0');
        }
        setInterval(updateClock, 1000);
        updateClock();
        
        // Socket events
        socket.on('connect', () => {
            console.log('Connected to ARIA');
            socket.emit('request_state');
        });
        
        socket.on('emotion_update', (data) => {
            updateEmotion(data.emotion);
        });
        
        socket.on('visitor_update', (data) => {
            document.getElementById('info-name').textContent = data.name;
            document.getElementById('info-visits').textContent = data.visits;
            document.getElementById('info-mood').textContent = data.mood;
        });
        
        socket.on('chat_history', (data) => {
            document.getElementById('chat-body').innerHTML = `
                <div class="msg" id="typing-indicator">
                    <span class="msg-sender robot">ARIA</span>
                    <div class="typing">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            `;
            data.messages.forEach(m => addMessage(m.sender, m.text, m.time));
        });
        
        socket.on('new_message', (data) => {
            addMessage(data.sender, data.text, data.time);
            // Hide typing indicator when robot responds
            if (data.sender === 'ROBOT') {
                document.querySelector('.typing').classList.remove('active');
            }
        });
        
        function updateEmotion(emotion) {
            const config = emotions[emotion] || emotions['NEUTRAL'];
            const pill = document.getElementById('emotion-pill');
            const caption = document.getElementById('face-caption');
            const head = document.getElementById('head-circle');
            const faceWrap = document.getElementById('face-wrap');
            
            pill.textContent = config.pill;
            pill.style.background = config.color + '20';
            pill.style.color = config.color;
            caption.textContent = config.caption;
            
            // Update SVG colors
            head.setAttribute('stroke', config.color);
            document.querySelectorAll('.ears').forEach(e => e.setAttribute('stroke', config.color));
            document.querySelectorAll('.eyes').forEach(e => e.setAttribute('stroke', config.color));
            document.querySelectorAll('.pupils').forEach(e => e.setAttribute('fill', config.color));
            document.getElementById('mouth').setAttribute('stroke', config.color);
            
            faceWrap.style.borderColor = config.color + '40';
        }
        
        function addMessage(sender, text, time) {
            const chatBody = document.getElementById('chat-body');
            const typingIndicator = document.getElementById('typing-indicator');
            
            const div = document.createElement('div');
            div.className = 'msg' + (sender === 'USER' ? ' user' : '');
            div.innerHTML = `
                <span class="msg-sender ${sender === 'ROBOT' ? 'robot' : ''}">${sender === 'ROBOT' ? 'ARIA' : 'You'}</span>
                <div class="msg-bubble ${sender === 'ROBOT' ? 'robot' : ''}">${escapeHtml(text)}</div>
            `;
            
            chatBody.insertBefore(div, typingIndicator);
            chatBody.scrollTop = chatBody.scrollHeight;
            
            msgCount++;
            document.getElementById('msg-count').textContent = 
                msgCount + ' message' + (msgCount !== 1 ? 's' : '');
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function setName() {
            const input = document.getElementById('name-input');
            const name = input.value.trim();
            if (name) {
                socket.emit('set_name', {name: name});
                input.value = '';
                // Show typing indicator
                document.querySelector('.typing').classList.add('active');
            }
        }
        
        function sendMessage() {
            const input = document.getElementById('msg-input');
            const text = input.value.trim();
            if (text) {
                socket.emit('send_message', {text: text});
                input.value = '';
                // Show typing indicator
                document.querySelector('.typing').classList.add('active');
            }
        }
        
        // Enter key handlers
        document.getElementById('name-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') setName();
        });
        document.getElementById('msg-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
'''
