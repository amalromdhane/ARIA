# database_node.py - Add this new node
"""
Database Node — SQLite-based visitor storage with conversation history
"""

import sqlite3
import json
import os
import time
import threading
import queue
from datetime import datetime


class DatabaseNode:
    def __init__(self, db_path='data/aria_visitors.db'):
        self.db_path = db_path
        self.command_queue = queue.Queue()
        self.running = False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._init_db()
        print(f"[DATABASE] Initialized: {db_path}")

    def _init_db(self):
        """Create tables if not exist"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Visitors table
        c.execute('''
            CREATE TABLE IF NOT EXISTS visitors (
                visitor_id TEXT PRIMARY KEY,
                name TEXT,
                encoding BLOB,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                visit_count INTEGER DEFAULT 0,
                preferences TEXT  -- JSON: {"topics": [], "notes": ""}
            )
        ''')
        
        # Conversations table
        c.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id TEXT,
                timestamp TIMESTAMP,
                role TEXT,  -- 'user' or 'assistant'
                message TEXT,
                emotion TEXT,  -- detected emotion during message
                FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id)
            )
        ''')
        
        # Visits table (session tracking)
        c.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                purpose TEXT,
                satisfaction INTEGER,  -- 1-5 rating if asked
                FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def get_or_create_visitor(self, visitor_id, face_encoding=None):
        """Get existing visitor or create new one"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT * FROM visitors WHERE visitor_id = ?', (visitor_id,))
        row = c.fetchone()
        
        if row:
            # Update last seen
            c.execute('''
                UPDATE visitors 
                SET last_seen = ?, visit_count = visit_count + 1
                WHERE visitor_id = ?
            ''', (datetime.now(), visitor_id))
            conn.commit()
            
            visitor = {
                'id': row[0],
                'name': row[1],
                'visit_count': row[5] + 1,
                'preferences': json.loads(row[6]) if row[6] else {}
            }
            is_new = False
        else:
            # Create new visitor
            encoding_bytes = face_encoding.tobytes() if face_encoding is not None else None
            
            c.execute('''
                INSERT INTO visitors 
                (visitor_id, name, encoding, first_seen, last_seen, visit_count, preferences)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            ''', (visitor_id, visitor_id, encoding_bytes, datetime.now(), 
                  datetime.now(), json.dumps({})))
            conn.commit()
            
            visitor = {
                'id': visitor_id,
                'name': visitor_id,
                'visit_count': 1,
                'preferences': {}
            }
            is_new = True
        
        conn.close()
        return visitor, is_new

    def update_name(self, visitor_id, name):
        """Update visitor name"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('UPDATE visitors SET name = ? WHERE visitor_id = ?', 
                  (name, visitor_id))
        conn.commit()
        conn.close()
        print(f"[DATABASE] Updated name: {name}")

    def log_conversation(self, visitor_id, role, message, emotion=None):
        """Store conversation message"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO conversations (visitor_id, timestamp, role, message, emotion)
            VALUES (?, ?, ?, ?, ?)
        ''', (visitor_id, datetime.now(), role, message, emotion))
        conn.commit()
        conn.close()

    def get_conversation_history(self, visitor_id, limit=10):
        """Get recent conversation history for context"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT role, message, emotion, timestamp 
            FROM conversations 
            WHERE visitor_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (visitor_id, limit))
        
        rows = c.fetchall()
        conn.close()
        
        # Return in chronological order
        return [{
            'role': r[0],
            'message': r[1],
            'emotion': r[2],
            'time': r[3]
        } for r in reversed(rows)]

    def start_visit(self, visitor_id, purpose=None):
        """Record visit start"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO visits (visitor_id, start_time, purpose)
            VALUES (?, ?, ?)
        ''', (visitor_id, datetime.now(), purpose))
        conn.commit()
        visit_id = c.lastrowid
        conn.close()
        return visit_id

    def end_visit(self, visit_id, satisfaction=None):
        """Record visit end"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            UPDATE visits SET end_time = ?, satisfaction = ?
            WHERE id = ?
        ''', (datetime.now(), satisfaction, visit_id))
        conn.commit()
        conn.close()

    def get_visitor_stats(self, visitor_id):
        """Get statistics for a visitor"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Total messages
        c.execute('SELECT COUNT(*) FROM conversations WHERE visitor_id = ?', 
                  (visitor_id,))
        msg_count = c.fetchone()[0]
        
        # Average emotion (simplified)
        c.execute('''
            SELECT emotion, COUNT(*) FROM conversations 
            WHERE visitor_id = ? AND emotion IS NOT NULL
            GROUP BY emotion
        ''', (visitor_id,))
        emotions = {r[0]: r[1] for r in c.fetchall()}
        
        # Visit history
        c.execute('''
            SELECT start_time, end_time, purpose 
            FROM visits WHERE visitor_id = ?
            ORDER BY start_time DESC
        ''', (visitor_id,))
        visits = [{
            'start': r[0], 'end': r[1], 'purpose': r[2]
        } for r in c.fetchall()]
        
        conn.close()
        
        return {
            'message_count': msg_count,
            'emotion_distribution': emotions,
            'visit_history': visits
        }

    def run(self):
        """Process database commands"""
        self.running = True
        while self.running:
            try:
                cmd = self.command_queue.get(timeout=0.5)
                # Handle commands here if needed
            except queue.Empty:
                pass

    def stop(self):
        self.running = False
