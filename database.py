"""
Database module - SQLite replacement for pickle persistence
Handles visitors, conversations, and system state
"""

import sqlite3
import json
import numpy as np
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import pickle  # Only for encoding face encodings


class VisitorDatabase:
    def __init__(self, db_path: str = 'aria_database.db'):
        self.db_path = db_path
        self._init_db()
        
    def _get_connection(self):
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize tables if they don't exist"""
        with self._get_connection() as conn:
            # Visitors table with face encoding storage
            conn.execute('''
                CREATE TABLE IF NOT EXISTS visitors (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    encoding BLOB NOT NULL,  -- numpy array as bytes
                    visits INTEGER DEFAULT 1,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT  -- JSON for extensibility
                )
            ''')
            
            # Conversation history for context recovery
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    role TEXT CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    emotion TEXT,
                    FOREIGN KEY (visitor_id) REFERENCES visitors(id)
                )
            ''')
            
            # System state for crash recovery
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Interaction logs for analytics
            conn.execute('''
                CREATE TABLE IF NOT EXISTS interaction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id TEXT,
                    event_type TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (visitor_id) REFERENCES visitors(id)
                )
            ''')
            
            conn.commit()
        print(f"[DB] Initialized: {self.db_path}")
    
    # ── Visitor Operations ─────────────────────────────────────────
    
    def encode_face(self, encoding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for storage"""
        return pickle.dumps(encoding, protocol=pickle.HIGHEST_PROTOCOL)
    
    def decode_face(self, data: bytes) -> np.ndarray:
        """Convert bytes back to numpy array"""
        return pickle.loads(data)
    
    def get_visitor_by_encoding(self, encoding: np.ndarray, threshold: float = 0.55) -> Optional[Dict]:
        """Find visitor by face encoding using Euclidean distance"""
        try:
            import face_recognition
        except ImportError:
            return None
            
        with self._get_connection() as conn:
            rows = conn.execute('SELECT * FROM visitors').fetchall()
            
        if not rows:
            return None
            
        known_encodings = [self.decode_face(row['encoding']) for row in rows]
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = int(distances.argmin())
        
        if distances[best_idx] < threshold:
            row = rows[best_idx]
            return {
                'id': row['id'],
                'name': row['name'],
                'encoding': self.decode_face(row['encoding']),
                'visits': row['visits'],
                'first_seen': row['first_seen'],
                'last_seen': row['last_seen'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            }
        return None
    
    def add_visitor(self, visitor_id: str, name: str, encoding: np.ndarray, 
                    metadata: Optional[Dict] = None) -> Dict:
        """Register new visitor"""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO visitors (id, name, encoding, visits, first_seen, last_seen, metadata)
                VALUES (?, ?, ?, 1, ?, ?, ?)
            ''', (
                visitor_id, 
                name, 
                self.encode_face(encoding),
                now, 
                now,
                json.dumps(metadata or {})
            ))
            conn.commit()
        
        print(f"[DB] Added visitor: {visitor_id} ({name})")
        return self.get_visitor_by_id(visitor_id)
    
    def update_visitor(self, visitor_id: str, **updates) -> Optional[Dict]:
        """Update visitor fields (name, visits, last_seen, etc.)"""
        allowed = {'name', 'visits', 'last_seen', 'metadata'}
        fields = {k: v for k, v in updates.items() if k in allowed}
        
        if not fields:
            return None
            
        # Handle metadata JSON serialization
        if 'metadata' in fields and isinstance(fields['metadata'], dict):
            fields['metadata'] = json.dumps(fields['metadata'])
        
        set_clause = ', '.join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values()) + [visitor_id]
        
        with self._get_connection() as conn:
            conn.execute(f'''
                UPDATE visitors SET {set_clause} WHERE id = ?
            ''', values)
            conn.commit()
        
        return self.get_visitor_by_id(visitor_id)
    
    def increment_visits(self, visitor_id: str) -> Optional[Dict]:
        """Increment visit count and update last_seen"""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute('''
                UPDATE visitors 
                SET visits = visits + 1, last_seen = ? 
                WHERE id = ?
            ''', (now, visitor_id))
            conn.commit()
        return self.get_visitor_by_id(visitor_id)
    
    def get_visitor_by_id(self, visitor_id: str) -> Optional[Dict]:
        """Get visitor by ID"""
        with self._get_connection() as conn:
            row = conn.execute(
                'SELECT * FROM visitors WHERE id = ?', 
                (visitor_id,)
            ).fetchone()
            
        if not row:
            return None
            
        return {
            'id': row['id'],
            'name': row['name'],
            'encoding': self.decode_face(row['encoding']),
            'visits': row['visits'],
            'first_seen': row['first_seen'],
            'last_seen': row['last_seen'],
            'metadata': json.loads(row['metadata']) if row['metadata'] else {}
        }
    
    def get_all_visitors(self) -> List[Dict]:
        """Get all visitors (for admin/debug)"""
        with self._get_connection() as conn:
            rows = conn.execute('SELECT * FROM visitors ORDER BY last_seen DESC').fetchall()
        
        return [{
            'id': r['id'],
            'name': r['name'],
            'visits': r['visits'],
            'first_seen': r['first_seen'],
            'last_seen': r['last_seen'],
            'metadata': json.loads(r['metadata']) if r['metadata'] else {}
        } for r in rows]
    
    # ── Conversation Operations ────────────────────────────────────
    
    def log_message(self, visitor_id: Optional[str], role: str, content: str, 
                    emotion: Optional[str] = None):
        """Log conversation message"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO conversations (visitor_id, role, content, emotion)
                VALUES (?, ?, ?, ?)
            ''', (visitor_id, role, content, emotion))
            conn.commit()
    
    def get_conversation_history(self, visitor_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for context"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM conversations 
                WHERE visitor_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (visitor_id, limit)).fetchall()
        
        return [{
            'role': r['role'],
            'content': r['content'],
            'emotion': r['emotion'],
            'timestamp': r['timestamp']
        } for r in reversed(rows)]  # Oldest first
    
    # ── System State Operations ────────────────────────────────────
    
    def save_state(self, key: str, value: str):
        """Save system state (for crash recovery)"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))
            conn.commit()
    
    def get_state(self, key: str) -> Optional[str]:
        """Retrieve system state"""
        with self._get_connection() as conn:
            row = conn.execute(
                'SELECT value FROM system_state WHERE key = ?', 
                (key,)
            ).fetchone()
        return row['value'] if row else None
    
    def clear_state(self, key: str):
        """Clear specific state key"""
        with self._get_connection() as conn:
            conn.execute('DELETE FROM system_state WHERE key = ?', (key,))
            conn.commit()
    
    # ── Analytics Operations ───────────────────────────────────────
    
    def log_event(self, visitor_id: Optional[str], event_type: str, details: Optional[Dict] = None):
        """Log system events"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO interaction_logs (visitor_id, event_type, details)
                VALUES (?, ?, ?)
            ''', (visitor_id, event_type, json.dumps(details) if details else None))
            conn.commit()
    
    def get_daily_stats(self) -> Dict:
        """Get today's interaction statistics"""
        today = datetime.now().strftime('%Y-%m-%d')
        with self._get_connection() as conn:
            new_visitors = conn.execute('''
                SELECT COUNT(*) as count FROM visitors 
                WHERE date(first_seen) = ?
            ''', (today,)).fetchone()['count']
            
            total_visits = conn.execute('''
                SELECT COUNT(*) as count FROM interaction_logs 
                WHERE date(timestamp) = ? AND event_type = 'visit'
            ''', (today,)).fetchone()['count']
            
            conversations = conn.execute('''
                SELECT COUNT(DISTINCT visitor_id) as count FROM conversations 
                WHERE date(timestamp) = ?
            ''', (today,)).fetchone()['count']
        
        return {
            'new_visitors': new_visitors,
            'total_visits': total_visits,
            'conversations': conversations,
            'date': today
        }
    
    # ── Migration from Pickle ──────────────────────────────────────
    
    def migrate_from_pickle(self, pickle_path: str = 'visitor_database.pkl'):
        """One-time migration from old pickle format"""
        if not os.path.exists(pickle_path):
            print(f"[DB] No pickle file found at {pickle_path}")
            return
        
        try:
            with open(pickle_path, 'rb') as f:
                old_data = pickle.load(f)
            
            print(f"[DB] Migrating {len(old_data)} visitors from pickle...")
            
            for vid, data in old_data.items():
                # Check if already exists
                if self.get_visitor_by_id(vid):
                    continue
                    
                self.add_visitor(
                    visitor_id=vid,
                    name=data.get('name', vid),
                    encoding=data['encoding'],
                    metadata={
                        'migrated_from_pickle': True,
                        'original_visits': data.get('visits', 1)
                    }
                )
                # Update visits count to match original
                visits = data.get('visits', 1)
                if visits > 1:
                    with self._get_connection() as conn:
                        conn.execute(
                            'UPDATE visitors SET visits = ? WHERE id = ?',
                            (visits, vid)
                        )
                        conn.commit()
            
            print(f"[DB] Migration complete! Backup saved as {pickle_path}.backup")
            os.rename(pickle_path, f"{pickle_path}.backup")
            
        except Exception as e:
            print(f"[DB] Migration failed: {e}")


# Singleton instance for easy import
_db_instance = None

def get_database(db_path: str = 'aria_database.db') -> VisitorDatabase:
    """Get or create database singleton"""
    global _db_instance
    if _db_instance is None:
        _db_instance = VisitorDatabase(db_path)
    return _db_instance
