"""
Lightweight Persistence Manager
==============================

SQLite + File System hybrid approach for efficient session state management.
Solves Streamlit state explosion by storing large artifacts on disk with 
ID-based references in session state.

Features:
- SQLite for session metadata and small structured data
- File system for large artifacts (.jsonl, embeddings, processed content)
- Lazy loading with automatic cleanup
- Memory-efficient ID-based references
- Automatic session lifecycle management
"""

import sqlite3
import json
import pickle
import os
import shutil
import tempfile
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import streamlit as st
from modules.logger import get_logger

class LightweightPersistenceManager:
    """
    Lightweight persistence manager using SQLite + file system.
    
    Replaces memory-heavy session state with efficient disk-based storage
    and ID-based references for optimal Streamlit performance.
    """
    
    def __init__(self, base_dir: str = None):
        self.logger = get_logger("persistence_manager")
        
        # Setup storage directories
        if base_dir is None:
            base_dir = os.path.join(tempfile.gettempdir(), "ai_trainer_persistence")
        
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "sessions.db"
        self.artifacts_dir = self.base_dir / "artifacts"
        
        # Create directories
        self.base_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Session management
        self.session_id = self._get_or_create_session_id()
        self.logger.info(f"Initialized persistence manager for session: {self.session_id}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        artifact_type TEXT,
                        file_path TEXT,
                        size_bytes INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_artifacts 
                    ON artifacts (session_id)
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def _get_or_create_session_id(self) -> str:
        """Get existing session ID or create new one."""
        if 'persistence_session_id' not in st.session_state:
            # Create new session ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_hash = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
            session_id = f"session_{timestamp}_{random_hash}"
            
            # Store in session state
            st.session_state['persistence_session_id'] = session_id
            
            # Register in database
            self._register_session(session_id)
            
        return st.session_state['persistence_session_id']
    
    def _register_session(self, session_id: str):
        """Register new session in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, metadata)
                    VALUES (?, ?)
                """, (session_id, json.dumps({"created": datetime.now().isoformat()})))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Session registration failed: {str(e)}")
    
    def store_artifact(self, 
                      data: Any, 
                      artifact_type: str,
                      artifact_id: str = None,
                      metadata: Dict[str, Any] = None) -> str:
        """
        Store large artifact on disk and return ID for session state.
        
        Args:
            data: The data to store (any serializable object)
            artifact_type: Type of artifact (content, embeddings, processed_data, etc.)
            artifact_id: Optional custom ID, auto-generated if None
            metadata: Optional metadata dictionary
        
        Returns:
            Artifact ID for storing in session state
        """
        try:
            # Generate artifact ID if not provided
            if artifact_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                artifact_id = f"{self.session_id}_{artifact_type}_{timestamp}"
            
            # Determine storage format and file extension
            if artifact_type in ['embeddings', 'vectors', 'model_data']:
                # Use pickle for binary data
                file_path = self.artifacts_dir / f"{artifact_id}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            elif artifact_type in ['content', 'processed_data', 'jsonl']:
                # Use JSON for text data
                file_path = self.artifacts_dir / f"{artifact_id}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, (dict, list)):
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    else:
                        json.dump({"content": str(data)}, f, ensure_ascii=False, indent=2)
            else:
                # Default to pickle
                file_path = self.artifacts_dir / f"{artifact_id}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Store metadata in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO artifacts 
                    (artifact_id, session_id, artifact_type, file_path, size_bytes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    artifact_id,
                    self.session_id,
                    artifact_type,
                    str(file_path),
                    file_size,
                    json.dumps(metadata or {})
                ))
                conn.commit()
            
            self.logger.info(f"Stored artifact {artifact_id} ({file_size} bytes)")
            return artifact_id
            
        except Exception as e:
            self.logger.error(f"Failed to store artifact: {str(e)}")
            raise
    
    def load_artifact(self, artifact_id: str) -> Any:
        """
        Load artifact by ID with lazy loading.
        
        Args:
            artifact_id: The artifact ID to load
        
        Returns:
            The loaded data
        """
        try:
            # Get artifact info from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT artifact_type, file_path, metadata
                    FROM artifacts
                    WHERE artifact_id = ?
                """, (artifact_id,))
                
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"Artifact {artifact_id} not found")
                
                artifact_type, file_path, metadata_json = result
            
            # Load data based on type
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Artifact file not found: {file_path}")
            
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract content if wrapped
                    if isinstance(data, dict) and 'content' in data and len(data) == 1:
                        return data['content']
                    return data
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            self.logger.error(f"Failed to load artifact {artifact_id}: {str(e)}")
            raise
    
    def get_artifact_info(self, artifact_id: str) -> Dict[str, Any]:
        """Get artifact metadata without loading the data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT artifact_type, size_bytes, created_at, metadata
                    FROM artifacts
                    WHERE artifact_id = ?
                """, (artifact_id,))
                
                result = cursor.fetchone()
                if not result:
                    return {}
                
                artifact_type, size_bytes, created_at, metadata_json = result
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                return {
                    "artifact_type": artifact_type,
                    "size_bytes": size_bytes,
                    "created_at": created_at,
                    "metadata": metadata
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get artifact info: {str(e)}")
            return {}
    
    def list_session_artifacts(self) -> List[Dict[str, Any]]:
        """List all artifacts for current session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT artifact_id, artifact_type, size_bytes, created_at
                    FROM artifacts
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                """, (self.session_id,))
                
                return [
                    {
                        "artifact_id": row[0],
                        "artifact_type": row[1],
                        "size_bytes": row[2],
                        "created_at": row[3]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to list artifacts: {str(e)}")
            return []
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact and its file."""
        try:
            # Get file path
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_path FROM artifacts WHERE artifact_id = ?
                """, (artifact_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                file_path = Path(result[0])
                
                # Delete file
                if file_path.exists():
                    file_path.unlink()
                
                # Delete database record
                conn.execute("""
                    DELETE FROM artifacts WHERE artifact_id = ?
                """, (artifact_id,))
                conn.commit()
                
            self.logger.info(f"Deleted artifact {artifact_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete artifact {artifact_id}: {str(e)}")
            return False
    
    def cleanup_session(self, session_id: str = None) -> int:
        """Clean up all artifacts for a session."""
        if session_id is None:
            session_id = self.session_id
        
        try:
            deleted_count = 0
            
            with sqlite3.connect(self.db_path) as conn:
                # Get all artifacts for session
                cursor = conn.execute("""
                    SELECT artifact_id, file_path FROM artifacts WHERE session_id = ?
                """, (session_id,))
                
                artifacts = cursor.fetchall()
                
                # Delete files and records
                for artifact_id, file_path in artifacts:
                    file_path = Path(file_path)
                    if file_path.exists():
                        file_path.unlink()
                    deleted_count += 1
                
                # Delete database records
                conn.execute("""
                    DELETE FROM artifacts WHERE session_id = ?
                """, (session_id,))
                
                # Delete session record
                conn.execute("""
                    DELETE FROM sessions WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
            
            self.logger.info(f"Cleaned up {deleted_count} artifacts for session {session_id}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup session {session_id}: {str(e)}")
            return 0
    
    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """Clean up sessions older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_sessions = 0
            
            with sqlite3.connect(self.db_path) as conn:
                # Get old sessions
                cursor = conn.execute("""
                    SELECT session_id FROM sessions 
                    WHERE last_accessed < ?
                """, (cutoff_date.isoformat(),))
                
                old_sessions = [row[0] for row in cursor.fetchall()]
                
                # Clean up each old session
                for session_id in old_sessions:
                    self.cleanup_session(session_id)
                    deleted_sessions += 1
            
            self.logger.info(f"Cleaned up {deleted_sessions} old sessions")
            return deleted_sessions
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {str(e)}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total artifacts and size
                cursor = conn.execute("""
                    SELECT COUNT(*), SUM(size_bytes) FROM artifacts
                """)
                total_artifacts, total_size = cursor.fetchone()
                
                # Session count
                cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                total_sessions = cursor.fetchone()[0]
                
                # Current session stats
                cursor = conn.execute("""
                    SELECT COUNT(*), SUM(size_bytes) FROM artifacts WHERE session_id = ?
                """, (self.session_id,))
                session_artifacts, session_size = cursor.fetchone()
            
            return {
                "total_sessions": total_sessions,
                "total_artifacts": total_artifacts or 0,
                "total_size_bytes": total_size or 0,
                "current_session_artifacts": session_artifacts or 0,
                "current_session_size_bytes": session_size or 0,
                "storage_directory": str(self.base_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {str(e)}")
            return {}

# Global instance for easy access
_persistence_manager = None

def get_persistence_manager() -> LightweightPersistenceManager:
    """Get global persistence manager instance."""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = LightweightPersistenceManager()
    return _persistence_manager

# Convenience functions for common operations
def store_large_content(content: Any, content_type: str = "content") -> str:
    """Store large content and return ID for session state."""
    pm = get_persistence_manager()
    return pm.store_artifact(content, content_type)

def load_large_content(content_id: str) -> Any:
    """Load large content by ID."""
    pm = get_persistence_manager()
    return pm.load_artifact(content_id)

def cleanup_current_session():
    """Clean up current session artifacts."""
    pm = get_persistence_manager()
    return pm.cleanup_session()

# Session state helpers
def store_in_session_state(key: str, data: Any, artifact_type: str = None):
    """Store data efficiently in session state (large data goes to disk)."""
    if artifact_type is None:
        artifact_type = key
    
    # Estimate data size
    try:
        data_size = len(str(data)) if isinstance(data, str) else len(pickle.dumps(data))
    except:
        data_size = 1000  # Default assumption
    
    # Store large data on disk, small data in memory
    if data_size > 10000:  # 10KB threshold
        artifact_id = store_large_content(data, artifact_type)
        st.session_state[key] = {"type": "artifact_id", "id": artifact_id}
    else:
        st.session_state[key] = {"type": "direct", "data": data}

def load_from_session_state(key: str, default: Any = None) -> Any:
    """Load data efficiently from session state."""
    if key not in st.session_state:
        return default
    
    stored_data = st.session_state[key]
    
    if isinstance(stored_data, dict) and "type" in stored_data:
        if stored_data["type"] == "artifact_id":
            return load_large_content(stored_data["id"])
        elif stored_data["type"] == "direct":
            return stored_data["data"]
    
    # Fallback for legacy data
    return stored_data

