"""
Lean Session Manager Module
===========================

Optimizes session state management for low-memory environments like Render free tier.
Implements disk-based storage and intelligent memory management.

Features:
- Disk-based storage for large objects
- Automatic memory cleanup
- Session persistence and recovery
- Memory usage monitoring
- Configurable storage modes
"""

import os
import gc
import pickle
import json
import tempfile
import shutil
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st


class LeanSessionManager:
    """Memory-efficient session state manager"""
    
    def __init__(self, 
                 storage_dir: Optional[str] = None,
                 low_memory_mode: bool = None,
                 max_memory_items: int = 5):
        
        # Determine storage directory
        if storage_dir is None:
            storage_dir = tempfile.gettempdir()
        self.storage_dir = Path(storage_dir) / "fine_tune_sessions"
        self.storage_dir.mkdir(exist_ok=True)
        
        # Determine memory mode
        if low_memory_mode is None:
            low_memory_mode = os.getenv("LOW_MEM_MODE", "1") == "1"
        self.low_memory_mode = low_memory_mode
        
        self.max_memory_items = max_memory_items
        self.logger = logging.getLogger(__name__)
        
        # Session ID
        self.session_id = self._get_session_id()
        self.session_dir = self.storage_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # Memory tracking
        self.memory_keys = set()
        self.disk_keys = set()
        
        self.logger.info(f"Session manager initialized - Low memory mode: {self.low_memory_mode}")
    
    def _get_session_id(self) -> str:
        """Get or create session ID"""
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return st.session_state['session_id']
    
    def store_large_object(self, key: str, data: Any, force_disk: bool = False) -> bool:
        """Store large object with automatic memory/disk decision"""
        
        try:
            # Estimate object size
            size_estimate = self._estimate_object_size(data)
            use_disk = force_disk or self.low_memory_mode or size_estimate > 10_000_000  # 10MB threshold
            
            if use_disk:
                return self._store_to_disk(key, data)
            else:
                return self._store_to_memory(key, data)
                
        except Exception as e:
            self.logger.error(f"Failed to store object {key}: {e}")
            return False
    
    def retrieve_object(self, key: str, default: Any = None) -> Any:
        """Retrieve object from memory or disk"""
        
        try:
            # Check memory first
            if key in self.memory_keys and key in st.session_state:
                return st.session_state[key]
            
            # Check disk
            if key in self.disk_keys:
                return self._load_from_disk(key)
            
            # Check if it's a regular session state item
            if key in st.session_state:
                return st.session_state[key]
            
            return default
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve object {key}: {e}")
            return default
    
    def _store_to_memory(self, key: str, data: Any) -> bool:
        """Store object in session state memory"""
        
        try:
            # Check memory limit
            if len(self.memory_keys) >= self.max_memory_items:
                self._cleanup_oldest_memory_item()
            
            st.session_state[key] = data
            self.memory_keys.add(key)
            
            # Remove from disk keys if it was there
            self.disk_keys.discard(key)
            
            self.logger.debug(f"Stored {key} in memory")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store {key} in memory: {e}")
            return False
    
    def _store_to_disk(self, key: str, data: Any) -> bool:
        """Store object to disk"""
        
        try:
            file_path = self.session_dir / f"{key}.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.disk_keys.add(key)
            
            # Remove from memory if it was there
            if key in st.session_state:
                del st.session_state[key]
            self.memory_keys.discard(key)
            
            # Store metadata
            self._store_metadata(key, data)
            
            self.logger.debug(f"Stored {key} to disk")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store {key} to disk: {e}")
            return False
    
    def _load_from_disk(self, key: str) -> Any:
        """Load object from disk"""
        
        try:
            file_path = self.session_dir / f"{key}.pkl"
            
            if not file_path.exists():
                self.logger.warning(f"Disk file for {key} not found")
                return None
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.debug(f"Loaded {key} from disk")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load {key} from disk: {e}")
            return None
    
    def _store_metadata(self, key: str, data: Any):
        """Store metadata about the object"""
        
        try:
            metadata = {
                "key": key,
                "timestamp": datetime.now().isoformat(),
                "size_estimate": self._estimate_object_size(data),
                "type": type(data).__name__
            }
            
            metadata_path = self.session_dir / f"{key}_meta.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            self.logger.error(f"Failed to store metadata for {key}: {e}")
    
    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        
        try:
            # Try pickle size first
            return len(pickle.dumps(obj))
        except:
            # Fallback estimates
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_object_size(item) for item in obj[:10]) * len(obj) // 10
            elif isinstance(obj, dict):
                sample_items = list(obj.items())[:10]
                sample_size = sum(self._estimate_object_size(k) + self._estimate_object_size(v) 
                                for k, v in sample_items)
                return sample_size * len(obj) // len(sample_items) if sample_items else 0
            else:
                return 1000  # Default estimate
    
    def _cleanup_oldest_memory_item(self):
        """Remove oldest item from memory"""
        
        if not self.memory_keys:
            return
        
        # For simplicity, remove the first item
        # In a more sophisticated implementation, we could track access times
        oldest_key = next(iter(self.memory_keys))
        
        if oldest_key in st.session_state:
            # Move to disk before removing from memory
            data = st.session_state[oldest_key]
            self._store_to_disk(oldest_key, data)
        
        self.memory_keys.discard(oldest_key)
    
    def clear_session_data(self, keep_essential: bool = True):
        """Clear session data to free memory"""
        
        essential_keys = {
            'session_id', 'current_step', 'step_statuses', 
            'selected_theme', 'debug_mode'
        } if keep_essential else set()
        
        # Clear memory items
        keys_to_remove = []
        for key in st.session_state.keys():
            if key not in essential_keys:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        self.memory_keys.clear()
        
        # Clear disk items
        try:
            if self.session_dir.exists():
                for file_path in self.session_dir.glob("*.pkl"):
                    file_path.unlink()
                for file_path in self.session_dir.glob("*_meta.json"):
                    file_path.unlink()
        except Exception as e:
            self.logger.error(f"Failed to clear disk data: {e}")
        
        self.disk_keys.clear()
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("Session data cleared")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        try:
            memory_count = len(self.memory_keys)
            disk_count = len(self.disk_keys)
            
            # Calculate disk usage
            disk_size = 0
            if self.session_dir.exists():
                for file_path in self.session_dir.glob("*.pkl"):
                    disk_size += file_path.stat().st_size
            
            return {
                "session_id": self.session_id,
                "low_memory_mode": self.low_memory_mode,
                "memory_items": memory_count,
                "disk_items": disk_count,
                "disk_size_mb": disk_size / (1024 * 1024),
                "storage_dir": str(self.storage_dir),
                "session_dir": str(self.session_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session directories"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for session_dir in self.storage_dir.iterdir():
                if session_dir.is_dir() and session_dir.name != self.session_id:
                    # Check if directory is old
                    dir_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
                    
                    if dir_time < cutoff_time:
                        shutil.rmtree(session_dir)
                        self.logger.info(f"Cleaned up old session: {session_dir.name}")
                        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")
    
    def save_session_snapshot(self) -> str:
        """Save current session state to a snapshot file"""
        
        try:
            snapshot_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "memory_keys": list(self.memory_keys),
                "disk_keys": list(self.disk_keys),
                "session_state": {}
            }
            
            # Save essential session state
            essential_keys = [
                'current_step', 'step_statuses', 'selected_theme', 
                'total_files_processed', 'total_items_processed'
            ]
            
            for key in essential_keys:
                if key in st.session_state:
                    snapshot_data["session_state"][key] = st.session_state[key]
            
            snapshot_path = self.session_dir / "session_snapshot.json"
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            
            return str(snapshot_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save session snapshot: {e}")
            return ""
    
    def restore_session_snapshot(self, snapshot_path: str) -> bool:
        """Restore session from snapshot"""
        
        try:
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            # Restore session state
            for key, value in snapshot_data.get("session_state", {}).items():
                st.session_state[key] = value
            
            # Restore key tracking
            self.memory_keys = set(snapshot_data.get("memory_keys", []))
            self.disk_keys = set(snapshot_data.get("disk_keys", []))
            
            self.logger.info(f"Restored session from snapshot: {snapshot_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore session snapshot: {e}")
            return False


# Global session manager instance
session_manager = LeanSessionManager()


def store_large_data(key: str, data: Any, force_disk: bool = False) -> bool:
    """Convenience function to store large data"""
    return session_manager.store_large_object(key, data, force_disk)


def retrieve_data(key: str, default: Any = None) -> Any:
    """Convenience function to retrieve data"""
    return session_manager.retrieve_object(key, default)


def clear_session(keep_essential: bool = True):
    """Convenience function to clear session"""
    session_manager.clear_session_data(keep_essential)


def get_memory_stats() -> Dict[str, Any]:
    """Get memory and storage statistics"""
    return session_manager.get_storage_stats()


def render_memory_controls():
    """Render memory management controls in sidebar"""
    
    with st.sidebar.expander("💾 Memory Management", expanded=False):
        stats = get_memory_stats()
        
        st.write(f"**Mode:** {'Low Memory' if stats.get('low_memory_mode') else 'Standard'}")
        st.write(f"**Memory Items:** {stats.get('memory_items', 0)}")
        st.write(f"**Disk Items:** {stats.get('disk_items', 0)}")
        st.write(f"**Disk Usage:** {stats.get('disk_size_mb', 0):.1f} MB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Session"):
                clear_session(keep_essential=True)
                st.success("Session cleared!")
                st.rerun()
        
        with col2:
            if st.button("💾 Save Snapshot"):
                snapshot_path = session_manager.save_session_snapshot()
                if snapshot_path:
                    st.success("Snapshot saved!")
                else:
                    st.error("Snapshot failed!")


def auto_cleanup_memory():
    """Automatic memory cleanup based on usage"""
    
    try:
        import psutil
        
        # Get current memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1e6
        
        # If memory usage is high, trigger cleanup
        if memory_mb > 400:  # 400MB threshold for free tier
            session_manager.logger.warning(f"High memory usage: {memory_mb:.1f}MB - triggering cleanup")
            
            # Clear non-essential items
            non_essential = []
            for key in st.session_state.keys():
                if key not in {'session_id', 'current_step', 'step_statuses', 'selected_theme'}:
                    non_essential.append(key)
            
            # Move largest items to disk
            for key in non_essential[:3]:  # Move up to 3 items
                if key in st.session_state:
                    data = st.session_state[key]
                    session_manager._store_to_disk(key, data)
            
            gc.collect()
            
    except Exception as e:
        logging.error(f"Auto cleanup failed: {e}")


# Auto-cleanup on module import
auto_cleanup_memory()

