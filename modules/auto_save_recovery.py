"""
Auto-Save & Recovery System
===========================

Comprehensive auto-save and recovery system to protect user progress.
Implements the user's exact patterns:
- Auto-save st.session_state to disk every X minutes
- Save on key actions (post enhancement, etc.)
- Offer "Resume Previous Session" option

Based on the user's exact pattern:
with open("session_cache.pkl", "wb") as f:
    pickle.dump(st.session_state, f)
"""

import os
import pickle
import json
import time
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import streamlit as st

class AutoSaveManager:
    """
    Automatic session persistence with comprehensive recovery
    
    Implements the user's exact pattern:
    - Auto-save st.session_state to disk every X minutes
    - Save on key actions (post enhancement, etc.)
    - Offer "Resume Previous Session" option
    """
    
    def __init__(self, save_interval_minutes: int = 2):
        self.save_interval = save_interval_minutes * 60  # Convert to seconds
        self.session_file = "session_cache.pkl"
        self.backup_dir = "session_backups"
        self.max_backups = 10
        
        # Setup logging
        self.setup_logging()
        
        # Initialize auto-save system
        self.setup_backup_directory()
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.cleanup_old_backups)
    
    def setup_logging(self):
        """Setup auto-save specific logging"""
        self.logger = logging.getLogger('auto_save')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def setup_backup_directory(self):
        """Setup backup directory structure"""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            
            # Create subdirectories for organization
            os.makedirs(os.path.join(self.backup_dir, "auto"), exist_ok=True)
            os.makedirs(os.path.join(self.backup_dir, "manual"), exist_ok=True)
            os.makedirs(os.path.join(self.backup_dir, "emergency"), exist_ok=True)
            
        except Exception as e:
            self.logger.error(f"Failed to setup backup directory: {str(e)}")
    
    def auto_save_session(self) -> bool:
        """
        Auto-save session state - USER'S EXACT PATTERN!
        
        Implements user's exact approach:
        with open("session_cache.pkl", "wb") as f:
            pickle.dump(st.session_state, f)
        """
        try:
            # USER'S EXACT IMPLEMENTATION!
            with open(self.session_file, "wb") as f:
                pickle.dump(st.session_state, f)
            
            # Create timestamped backup
            self.create_timestamped_backup("auto")
            
            # Update last save time
            st.session_state['_last_auto_save'] = time.time()
            
            self.logger.info("Session auto-saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Auto-save failed: {str(e)}")
            return False
    
    def save_on_key_action(self, action_name: str) -> bool:
        """
        Save session on key actions - USER'S REQUIREMENT!
        
        Saves session after important workflow actions like:
        - Post enhancement
        - File upload
        - Manual review completion
        """
        try:
            # USER'S EXACT PATTERN FOR KEY ACTIONS!
            with open(self.session_file, "wb") as f:
                pickle.dump(st.session_state, f)
            
            # Log the action that triggered save
            st.session_state['_last_save_action'] = action_name
            st.session_state['_last_save_time'] = time.time()
            
            # Create manual backup for important actions
            self.create_timestamped_backup("manual")
            
            self.logger.info(f"Session saved after action: {action_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Action save failed ({action_name}): {str(e)}")
            return False
    
    def has_recoverable_session(self) -> bool:
        """Check if there's a recoverable session available"""
        try:
            if os.path.exists(self.session_file):
                # Check if file is not empty and readable
                if os.path.getsize(self.session_file) > 0:
                    with open(self.session_file, "rb") as f:
                        pickle.load(f)  # Test if file is valid
                    return True
            
        except Exception as e:
            self.logger.warning(f"Session file validation failed: {str(e)}")
        
        return False
    
    def get_session_info(self) -> Dict[str, str]:
        """Get information about the saved session"""
        session_info = {
            'last_save_time': 'Unknown',
            'progress_summary': 'No progress information',
            'data_summary': 'No data information'
        }
        
        try:
            if os.path.exists(self.session_file):
                # Get file modification time
                mod_time = os.path.getmtime(self.session_file)
                session_info['last_save_time'] = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                
                # Load session data to get progress info
                with open(self.session_file, "rb") as f:
                    session_data = pickle.load(f)
                
                # Extract progress information
                current_step = session_data.get('current_step', 'unknown')
                session_info['progress_summary'] = f"Current step: {current_step}"
                
                # Extract data information
                if 'uploaded_file' in session_data:
                    session_info['data_summary'] = "File uploaded"
                if 'content_enhanced' in session_data:
                    session_info['data_summary'] += ", Content enhanced"
                if 'manual_review_completed' in session_data:
                    session_info['data_summary'] += ", Review completed"
                
                if session_info['data_summary'] == 'No data information':
                    session_info['data_summary'] = "Session in progress"
        
        except Exception as e:
            self.logger.warning(f"Failed to get session info: {str(e)}")
        
        return session_info
    
    def offer_session_recovery(self) -> bool:
        """
        Offer session recovery option - USER'S EXACT REQUIREMENT!
        
        Shows "Resume Previous Session" option as requested by user
        """
        if self.has_recoverable_session():
            
            st.info("🔄 **Previous session detected!**")
            
            # Show session details
            session_info = self.get_session_info()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Last saved:** {session_info['last_save_time']}")
                st.write(f"**Progress:** {session_info['progress_summary']}")
                st.write(f"**Data:** {session_info['data_summary']}")
            
            with col2:
                # USER'S EXACT RECOVERY OPTION!
                if st.button("🔄 Resume Previous Session", type="primary"):
                    if self.restore_session():
                        st.success("✅ Session restored successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to restore session")
                
                if st.button("🗑️ Start Fresh"):
                    self.clear_saved_session()
                    st.success("✅ Starting fresh session")
                    st.rerun()
            
            return True
        
        return False
    
    def restore_session(self) -> bool:
        """Restore session from saved file"""
        try:
            if not os.path.exists(self.session_file):
                raise Exception("No session file to restore")
            
            # Create emergency backup of current session
            self.create_timestamped_backup("emergency")
            
            # Load saved session
            with open(self.session_file, "rb") as f:
                saved_session = pickle.load(f)
            
            # Validate session data
            if not isinstance(saved_session, dict):
                raise Exception("Invalid session data format")
            
            # Restore session state
            for key, value in saved_session.items():
                st.session_state[key] = value
            
            # Mark as restored session
            st.session_state['_restored_from_backup'] = True
            st.session_state['_restore_timestamp'] = time.time()
            
            self.logger.info("Session restored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Session restore failed: {str(e)}")
            return False
    
    def clear_saved_session(self):
        """Clear saved session file"""
        try:
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
                self.logger.info("Saved session cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear session: {str(e)}")
    
    def create_timestamped_backup(self, backup_type: str = "auto") -> Optional[str]:
        """Create timestamped backup of current session"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"session_backup_{timestamp}.pkl"
            backup_path = os.path.join(self.backup_dir, backup_type, backup_filename)
            
            # USER'S EXACT BACKUP PATTERN!
            with open(backup_path, "wb") as f:
                pickle.dump(st.session_state, f)
            
            # Add metadata
            metadata = {
                'timestamp': timestamp,
                'backup_type': backup_type,
                'session_size': len(pickle.dumps(st.session_state)),
                'progress_stage': st.session_state.get('current_step', 'unknown'),
                'creation_time': datetime.now().isoformat()
            }
            
            metadata_path = backup_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {str(e)}")
            return None
    
    def get_available_backups(self) -> List[Dict[str, Any]]:
        """Get list of available backup sessions"""
        backups = []
        
        try:
            for backup_type in ['auto', 'manual', 'emergency']:
                backup_subdir = os.path.join(self.backup_dir, backup_type)
                
                if os.path.exists(backup_subdir):
                    for filename in os.listdir(backup_subdir):
                        if filename.endswith('.pkl'):
                            backup_path = os.path.join(backup_subdir, filename)
                            metadata_path = backup_path.replace('.pkl', '_metadata.json')
                            
                            backup_info = {
                                'path': backup_path,
                                'type': backup_type,
                                'filename': filename,
                                'timestamp': os.path.getctime(backup_path),
                                'size': os.path.getsize(backup_path)
                            }
                            
                            # Load metadata if available
                            if os.path.exists(metadata_path):
                                try:
                                    with open(metadata_path, 'r') as f:
                                        metadata = json.load(f)
                                        backup_info.update(metadata)
                                except:
                                    pass
                            
                            backups.append(backup_info)
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to get backup list: {str(e)}")
        
        return backups
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore session from specific backup"""
        try:
            # Validate backup file
            if not os.path.exists(backup_path):
                raise Exception(f"Backup file not found: {backup_path}")
            
            # Load backup data
            with open(backup_path, "rb") as f:
                backup_data = pickle.load(f)
            
            # Validate backup data
            if not isinstance(backup_data, dict):
                raise Exception("Invalid backup data format")
            
            # Create emergency backup of current session
            self.create_timestamped_backup("emergency")
            
            # Restore session state
            for key, value in backup_data.items():
                st.session_state[key] = value
            
            # Mark as restored session
            st.session_state['_restored_from_backup'] = True
            st.session_state['_restore_timestamp'] = time.time()
            st.session_state['_restore_source'] = backup_path
            
            self.logger.info(f"Session restored from: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Session restore failed: {str(e)}")
            return False
    
    def cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            for backup_type in ['auto', 'manual', 'emergency']:
                backup_subdir = os.path.join(self.backup_dir, backup_type)
                
                if os.path.exists(backup_subdir):
                    # Get all backup files
                    backup_files = []
                    for filename in os.listdir(backup_subdir):
                        if filename.endswith('.pkl'):
                            file_path = os.path.join(backup_subdir, filename)
                            backup_files.append((file_path, os.path.getctime(file_path)))
                    
                    # Sort by creation time (oldest first)
                    backup_files.sort(key=lambda x: x[1])
                    
                    # Remove excess backups
                    while len(backup_files) > self.max_backups:
                        old_backup = backup_files.pop(0)
                        try:
                            os.remove(old_backup[0])
                            # Remove associated metadata
                            metadata_path = old_backup[0].replace('.pkl', '_metadata.json')
                            if os.path.exists(metadata_path):
                                os.remove(metadata_path)
                            
                            self.logger.info(f"Cleaned up old backup: {old_backup[0]}")
                        except:
                            pass
        
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {str(e)}")

class CrashDetectionManager:
    """Crash detection and recovery management"""
    
    def __init__(self, auto_save_manager: AutoSaveManager):
        self.auto_save_manager = auto_save_manager
        self.logger = logging.getLogger('crash_detection')
    
    def setup_crash_detection(self):
        """Setup crash detection and recovery"""
        
        # Mark session as active
        st.session_state['_session_active'] = True
        st.session_state['_session_start_time'] = time.time()
        
        # Check for previous crash
        if self.detect_previous_crash():
            self.handle_crash_recovery()
    
    def detect_previous_crash(self) -> bool:
        """Detect if previous session ended unexpectedly"""
        try:
            if os.path.exists("session_cache.pkl"):
                with open("session_cache.pkl", "rb") as f:
                    previous_session = pickle.load(f)
                
                # Check if previous session was marked as active
                if previous_session.get('_session_active', False):
                    return True
            
        except Exception as e:
            self.logger.warning(f"Crash detection failed: {str(e)}")
        
        return False
    
    def handle_crash_recovery(self):
        """Handle recovery from detected crash"""
        st.warning("🚨 **Unexpected session termination detected!**")
        st.info("Your previous session may have ended unexpectedly. You can recover your progress below.")
        
        # Offer recovery options
        self.auto_save_manager.offer_session_recovery()
    
    def mark_session_end(self):
        """Mark session as properly ended"""
        try:
            st.session_state['_session_active'] = False
            self.auto_save_manager.auto_save_session()
        except Exception as e:
            self.logger.error(f"Failed to mark session end: {str(e)}")

class ProgressProtectionManager:
    """Comprehensive progress protection and recovery"""
    
    def __init__(self, save_interval_minutes: int = 2):
        self.auto_save_manager = AutoSaveManager(save_interval_minutes)
        self.crash_detection = CrashDetectionManager(self.auto_save_manager)
        
        # Setup periodic auto-save
        self.setup_periodic_auto_save()
        
        # Setup crash detection
        self.crash_detection.setup_crash_detection()
    
    def setup_periodic_auto_save(self):
        """Setup periodic auto-save using Streamlit's session state"""
        
        # Initialize auto-save tracking
        if '_last_auto_save' not in st.session_state:
            st.session_state['_last_auto_save'] = 0
        
        # Check if auto-save is due
        current_time = time.time()
        time_since_last_save = current_time - st.session_state['_last_auto_save']
        
        # Auto-save every X minutes (user's requirement)
        if time_since_last_save > self.auto_save_manager.save_interval:
            if st.session_state.get('_auto_save_enabled', True):
                self.auto_save_manager.auto_save_session()
    
    def save_on_key_actions(self, action_name: str):
        """Save session on key workflow actions"""
        
        # List of key actions that trigger auto-save
        key_actions = [
            'file_uploaded',
            'content_extracted', 
            'content_enhanced',
            'quality_analyzed',
            'manual_review_completed',
            'export_configured'
        ]
        
        if action_name in key_actions:
            self.auto_save_manager.save_on_key_action(action_name)
    
    def render_recovery_interface(self):
        """Render comprehensive recovery interface"""
        
        st.subheader("💾 Session Recovery & Backup")
        
        # Current session status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            last_save = st.session_state.get('_last_auto_save', 0)
            if last_save > 0:
                time_since_save = time.time() - last_save
                st.metric("Last Auto-Save", f"{int(time_since_save)}s ago")
            else:
                st.metric("Last Auto-Save", "Never")
        
        with col2:
            session_start = st.session_state.get('_session_start_time', time.time())
            session_duration = time.time() - session_start
            st.metric("Session Duration", f"{int(session_duration/60)}m")
        
        with col3:
            if st.button("💾 Manual Save"):
                if self.auto_save_manager.auto_save_session():
                    st.success("✅ Session saved manually")
                else:
                    st.error("❌ Manual save failed")
        
        # Available backups
        st.markdown("---")
        st.write("**Available Backups:**")
        
        backups = self.auto_save_manager.get_available_backups()
        
        if backups:
            for i, backup in enumerate(backups[:5]):  # Show last 5 backups
                backup_time = datetime.fromtimestamp(backup['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                
                with st.expander(f"Backup {i+1}: {backup_time}"):
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Type:** {backup['type'].title()}")
                        st.write(f"**Size:** {backup.get('size', 0):,} bytes")
                        st.write(f"**Stage:** {backup.get('progress_stage', 'Unknown')}")
                    
                    with col2:
                        if st.button(f"🔄 Restore", key=f"restore_{i}"):
                            if self.auto_save_manager.restore_from_backup(backup['path']):
                                st.success("✅ Backup restored!")
                                st.rerun()
                            else:
                                st.error("❌ Restore failed")
        else:
            st.info("No backups available yet. Backups are created automatically as you work.")
        
        # Recovery settings
        st.markdown("---")
        st.write("**Recovery Settings:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_save_enabled = st.checkbox(
                "Enable Auto-Save",
                value=st.session_state.get('_auto_save_enabled', True),
                help="Automatically save session every 2 minutes"
            )
            
            st.session_state['_auto_save_enabled'] = auto_save_enabled
        
        with col2:
            crash_detection_enabled = st.checkbox(
                "Enable Crash Detection",
                value=st.session_state.get('_crash_detection_enabled', True),
                help="Detect unexpected session termination and offer recovery"
            )
            
            st.session_state['_crash_detection_enabled'] = crash_detection_enabled
        
        # Emergency actions
        st.markdown("---")
        st.write("**Emergency Actions:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚨 Emergency Save"):
                backup_path = self.auto_save_manager.create_timestamped_backup("emergency")
                if backup_path:
                    st.success("✅ Emergency backup created")
                else:
                    st.error("❌ Emergency backup failed")
        
        with col2:
            if st.button("🗑️ Clear All Backups"):
                if st.session_state.get('_confirm_clear_backups', False):
                    try:
                        shutil.rmtree(self.auto_save_manager.backup_dir)
                        self.auto_save_manager.setup_backup_directory()
                        st.success("✅ All backups cleared")
                        st.session_state['_confirm_clear_backups'] = False
                    except Exception as e:
                        st.error(f"❌ Failed to clear backups: {str(e)}")
                else:
                    st.session_state['_confirm_clear_backups'] = True
                    st.warning("⚠️ Click again to confirm clearing all backups")
        
        with col3:
            if st.button("📥 Export Session"):
                try:
                    session_export = {
                        'session_state': dict(st.session_state),
                        'export_timestamp': datetime.now().isoformat(),
                        'export_version': '1.0'
                    }
                    
                    export_data = json.dumps(session_export, indent=2, default=str)
                    
                    st.download_button(
                        "📥 Download Session Export",
                        data=export_data,
                        file_name=f"session_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Session export failed: {str(e)}")

# Integration with main application
class EnhancedAppWithAutoSave:
    """Enhanced application with comprehensive auto-save and recovery"""
    
    def __init__(self, save_interval_minutes: int = 2):
        self.progress_protection = ProgressProtectionManager(save_interval_minutes)
        
        # Initialize session if needed
        self.initialize_session()
        
        # Setup auto-save hooks
        self.setup_auto_save_hooks()
    
    def initialize_session(self):
        """Initialize session with recovery check"""
        
        # Check for session recovery first
        if not hasattr(st.session_state, '_initialized'):
            
            # Offer session recovery if available
            if self.progress_protection.auto_save_manager.offer_session_recovery():
                return  # Wait for user decision
            
            # Initialize fresh session
            st.session_state['_initialized'] = True
            st.session_state['current_step'] = 'upload'
            st.session_state['workflow_progress'] = {}
    
    def setup_auto_save_hooks(self):
        """Setup auto-save hooks for key actions"""
        
        # Periodic auto-save
        self.progress_protection.setup_periodic_auto_save()
        
        # Mark session as properly initialized
        st.session_state['_session_active'] = True
    
    def trigger_auto_save(self, action_name: str):
        """Trigger auto-save for specific action"""
        self.progress_protection.save_on_key_actions(action_name)
    
    def render_main_interface(self):
        """Render main application interface with auto-save"""
        
        st.title("🧠 Enhanced Universal AI Training Data Creator")
        st.write("*With Auto-Save & Recovery Protection*")
        
        # Show recovery interface in sidebar
        with st.sidebar:
            self.progress_protection.render_recovery_interface()
        
        # Main application tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "✨ Enhance", "📋 Review", "📦 Export"])
        
        with tab1:
            self.render_upload_tab()
        
        with tab2:
            self.render_enhance_tab()
        
        with tab3:
            self.render_review_tab()
        
        with tab4:
            self.render_export_tab()
    
    def render_upload_tab(self):
        """Render upload tab with auto-save"""
        st.subheader("📁 File Upload")
        
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])
        
        if uploaded_file and uploaded_file != st.session_state.get('uploaded_file'):
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['current_step'] = 'extract'
            
            # Trigger auto-save on file upload
            self.trigger_auto_save('file_uploaded')
            
            st.success("✅ File uploaded and session saved!")
    
    def render_enhance_tab(self):
        """Render enhance tab with auto-save"""
        st.subheader("✨ Content Enhancement")
        
        if st.session_state.get('uploaded_file'):
            if st.button("🚀 Enhance Content"):
                # Simulate enhancement process
                with st.spinner("Enhancing content..."):
                    time.sleep(2)  # Simulate processing
                    
                    st.session_state['content_enhanced'] = True
                    st.session_state['current_step'] = 'review'
                    
                    # Trigger auto-save after enhancement
                    self.trigger_auto_save('content_enhanced')
                    
                    st.success("✅ Content enhanced and session saved!")
        else:
            st.info("Please upload a file first.")
    
    def render_review_tab(self):
        """Render review tab with auto-save"""
        st.subheader("📋 Manual Review")
        
        if st.session_state.get('content_enhanced'):
            
            if st.button("✅ Approve All"):
                st.session_state['manual_review_completed'] = True
                st.session_state['current_step'] = 'export'
                
                # Trigger auto-save after review
                self.trigger_auto_save('manual_review_completed')
                
                st.success("✅ Review completed and session saved!")
        else:
            st.info("Please enhance content first.")
    
    def render_export_tab(self):
        """Render export tab with auto-save"""
        st.subheader("📦 Export Dataset")
        
        if st.session_state.get('manual_review_completed'):
            
            export_format = st.selectbox("Export Format", ["JSON", "JSONL", "CSV"])
            
            if st.button("📥 Export"):
                st.session_state['export_configured'] = True
                st.session_state['export_format'] = export_format
                st.session_state['current_step'] = 'completed'
                
                # Trigger auto-save before export
                self.trigger_auto_save('export_configured')
                
                st.success("✅ Export configuration saved!")
        else:
            st.info("Please complete manual review first.")

# Example usage
def main_with_auto_save():
    """Main application with auto-save and recovery"""
    
    app = EnhancedAppWithAutoSave(save_interval_minutes=2)
    app.render_main_interface()

if __name__ == "__main__":
    main_with_auto_save()

