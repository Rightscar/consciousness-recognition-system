"""
Session State Manager Module
===========================

Comprehensive session state management for Streamlit applications to prevent data loss
and ensure persistence across reruns and navigation.

Features:
- Centralized session state initialization and management
- Safe access patterns with default values
- Progress tracking and state validation
- Component-specific state managers
- Error recovery and cleanup utilities
- Performance optimization through intelligent caching
"""

import streamlit as st
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
import pickle
import base64

@dataclass
class SessionProgress:
    """Track progress through the application workflow"""
    file_uploaded: bool = False
    content_extracted: bool = False
    content_enhanced: bool = False
    quality_analyzed: bool = False
    manual_review_done: bool = False
    export_ready: bool = False
    
    def get_completion_percentage(self) -> float:
        """Get overall completion percentage"""
        completed = sum([
            self.file_uploaded,
            self.content_extracted,
            self.content_enhanced,
            self.quality_analyzed,
            self.manual_review_done,
            self.export_ready
        ])
        return completed / 6.0
    
    def get_current_step(self) -> int:
        """Get current step number (1-6)"""
        if not self.file_uploaded:
            return 1
        elif not self.content_extracted:
            return 2
        elif not self.content_enhanced:
            return 3
        elif not self.quality_analyzed:
            return 4
        elif not self.manual_review_done:
            return 5
        else:
            return 6
    
    def get_next_action(self) -> str:
        """Get description of next required action"""
        if not self.file_uploaded:
            return "Upload a file to begin"
        elif not self.content_extracted:
            return "Extract content from uploaded file"
        elif not self.content_enhanced:
            return "Enhance content with AI"
        elif not self.quality_analyzed:
            return "Analyze content quality"
        elif not self.manual_review_done:
            return "Complete manual review"
        elif not self.export_ready:
            return "Prepare final dataset for export"
        else:
            return "Download your training dataset"

class SessionStateManager:
    """
    Centralized session state management for the entire application
    
    Implements the pattern suggested by the user:
    if 'variable_name' not in st.session_state:
        st.session_state['variable_name'] = default_value
    """
    
    def __init__(self):
        """Initialize session state manager and all required variables"""
        self.initialize_all_session_variables()
        self.validate_session_integrity()
    
    def initialize_all_session_variables(self):
        """Initialize all session state variables with proper defaults"""
        
        # File upload and processing state
        self.init_if_not_exists('uploaded_file', None)
        self.init_if_not_exists('current_file_id', None)
        self.init_if_not_exists('file_content', None)
        self.init_if_not_exists('file_metadata', {})
        self.init_if_not_exists('extraction_method', 'auto')
        
        # Content extraction state
        self.init_if_not_exists('extracted_content', [])
        self.init_if_not_exists('content_type', None)
        self.init_if_not_exists('content_statistics', {})
        self.init_if_not_exists('extraction_complete', False)
        
        # Content enhancement state
        self.init_if_not_exists('enhanced_content', [])
        self.init_if_not_exists('enhancement_settings', {
            'tone': 'universal_wisdom',
            'creativity_level': 0.7,
            'preserve_structure': True,
            'batch_size': 10
        })
        self.init_if_not_exists('enhancement_progress', {})
        self.init_if_not_exists('enhancement_complete', False)
        
        # Quality analysis state
        self.init_if_not_exists('quality_scores', {})
        self.init_if_not_exists('quality_thresholds', {
            'coherence_min': 0.75,
            'length_ratio_max': 1.8,
            'length_ratio_min': 0.6,
            'overall_quality_min': 0.7
        })
        self.init_if_not_exists('flagged_for_review', [])
        self.init_if_not_exists('quality_statistics', {})
        self.init_if_not_exists('quality_analysis_complete', False)
        
        # Manual review state
        self.init_if_not_exists('manual_review_items', [])
        self.init_if_not_exists('review_decisions', {})
        self.init_if_not_exists('edited_content', {})
        self.init_if_not_exists('review_notes', {})
        self.init_if_not_exists('manual_review_complete', False)
        
        # Export configuration state
        self.init_if_not_exists('export_format', 'jsonl')
        self.init_if_not_exists('export_settings', {
            'include_metadata': True,
            'include_quality_scores': True,
            'minimum_quality_threshold': 0.7,
            'include_source_attribution': True
        })
        self.init_if_not_exists('final_dataset', [])
        self.init_if_not_exists('export_ready', False)
        
        # UI state and preferences
        self.init_if_not_exists('current_tab', 'upload')
        self.init_if_not_exists('show_advanced_options', False)
        self.init_if_not_exists('theme_selection', 'default')
        self.init_if_not_exists('sidebar_collapsed', False)
        self.init_if_not_exists('comparison_view_enabled', True)
        
        # Progress tracking
        self.init_if_not_exists('session_progress', SessionProgress())
        self.init_if_not_exists('workflow_step', 1)
        self.init_if_not_exists('last_activity', datetime.now())
        
        # Analytics and history
        self.init_if_not_exists('processing_history', [])
        self.init_if_not_exists('quality_history', [])
        self.init_if_not_exists('session_analytics', {
            'session_start': datetime.now(),
            'total_files_processed': 0,
            'total_content_enhanced': 0,
            'average_quality_score': 0.0
        })
        
        # Error handling and logging
        self.init_if_not_exists('error_log', [])
        self.init_if_not_exists('warning_log', [])
        self.init_if_not_exists('debug_info', {})
        
        # Feature flags and configuration
        self.init_if_not_exists('features_enabled', {
            'smart_content_detection': True,
            'quality_threshold_analysis': True,
            'manual_review_system': True,
            'comparison_viewer': True,
            'advanced_export_options': True,
            'real_time_analytics': True,
            'debug_mode': False
        })
        
        # Cache management
        self.init_if_not_exists('cache_enabled', True)
        self.init_if_not_exists('cache_data', {})
        self.init_if_not_exists('cache_timestamps', {})
        
        # User preferences
        self.init_if_not_exists('user_preferences', {
            'auto_save_interval': 30,  # seconds
            'show_progress_details': True,
            'enable_notifications': True,
            'preferred_export_format': 'jsonl'
        })
    
    def init_if_not_exists(self, key: str, default_value: Any):
        """
        Initialize session state key if it doesn't exist
        
        This implements the exact pattern suggested by the user:
        if 'variable_name' not in st.session_state:
            st.session_state['variable_name'] = default_value
        """
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Safely get session state value with optional default"""
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set session state value and update last activity"""
        st.session_state[key] = value
        st.session_state['last_activity'] = datetime.now()
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple session state values atomically"""
        for key, value in updates.items():
            st.session_state[key] = value
        st.session_state['last_activity'] = datetime.now()
    
    def delete(self, key: str):
        """Safely delete session state key"""
        if key in st.session_state:
            del st.session_state[key]
    
    def clear_processing_data(self):
        """Clear processing-related data for new upload while preserving settings"""
        keys_to_clear = [
            'file_content', 'extracted_content', 'enhanced_content',
            'quality_scores', 'manual_review_items', 'final_dataset',
            'flagged_for_review', 'review_decisions', 'edited_content'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                if key.endswith('content') or key.endswith('items') or key.endswith('dataset'):
                    st.session_state[key] = []
                elif key.endswith('scores') or key.endswith('decisions') or key.endswith('content'):
                    st.session_state[key] = {}
                else:
                    st.session_state[key] = None
        
        # Reset completion flags
        self.update({
            'extraction_complete': False,
            'enhancement_complete': False,
            'quality_analysis_complete': False,
            'manual_review_complete': False,
            'export_ready': False,
            'session_progress': SessionProgress(),
            'workflow_step': 1
        })
        
        # Clear cache for new file
        self.clear_cache()
        
        logging.info("Cleared processing data for new file upload")
    
    def get_progress_summary(self) -> SessionProgress:
        """Get comprehensive progress summary"""
        progress = SessionProgress()
        
        # Check each step completion
        progress.file_uploaded = self.get('uploaded_file') is not None
        progress.content_extracted = len(self.get('extracted_content', [])) > 0
        progress.content_enhanced = len(self.get('enhanced_content', [])) > 0
        progress.quality_analyzed = len(self.get('quality_scores', {})) > 0
        progress.manual_review_done = self.get('manual_review_complete', False)
        progress.export_ready = self.get('export_ready', False)
        
        # Update stored progress
        self.set('session_progress', progress)
        
        return progress
    
    def validate_session_integrity(self) -> bool:
        """Validate session state integrity and fix common issues"""
        try:
            # Check for required keys
            required_keys = [
                'uploaded_file', 'extracted_content', 'enhanced_content',
                'quality_scores', 'enhancement_settings', 'quality_thresholds'
            ]
            
            missing_keys = []
            for key in required_keys:
                if key not in st.session_state:
                    missing_keys.append(key)
            
            if missing_keys:
                logging.warning(f"Session state missing keys: {missing_keys}")
                self.initialize_all_session_variables()
                return False
            
            # Validate data types
            list_keys = ['extracted_content', 'enhanced_content', 'manual_review_items', 'final_dataset']
            for key in list_keys:
                if not isinstance(st.session_state.get(key), list):
                    st.session_state[key] = []
            
            dict_keys = ['quality_scores', 'review_decisions', 'enhancement_settings', 'quality_thresholds']
            for key in dict_keys:
                if not isinstance(st.session_state.get(key), dict):
                    st.session_state[key] = {}
            
            # Validate data consistency
            extracted_count = len(self.get('extracted_content', []))
            enhanced_count = len(self.get('enhanced_content', []))
            
            if enhanced_count > extracted_count:
                logging.warning("Enhanced content count exceeds extracted content count")
                # Could implement auto-fix here
            
            return True
            
        except Exception as e:
            logging.error(f"Session state validation failed: {e}")
            self.add_error(f"Session validation error: {str(e)}")
            return False
    
    def add_error(self, error_message: str):
        """Add error to error log with timestamp"""
        error_entry = {
            'timestamp': datetime.now(),
            'message': error_message,
            'session_state_keys': list(st.session_state.keys())
        }
        
        error_log = self.get('error_log', [])
        error_log.append(error_entry)
        
        # Keep only recent errors (last 50)
        if len(error_log) > 50:
            error_log = error_log[-25:]
        
        self.set('error_log', error_log)
        logging.error(error_message)
    
    def add_warning(self, warning_message: str):
        """Add warning to warning log with timestamp"""
        warning_entry = {
            'timestamp': datetime.now(),
            'message': warning_message
        }
        
        warning_log = self.get('warning_log', [])
        warning_log.append(warning_entry)
        
        # Keep only recent warnings (last 30)
        if len(warning_log) > 30:
            warning_log = warning_log[-15:]
        
        self.set('warning_log', warning_log)
        logging.warning(warning_message)
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get comprehensive session analytics"""
        analytics = self.get('session_analytics', {})
        
        # Update real-time metrics
        current_time = datetime.now()
        session_start = analytics.get('session_start', current_time)
        session_duration = current_time - session_start
        
        progress = self.get_progress_summary()
        
        updated_analytics = {
            **analytics,
            'session_duration_minutes': session_duration.total_seconds() / 60,
            'current_step': progress.get_current_step(),
            'completion_percentage': progress.get_completion_percentage() * 100,
            'last_activity': self.get('last_activity', current_time),
            'total_errors': len(self.get('error_log', [])),
            'total_warnings': len(self.get('warning_log', [])),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
        
        self.set('session_analytics', updated_analytics)
        return updated_analytics
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for performance monitoring"""
        cache_data = self.get('cache_data', {})
        if not cache_data:
            return 0.0
        
        # Simple cache hit rate calculation
        # In a real implementation, you'd track hits vs misses
        return min(len(cache_data) / 10.0, 1.0)  # Placeholder calculation
    
    # Cache management methods
    def set_cache(self, key: str, value: Any, ttl_minutes: int = 60):
        """Set cache value with TTL"""
        cache_data = self.get('cache_data', {})
        cache_timestamps = self.get('cache_timestamps', {})
        
        cache_data[key] = value
        cache_timestamps[key] = datetime.now() + timedelta(minutes=ttl_minutes)
        
        self.update({
            'cache_data': cache_data,
            'cache_timestamps': cache_timestamps
        })
    
    def get_cache(self, key: str, default: Any = None) -> Any:
        """Get cache value if not expired"""
        if not self.get('cache_enabled', True):
            return default
        
        cache_data = self.get('cache_data', {})
        cache_timestamps = self.get('cache_timestamps', {})
        
        if key not in cache_data or key not in cache_timestamps:
            return default
        
        # Check if expired
        if datetime.now() > cache_timestamps[key]:
            # Remove expired entry
            del cache_data[key]
            del cache_timestamps[key]
            self.update({
                'cache_data': cache_data,
                'cache_timestamps': cache_timestamps
            })
            return default
        
        return cache_data[key]
    
    def clear_cache(self):
        """Clear all cached data"""
        self.update({
            'cache_data': {},
            'cache_timestamps': {}
        })
    
    def export_session_state(self) -> str:
        """Export session state for backup/debugging"""
        try:
            # Create exportable version (exclude non-serializable objects)
            exportable_state = {}
            
            for key, value in st.session_state.items():
                try:
                    # Test if value is JSON serializable
                    json.dumps(value, default=str)
                    exportable_state[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values
                    exportable_state[key] = f"<non-serializable: {type(value).__name__}>"
            
            return json.dumps(exportable_state, indent=2, default=str)
            
        except Exception as e:
            logging.error(f"Failed to export session state: {e}")
            return f"Export failed: {str(e)}"
    
    def import_session_state(self, exported_state: str) -> bool:
        """Import session state from backup"""
        try:
            imported_data = json.loads(exported_state)
            
            # Validate imported data
            if not isinstance(imported_data, dict):
                return False
            
            # Update session state with imported data
            for key, value in imported_data.items():
                if not key.startswith('<non-serializable'):
                    st.session_state[key] = value
            
            # Re-validate after import
            return self.validate_session_integrity()
            
        except Exception as e:
            logging.error(f"Failed to import session state: {e}")
            self.add_error(f"Session import failed: {str(e)}")
            return False

class FileUploadSessionManager:
    """Manages file upload state and prevents re-processing"""
    
    def __init__(self, session_manager: SessionStateManager):
        self.session = session_manager
    
    def handle_file_upload(self, file_uploader_key: str = "main_file_uploader"):
        """
        Handle file upload with session state persistence
        
        Returns the uploaded file and whether it's a new file
        """
        # File uploader with session state tracking
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'docx', 'md', 'json', 'jsonl'],
            key=file_uploader_key,
            help="Upload text files, PDFs, Word documents, or JSON/JSONL files"
        )
        
        if uploaded_file is not None:
            current_file_id = self._generate_file_id(uploaded_file)
            stored_file_id = self.session.get('current_file_id')
            
            if current_file_id != stored_file_id:
                # New file uploaded - clear previous data and update session
                self.session.clear_processing_data()
                
                file_metadata = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'type': uploaded_file.type,
                    'upload_time': datetime.now(),
                    'file_id': current_file_id
                }
                
                self.session.update({
                    'uploaded_file': uploaded_file,
                    'current_file_id': current_file_id,
                    'file_metadata': file_metadata
                })
                
                st.success(f"✅ New file uploaded: {uploaded_file.name}")
                return uploaded_file, True  # New file
            else:
                # Same file - just update the file object (Streamlit requirement)
                self.session.set('uploaded_file', uploaded_file)
                return uploaded_file, False  # Existing file
        
        # No file uploaded
        return None, False
    
    def _generate_file_id(self, uploaded_file) -> str:
        """Generate unique ID for uploaded file based on content hash"""
        if uploaded_file is None:
            return None
        
        # Create hash from file content for reliable identification
        file_content = uploaded_file.getvalue()
        content_hash = hashlib.md5(file_content).hexdigest()
        
        return f"{uploaded_file.name}_{uploaded_file.size}_{content_hash[:8]}"
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get comprehensive information about currently uploaded file"""
        file_metadata = self.session.get('file_metadata', {})
        
        if not file_metadata:
            return {}
        
        # Add computed information
        upload_time = file_metadata.get('upload_time')
        if upload_time:
            time_since_upload = datetime.now() - upload_time
            file_metadata['time_since_upload_minutes'] = time_since_upload.total_seconds() / 60
        
        return file_metadata
    
    def is_file_uploaded(self) -> bool:
        """Check if a file is currently uploaded"""
        return self.session.get('uploaded_file') is not None

class ContentProcessingSessionManager:
    """Manages content processing state and caching"""
    
    def __init__(self, session_manager: SessionStateManager):
        self.session = session_manager
    
    def process_content_with_session(self, force_reprocess: bool = False):
        """
        Process content with session state caching
        
        Args:
            force_reprocess: Force reprocessing even if cached content exists
            
        Returns:
            List of extracted content items
        """
        # Check if content already processed and cached
        if (not force_reprocess and 
            self.session.get('extraction_complete') and 
            self.session.get('extracted_content')):
            
            st.info("📋 Using previously processed content from session")
            return self.session.get('extracted_content')
        
        uploaded_file = self.session.get('uploaded_file')
        if uploaded_file is None:
            st.warning("⚠️ No file uploaded. Please upload a file first.")
            return []
        
        # Show processing indicator
        with st.spinner("🔄 Processing content..."):
            try:
                # Extract content using appropriate method
                extraction_method = self.session.get('extraction_method', 'auto')
                extracted_content = self._extract_content(uploaded_file, extraction_method)
                
                # Detect content type
                content_type = self._detect_content_type(extracted_content)
                
                # Calculate statistics
                content_stats = self._calculate_content_statistics(extracted_content)
                
                # Store in session state
                self.session.update({
                    'extracted_content': extracted_content,
                    'content_type': content_type,
                    'content_statistics': content_stats,
                    'extraction_complete': True
                })
                
                # Cache the result
                self.session.set_cache(
                    f"extracted_content_{self.session.get('current_file_id')}", 
                    extracted_content,
                    ttl_minutes=120
                )
                
                st.success(f"✅ Processed {len(extracted_content)} content items")
                return extracted_content
                
            except Exception as e:
                error_msg = f"Content processing failed: {str(e)}"
                self.session.add_error(error_msg)
                st.error(error_msg)
                return []
    
    def _extract_content(self, uploaded_file, method: str = 'auto') -> List[Dict[str, Any]]:
        """Extract content from uploaded file"""
        try:
            file_content = uploaded_file.getvalue()
            
            # Handle different file types
            if uploaded_file.type == 'text/plain' or uploaded_file.name.endswith('.txt'):
                content = file_content.decode('utf-8')
                return self._process_text_content(content, uploaded_file.name)
            
            elif uploaded_file.name.endswith('.json'):
                content = json.loads(file_content.decode('utf-8'))
                return self._process_json_content(content, uploaded_file.name)
            
            elif uploaded_file.name.endswith('.jsonl'):
                lines = file_content.decode('utf-8').strip().split('\n')
                content = [json.loads(line) for line in lines if line.strip()]
                return self._process_jsonl_content(content, uploaded_file.name)
            
            else:
                # Fallback to text processing
                content = file_content.decode('utf-8', errors='ignore')
                return self._process_text_content(content, uploaded_file.name)
                
        except Exception as e:
            raise Exception(f"Failed to extract content from {uploaded_file.name}: {str(e)}")
    
    def _process_text_content(self, content: str, filename: str) -> List[Dict[str, Any]]:
        """Process plain text content"""
        # Split into paragraphs or sections
        sections = content.split('\n\n')
        
        processed_content = []
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) > 20:  # Skip very short sections
                processed_content.append({
                    'id': i,
                    'text': section,
                    'source': filename,
                    'type': 'text_section',
                    'word_count': len(section.split()),
                    'char_count': len(section)
                })
        
        return processed_content
    
    def _process_json_content(self, content: Union[Dict, List], filename: str) -> List[Dict[str, Any]]:
        """Process JSON content"""
        if isinstance(content, list):
            return [self._normalize_json_item(item, i, filename) for i, item in enumerate(content)]
        else:
            return [self._normalize_json_item(content, 0, filename)]
    
    def _process_jsonl_content(self, content: List[Dict], filename: str) -> List[Dict[str, Any]]:
        """Process JSONL content"""
        return [self._normalize_json_item(item, i, filename) for i, item in enumerate(content)]
    
    def _normalize_json_item(self, item: Dict, index: int, filename: str) -> Dict[str, Any]:
        """Normalize JSON item to standard format"""
        # Try to extract question/answer or text content
        text_content = ""
        
        if 'question' in item and 'answer' in item:
            text_content = f"Q: {item['question']}\nA: {item['answer']}"
        elif 'text' in item:
            text_content = item['text']
        elif 'content' in item:
            text_content = item['content']
        else:
            # Use the entire item as text
            text_content = json.dumps(item, indent=2)
        
        return {
            'id': index,
            'text': text_content,
            'source': filename,
            'type': 'json_item',
            'original_data': item,
            'word_count': len(text_content.split()),
            'char_count': len(text_content)
        }
    
    def _detect_content_type(self, content: List[Dict[str, Any]]) -> str:
        """Detect the type of content (Q&A, dialogue, monologue, mixed)"""
        if not content:
            return "unknown"
        
        # Simple heuristic-based detection
        qa_indicators = 0
        dialogue_indicators = 0
        
        for item in content[:10]:  # Check first 10 items
            text = item.get('text', '')
            
            # Check for Q&A patterns
            if any(pattern in text.lower() for pattern in ['q:', 'question:', 'a:', 'answer:']):
                qa_indicators += 1
            
            # Check for dialogue patterns
            if any(pattern in text for pattern in [':', '"', "'"]) and len(text.split('\n')) > 1:
                dialogue_indicators += 1
        
        total_checked = min(len(content), 10)
        
        if qa_indicators / total_checked > 0.5:
            return "qa_pair"
        elif dialogue_indicators / total_checked > 0.5:
            return "dialogue"
        elif qa_indicators > 0 and dialogue_indicators > 0:
            return "mixed"
        else:
            return "monologue"
    
    def _calculate_content_statistics(self, content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about the extracted content"""
        if not content:
            return {}
        
        total_items = len(content)
        total_words = sum(item.get('word_count', 0) for item in content)
        total_chars = sum(item.get('char_count', 0) for item in content)
        
        avg_words = total_words / total_items if total_items > 0 else 0
        avg_chars = total_chars / total_items if total_items > 0 else 0
        
        return {
            'total_items': total_items,
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_item': round(avg_words, 1),
            'average_characters_per_item': round(avg_chars, 1),
            'content_types': list(set(item.get('type', 'unknown') for item in content))
        }

class ProgressTracker:
    """Tracks and displays progress through the application workflow"""
    
    def __init__(self, session_manager: SessionStateManager):
        self.session = session_manager
    
    def render_progress_indicator(self):
        """Render comprehensive progress indicator"""
        progress = self.session.get_progress_summary()
        
        st.subheader("📈 Workflow Progress")
        
        # Progress steps with status indicators
        steps = [
            ("📁", "Upload", progress.file_uploaded),
            ("🔄", "Extract", progress.content_extracted),
            ("✨", "Enhance", progress.content_enhanced),
            ("📊", "Analyze", progress.quality_analyzed),
            ("📋", "Review", progress.manual_review_done),
            ("📦", "Export", progress.export_ready)
        ]
        
        cols = st.columns(len(steps))
        
        for i, (icon, label, completed) in enumerate(steps):
            with cols[i]:
                if completed:
                    st.success(f"{icon} {label}")
                elif i == progress.get_current_step() - 1:
                    st.info(f"{icon} {label}")
                else:
                    st.write(f"{icon} {label}")
        
        # Overall progress bar
        completion_pct = progress.get_completion_percentage()
        st.progress(completion_pct, text=f"Overall Progress: {completion_pct:.0%}")
        
        # Next action
        next_action = progress.get_next_action()
        if completion_pct < 1.0:
            st.info(f"**Next:** {next_action}")
        else:
            st.success("🎉 **Workflow Complete!** Ready to download your dataset.")
    
    def render_detailed_progress(self):
        """Render detailed progress information"""
        with st.expander("📊 Detailed Progress Information"):
            
            analytics = self.session.get_session_analytics()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Session Information:**")
                st.write(f"- Duration: {analytics.get('session_duration_minutes', 0):.1f} minutes")
                st.write(f"- Current Step: {analytics.get('current_step', 1)}/6")
                st.write(f"- Completion: {analytics.get('completion_percentage', 0):.1f}%")
            
            with col2:
                st.write("**Content Statistics:**")
                content_stats = self.session.get('content_statistics', {})
                st.write(f"- Items Extracted: {content_stats.get('total_items', 0)}")
                st.write(f"- Items Enhanced: {len(self.session.get('enhanced_content', []))}")
                st.write(f"- Items for Review: {len(self.session.get('flagged_for_review', []))}")
            
            # Error and warning summary
            errors = len(self.session.get('error_log', []))
            warnings = len(self.session.get('warning_log', []))
            
            if errors > 0 or warnings > 0:
                st.write("**Issues:**")
                if errors > 0:
                    st.error(f"Errors: {errors}")
                if warnings > 0:
                    st.warning(f"Warnings: {warnings}")

# Example usage and integration
def main():
    """Example of how to use the session state management system"""
    
    # Initialize session state manager
    session_manager = SessionStateManager()
    
    # Initialize component managers
    file_manager = FileUploadSessionManager(session_manager)
    content_manager = ContentProcessingSessionManager(session_manager)
    progress_tracker = ProgressTracker(session_manager)
    
    st.title("🧠 Enhanced Universal AI Training Data Creator")
    st.write("*With Comprehensive Session State Management*")
    
    # Render progress indicator
    progress_tracker.render_progress_indicator()
    
    # Main application tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "🔄 Process", "📊 Analytics", "🔧 Debug"])
    
    with tab1:
        st.subheader("File Upload")
        
        # File upload with session management
        uploaded_file, is_new_file = file_manager.handle_file_upload()
        
        if uploaded_file:
            file_info = file_manager.get_file_info()
            
            if is_new_file:
                st.info("🆕 New file detected - previous processing data cleared")
            
            # Display file information
            with st.expander("📄 File Information"):
                st.json(file_info)
    
    with tab2:
        st.subheader("Content Processing")
        
        if file_manager.is_file_uploaded():
            
            # Processing options
            col1, col2 = st.columns(2)
            
            with col1:
                extraction_method = st.selectbox(
                    "Extraction Method:",
                    ["auto", "text", "json", "structured"],
                    index=0,
                    key="extraction_method_selector"
                )
                session_manager.set('extraction_method', extraction_method)
            
            with col2:
                force_reprocess = st.checkbox(
                    "Force Reprocessing",
                    help="Reprocess content even if cached version exists"
                )
            
            # Process content button
            if st.button("🔄 Process Content", type="primary"):
                extracted_content = content_manager.process_content_with_session(force_reprocess)
                
                if extracted_content:
                    st.success(f"✅ Successfully processed {len(extracted_content)} items")
                    
                    # Show content preview
                    with st.expander("👀 Content Preview"):
                        for i, item in enumerate(extracted_content[:3]):
                            st.write(f"**Item {i+1}:**")
                            st.text(item.get('text', '')[:200] + "...")
                            st.write("---")
            
            # Show content statistics if available
            content_stats = session_manager.get('content_statistics')
            if content_stats:
                st.subheader("📊 Content Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Items", content_stats.get('total_items', 0))
                
                with col2:
                    st.metric("Total Words", f"{content_stats.get('total_words', 0):,}")
                
                with col3:
                    st.metric("Avg Words/Item", content_stats.get('average_words_per_item', 0))
        
        else:
            st.info("👆 Please upload a file first")
    
    with tab3:
        st.subheader("Session Analytics")
        
        # Render detailed progress
        progress_tracker.render_detailed_progress()
        
        # Session analytics
        analytics = session_manager.get_session_analytics()
        
        st.write("**Session Analytics:**")
        st.json(analytics)
    
    with tab4:
        st.subheader("Debug Information")
        
        # Session state export/import
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Session State"):
                exported_state = session_manager.export_session_state()
                st.download_button(
                    label="Download Session State",
                    data=exported_state,
                    file_name=f"session_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_session = st.file_uploader(
                "Import Session State",
                type=['json'],
                key="session_import"
            )
            
            if uploaded_session:
                if st.button("📥 Import Session State"):
                    imported_data = uploaded_session.getvalue().decode('utf-8')
                    success = session_manager.import_session_state(imported_data)
                    
                    if success:
                        st.success("✅ Session state imported successfully")
                        st.rerun()
                    else:
                        st.error("❌ Failed to import session state")
        
        # Error and warning logs
        with st.expander("🐛 Error Log"):
            error_log = session_manager.get('error_log', [])
            if error_log:
                for error in error_log[-10:]:  # Show last 10 errors
                    st.error(f"{error['timestamp']}: {error['message']}")
            else:
                st.info("No errors logged")
        
        with st.expander("⚠️ Warning Log"):
            warning_log = session_manager.get('warning_log', [])
            if warning_log:
                for warning in warning_log[-10:]:  # Show last 10 warnings
                    st.warning(f"{warning['timestamp']}: {warning['message']}")
            else:
                st.info("No warnings logged")
        
        # Raw session state (for debugging)
        with st.expander("🔍 Raw Session State"):
            st.json(dict(st.session_state))

if __name__ == "__main__":
    main()

