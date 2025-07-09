"""
Enhanced Debugging Module
========================

Provides comprehensive debugging capabilities for the Enhanced Universal AI Training Data Creator.
Includes hidden debug mode, state inspection, performance monitoring, and troubleshooting tools.

Features:
- Hidden debug mode activated via os.getenv("DEBUG_MODE")
- Session state inspection and manipulation
- Performance monitoring and profiling
- Error tracking and analysis
- Debug logging with detailed traces
"""

import os
import time
import traceback
import streamlit as st
from datetime import datetime
from functools import wraps
from typing import Dict, Any, List, Optional
import json
import psutil
import sys
import inspect
import logging

logger = logging.getLogger(__name__)


class EnhancedDebugging:
    """Enhanced debugging system with comprehensive troubleshooting tools"""
    
    def __init__(self):
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.performance_logs = []
        self.error_logs = []
        self.state_snapshots = []
        
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.debug_mode or os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    def debug_trace(self, func):
        """Decorator to trace function calls in debug mode"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.is_debug_mode():
                start_time = time.time()
                func_name = func.__name__
                
                logger.debug(f"🔍 TRACE: Entering {func_name}")
                logger.debug(f"🔍 ARGS: {args[:2]}...")  # Log first 2 args only
                logger.debug(f"🔍 KWARGS: {list(kwargs.keys())}")
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    logger.debug(f"🔍 TRACE: Exiting {func_name} (took {execution_time:.3f}s)")
                    
                    # Log performance data
                    self.performance_logs.append({
                        'function': func_name,
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat(),
                        'success': True
                    })
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    logger.error(f"🔍 ERROR in {func_name}: {str(e)}")
                    logger.error(f"🔍 TRACEBACK: {traceback.format_exc()}")
                    
                    # Log error data
                    self.error_logs.append({
                        'function': func_name,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    
    def capture_state_snapshot(self, label: str = ""):
        """Capture current session state snapshot for debugging"""
        if self.is_debug_mode():
            snapshot = {
                'label': label,
                'timestamp': datetime.now().isoformat(),
                'session_state_keys': list(st.session_state.keys()),
                'session_state_summary': {},
                'memory_usage': self.get_memory_usage()
            }
            
            # Capture summary of session state (avoid large objects)
            for key in st.session_state.keys():
                try:
                    value = st.session_state[key]
                    if isinstance(value, (str, int, float, bool)):
                        snapshot['session_state_summary'][key] = value
                    elif isinstance(value, (list, dict)):
                        snapshot['session_state_summary'][key] = f"{type(value).__name__}(len={len(value)})"
                    else:
                        snapshot['session_state_summary'][key] = f"{type(value).__name__}"
                except Exception as e:
                    snapshot['session_state_summary'][key] = f"Error: {str(e)}"
            
            self.state_snapshots.append(snapshot)
            logger.debug(f"🔍 STATE SNAPSHOT: {label} - {len(st.session_state)} keys")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return {'error': str(e)}
    
    def log_enhancement_step(self, step: str, content_length: int, tone: str, **kwargs):
        """Log detailed enhancement step information"""
        if self.is_debug_mode():
            logger.debug(f"🎭 ENHANCEMENT STEP: {step}")
            logger.debug(f"🎭 CONTENT LENGTH: {content_length} chars")
            logger.debug(f"🎭 TONE: {tone}")
            
            for key, value in kwargs.items():
                logger.debug(f"🎭 {key.upper()}: {value}")
    
    def log_extraction_step(self, step: str, file_type: str, file_size: int, **kwargs):
        """Log detailed extraction step information"""
        if self.is_debug_mode():
            logger.debug(f"📄 EXTRACTION STEP: {step}")
            logger.debug(f"📄 FILE TYPE: {file_type}")
            logger.debug(f"📄 FILE SIZE: {file_size} bytes")
            
            for key, value in kwargs.items():
                logger.debug(f"📄 {key.upper()}: {value}")
    
    def log_detection_step(self, step: str, content_type: str, confidence: float, **kwargs):
        """Log detailed detection step information"""
        if self.is_debug_mode():
            logger.debug(f"🔍 DETECTION STEP: {step}")
            logger.debug(f"🔍 CONTENT TYPE: {content_type}")
            logger.debug(f"🔍 CONFIDENCE: {confidence:.3f}")
            
            for key, value in kwargs.items():
                logger.debug(f"🔍 {key.upper()}: {value}")
    
    def render_debug_panel(self):
        """Render comprehensive debug panel in Streamlit"""
        if self.is_debug_mode():
            st.markdown("---")
            st.markdown("## 🔧 **Debug Panel** (Hidden Mode Active)")
            
            # Debug mode indicator
            st.success("🔍 Debug mode is ACTIVE")
            
            # Create debug tabs
            debug_tab1, debug_tab2, debug_tab3, debug_tab4, debug_tab5 = st.tabs([
                "📊 Performance", "🐛 Errors", "💾 Session State", "🖥️ System Info", "📝 Logs"
            ])
            
            with debug_tab1:
                self._render_performance_debug()
            
            with debug_tab2:
                self._render_error_debug()
            
            with debug_tab3:
                self._render_session_state_debug()
            
            with debug_tab4:
                self._render_system_info_debug()
            
            with debug_tab5:
                self._render_logs_debug()
    
    def _render_performance_debug(self):
        """Render performance debugging information"""
        st.markdown("### 📊 Performance Monitoring")
        
        if self.performance_logs:
            # Performance summary
            total_functions = len(self.performance_logs)
            avg_time = sum(log['execution_time'] for log in self.performance_logs) / total_functions
            slowest = max(self.performance_logs, key=lambda x: x['execution_time'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Function Calls", total_functions)
            
            with col2:
                st.metric("Average Execution Time", f"{avg_time:.3f}s")
            
            with col3:
                st.metric("Slowest Function", f"{slowest['function']} ({slowest['execution_time']:.3f}s)")
            
            # Recent performance logs
            st.markdown("**Recent Function Calls:**")
            recent_logs = self.performance_logs[-10:]  # Last 10 calls
            
            for log in reversed(recent_logs):
                with st.expander(f"{log['function']} - {log['execution_time']:.3f}s"):
                    st.json(log)
        else:
            st.info("No performance data available yet.")
    
    def _render_error_debug(self):
        """Render error debugging information"""
        st.markdown("### 🐛 Error Tracking")
        
        if self.error_logs:
            st.error(f"Found {len(self.error_logs)} errors in this session")
            
            for i, error in enumerate(reversed(self.error_logs)):
                with st.expander(f"Error {len(self.error_logs) - i}: {error['function']} - {error['error'][:50]}..."):
                    st.markdown(f"**Function:** {error['function']}")
                    st.markdown(f"**Error:** {error['error']}")
                    st.markdown(f"**Time:** {error['timestamp']}")
                    st.markdown(f"**Execution Time:** {error['execution_time']:.3f}s")
                    st.markdown("**Traceback:**")
                    st.code(error['traceback'])
        else:
            st.success("No errors recorded in this session!")
    
    def _render_session_state_debug(self):
        """Render session state debugging information"""
        st.markdown("### 💾 Session State Inspector")
        
        # Current session state
        st.markdown("**Current Session State:**")
        st.markdown(f"Total keys: {len(st.session_state)}")
        
        # Session state editor
        if st.button("Capture State Snapshot"):
            self.capture_state_snapshot("Manual snapshot")
            st.success("State snapshot captured!")
        
        # Display session state keys and values
        for key in sorted(st.session_state.keys()):
            try:
                value = st.session_state[key]
                with st.expander(f"🔑 {key} ({type(value).__name__})"):
                    if isinstance(value, (str, int, float, bool)):
                        st.write(f"**Value:** {value}")
                    elif isinstance(value, (list, dict)):
                        st.write(f"**Type:** {type(value).__name__}")
                        st.write(f"**Length:** {len(value)}")
                        if len(value) < 10:  # Only show small collections
                            st.json(value)
                        else:
                            st.write("(Too large to display)")
                    else:
                        st.write(f"**Type:** {type(value).__name__}")
                        st.write(f"**String representation:** {str(value)[:200]}...")
            except Exception as e:
                st.error(f"Error inspecting {key}: {e}")
        
        # State snapshots
        if self.state_snapshots:
            st.markdown("**State Snapshots:**")
            for i, snapshot in enumerate(reversed(self.state_snapshots)):
                with st.expander(f"Snapshot {len(self.state_snapshots) - i}: {snapshot['label']} - {snapshot['timestamp']}"):
                    st.json(snapshot)
    
    def _render_system_info_debug(self):
        """Render system information for debugging"""
        st.markdown("### 🖥️ System Information")
        
        # Memory usage
        memory = self.get_memory_usage()
        if 'error' not in memory:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RSS Memory", f"{memory['rss_mb']:.1f} MB")
            
            with col2:
                st.metric("Virtual Memory", f"{memory['vms_mb']:.1f} MB")
            
            with col3:
                st.metric("Memory %", f"{memory['percent']:.1f}%")
        
        # Python information
        st.markdown("**Python Environment:**")
        st.write(f"Python version: {sys.version}")
        st.write(f"Streamlit version: {st.__version__}")
        
        # Installed packages (key ones)
        st.markdown("**Key Dependencies:**")
        try:
            import openai
            st.write(f"OpenAI: {openai.__version__}")
        except:
            st.write("OpenAI: Not installed")
        
        try:
            import pandas as pd
            st.write(f"Pandas: {pd.__version__}")
        except:
            st.write("Pandas: Not installed")
    
    def _render_logs_debug(self):
        """Render recent logs for debugging"""
        st.markdown("### 📝 Recent Logs")
        
        # This would integrate with the logging system
        # For now, show a placeholder
        st.info("Log viewer would show recent application logs here")
        
        # Log level controls
        st.markdown("**Log Level Controls:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Set DEBUG Level"):
                logging.getLogger().setLevel(logging.DEBUG)
                st.success("Log level set to DEBUG")
        
        with col2:
            if st.button("Set INFO Level"):
                logging.getLogger().setLevel(logging.INFO)
                st.success("Log level set to INFO")
        
        with col3:
            if st.button("Set WARNING Level"):
                logging.getLogger().setLevel(logging.WARNING)
                st.success("Log level set to WARNING")


# Global debugging instance
enhanced_debugging = EnhancedDebugging()

