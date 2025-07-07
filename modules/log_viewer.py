"""
Log Viewer Component
===================

Integrated log viewer for the Streamlit app to monitor logs in real-time.
Provides filtering, searching, and real-time monitoring capabilities.
"""

import streamlit as st
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.logger import CentralizedLogger, get_logger
import time
import re

class LogViewer:
    """
    Comprehensive log viewer for Streamlit integration
    
    Features:
    - Real-time log monitoring
    - Log filtering and searching
    - Error analysis and statistics
    - Performance monitoring
    - Event tracking
    - Export capabilities
    """
    
    def __init__(self):
        self.central_logger = CentralizedLogger()
        self.logger = get_logger("log_viewer")
        
        # Initialize session state for log viewer
        if 'log_viewer_auto_refresh' not in st.session_state:
            st.session_state['log_viewer_auto_refresh'] = False
        
        if 'log_viewer_filter_level' not in st.session_state:
            st.session_state['log_viewer_filter_level'] = 'ALL'
        
        if 'log_viewer_search_query' not in st.session_state:
            st.session_state['log_viewer_search_query'] = ''
    
    def render_log_viewer_interface(self):
        """Render the complete log viewer interface"""
        
        st.header("📊 System Logs & Monitoring")
        
        # Log viewer tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Live Logs", 
            "🔍 Search & Filter", 
            "📈 Analytics", 
            "⚠️ Errors", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_live_logs_tab()
        
        with tab2:
            self.render_search_filter_tab()
        
        with tab3:
            self.render_analytics_tab()
        
        with tab4:
            self.render_errors_tab()
        
        with tab5:
            self.render_settings_tab()
    
    def render_live_logs_tab(self):
        """Render live logs monitoring tab"""
        
        st.subheader("📋 Live Log Stream")
        
        # Controls
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            log_type = st.selectbox(
                "Log Type",
                ["main", "errors", "performance", "events"],
                help="Select which log file to monitor"
            )
        
        with col2:
            lines_to_show = st.selectbox(
                "Lines to Show",
                [50, 100, 200, 500],
                index=1,
                help="Number of recent log lines to display"
            )
        
        with col3:
            auto_refresh = st.checkbox(
                "Auto Refresh",
                value=st.session_state['log_viewer_auto_refresh'],
                help="Automatically refresh logs every 5 seconds"
            )
            st.session_state['log_viewer_auto_refresh'] = auto_refresh
        
        with col4:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        
        # Get recent logs
        recent_logs = self.central_logger.get_recent_logs(log_type, lines_to_show)
        
        if recent_logs:
            # Display logs with syntax highlighting
            st.markdown("**Recent Log Entries:**")
            
            # Create a container for logs
            log_container = st.container()
            
            with log_container:
                # Process and display logs
                for i, log_line in enumerate(reversed(recent_logs[-50:])):  # Show last 50 for performance
                    log_line = log_line.strip()
                    if log_line:
                        # Color code by log level
                        if "ERROR" in log_line:
                            st.markdown(f"🔴 `{log_line}`")
                        elif "WARNING" in log_line:
                            st.markdown(f"🟡 `{log_line}`")
                        elif "INFO" in log_line:
                            st.markdown(f"🔵 `{log_line}`")
                        elif "DEBUG" in log_line:
                            st.markdown(f"⚪ `{log_line}`")
                        else:
                            st.markdown(f"`{log_line}`")
        else:
            st.info(f"No logs found in {log_type}.log")
        
        # Log statistics
        st.markdown("---")
        self.render_log_statistics()
    
    def render_search_filter_tab(self):
        """Render search and filter tab"""
        
        st.subheader("🔍 Search & Filter Logs")
        
        # Search controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search Query",
                value=st.session_state['log_viewer_search_query'],
                placeholder="Enter search terms (e.g., 'ERROR', 'enhancement', 'user_action')",
                help="Search across all log content"
            )
            st.session_state['log_viewer_search_query'] = search_query
        
        with col2:
            if st.button("🔍 Search"):
                if search_query:
                    self.perform_log_search(search_query)
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_level_filter = st.selectbox(
                "Log Level",
                ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=0,
                help="Filter by log level"
            )
            st.session_state['log_viewer_filter_level'] = log_level_filter
        
        with col2:
            date_filter = st.date_input(
                "Date Filter",
                value=datetime.now().date(),
                help="Filter logs by date"
            )
        
        with col3:
            module_filter = st.selectbox(
                "Module Filter",
                ["ALL", "extractor", "enhancer", "validator", "export", "ui", "ai"],
                help="Filter by module"
            )
        
        # Perform search if query exists
        if search_query:
            self.perform_log_search(search_query)
    
    def perform_log_search(self, query: str):
        """Perform log search and display results"""
        
        st.markdown(f"**Search Results for: '{query}'**")
        
        # Search across all log types
        all_results = {}
        log_types = ["main", "errors", "performance", "events"]
        
        for log_type in log_types:
            results = self.central_logger.search_logs(query, log_type, max_results=20)
            if results:
                all_results[log_type] = results
        
        if all_results:
            # Display results by log type
            for log_type, results in all_results.items():
                with st.expander(f"📁 {log_type.title()} Logs ({len(results)} matches)"):
                    for result in results:
                        # Highlight search term
                        highlighted = result.replace(
                            query, 
                            f"**{query}**"
                        )
                        st.markdown(f"`{highlighted}`")
        else:
            st.info("No matching log entries found.")
    
    def render_analytics_tab(self):
        """Render analytics and statistics tab"""
        
        st.subheader("📈 Log Analytics & Statistics")
        
        # Get log statistics
        stats = self.central_logger.get_log_statistics()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Session ID", stats['session_id'][-8:])  # Show last 8 chars
        
        with col2:
            st.metric("Events Logged", stats['events_logged'])
        
        with col3:
            total_size = sum(
                file_info.get('size_mb', 0) 
                for file_info in stats['log_files'].values() 
                if isinstance(file_info, dict) and 'size_mb' in file_info
            )
            st.metric("Total Log Size", f"{total_size:.1f} MB")
        
        with col4:
            total_lines = sum(
                file_info.get('line_count', 0) 
                for file_info in stats['log_files'].values() 
                if isinstance(file_info, dict) and 'line_count' in file_info
            )
            st.metric("Total Log Lines", f"{total_lines:,}")
        
        # Log file breakdown
        st.markdown("---")
        st.markdown("**Log File Breakdown:**")
        
        if stats['log_files']:
            # Create DataFrame for visualization
            log_data = []
            for log_type, file_info in stats['log_files'].items():
                if isinstance(file_info, dict) and 'size_mb' in file_info:
                    log_data.append({
                        'Log Type': log_type.title(),
                        'Size (MB)': file_info['size_mb'],
                        'Lines': file_info['line_count'],
                        'Path': file_info['path']
                    })
            
            if log_data:
                df = pd.DataFrame(log_data)
                
                # Display table
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Size distribution pie chart
                    fig_size = px.pie(
                        df, 
                        values='Size (MB)', 
                        names='Log Type',
                        title="Log Size Distribution"
                    )
                    st.plotly_chart(fig_size, use_container_width=True)
                
                with col2:
                    # Lines distribution bar chart
                    fig_lines = px.bar(
                        df, 
                        x='Log Type', 
                        y='Lines',
                        title="Log Lines by Type"
                    )
                    st.plotly_chart(fig_lines, use_container_width=True)
        
        # Performance analytics
        self.render_performance_analytics()
    
    def render_performance_analytics(self):
        """Render performance analytics"""
        
        st.markdown("---")
        st.markdown("**Performance Analytics:**")
        
        # Try to parse performance logs
        try:
            performance_logs = self.central_logger.get_recent_logs("performance", 100)
            
            if performance_logs:
                performance_data = []
                
                for log_line in performance_logs:
                    try:
                        # Extract JSON from log line
                        if '{' in log_line and '}' in log_line:
                            json_start = log_line.find('{')
                            json_data = json.loads(log_line[json_start:])
                            performance_data.append(json_data)
                    except:
                        continue
                
                if performance_data:
                    df_perf = pd.DataFrame(performance_data)
                    
                    # Performance summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'duration_seconds' in df_perf.columns:
                            avg_duration = df_perf['duration_seconds'].mean()
                            st.metric("Avg Operation Time", f"{avg_duration:.2f}s")
                    
                    with col2:
                        if 'operation' in df_perf.columns:
                            total_ops = len(df_perf)
                            st.metric("Total Operations", total_ops)
                    
                    # Performance trends
                    if len(df_perf) > 1 and 'duration_seconds' in df_perf.columns:
                        fig_perf = px.line(
                            df_perf, 
                            y='duration_seconds',
                            title="Operation Performance Over Time",
                            labels={'duration_seconds': 'Duration (seconds)'}
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
            else:
                st.info("No performance data available yet.")
        
        except Exception as e:
            st.warning(f"Could not load performance analytics: {str(e)}")
    
    def render_errors_tab(self):
        """Render errors analysis tab"""
        
        st.subheader("⚠️ Error Analysis")
        
        # Get error logs
        error_logs = self.central_logger.get_recent_logs("errors", 50)
        
        if error_logs:
            # Parse error data
            error_data = []
            
            for log_line in error_logs:
                try:
                    if '{' in log_line and '}' in log_line:
                        json_start = log_line.find('{')
                        json_data = json.loads(log_line[json_start:])
                        error_data.append(json_data)
                except:
                    continue
            
            if error_data:
                # Error statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Errors", len(error_data))
                
                with col2:
                    error_types = [err.get('error_type', 'Unknown') for err in error_data]
                    unique_types = len(set(error_types))
                    st.metric("Error Types", unique_types)
                
                with col3:
                    recent_errors = [
                        err for err in error_data 
                        if 'timestamp' in err and 
                        datetime.fromisoformat(err['timestamp']) > datetime.now() - timedelta(hours=1)
                    ]
                    st.metric("Recent (1h)", len(recent_errors))
                
                # Error breakdown
                st.markdown("**Error Breakdown:**")
                
                df_errors = pd.DataFrame(error_data)
                
                if 'error_type' in df_errors.columns:
                    error_counts = df_errors['error_type'].value_counts()
                    
                    fig_errors = px.bar(
                        x=error_counts.index,
                        y=error_counts.values,
                        title="Error Types Distribution",
                        labels={'x': 'Error Type', 'y': 'Count'}
                    )
                    st.plotly_chart(fig_errors, use_container_width=True)
                
                # Recent errors details
                st.markdown("**Recent Errors:**")
                
                for i, error in enumerate(error_data[-5:]):  # Show last 5 errors
                    with st.expander(f"Error {i+1}: {error.get('error_type', 'Unknown')}"):
                        st.write(f"**Message:** {error.get('error_message', 'No message')}")
                        st.write(f"**Module:** {error.get('module', 'Unknown')}")
                        st.write(f"**Time:** {error.get('timestamp', 'Unknown')}")
                        
                        if 'context' in error:
                            st.write(f"**Context:** {error['context']}")
                        
                        if 'stack_trace' in error:
                            st.code(error['stack_trace'], language='python')
            else:
                st.info("No structured error data found.")
        else:
            st.success("🎉 No errors found in recent logs!")
    
    def render_settings_tab(self):
        """Render log viewer settings tab"""
        
        st.subheader("⚙️ Log Viewer Settings")
        
        # Log level settings
        st.markdown("**Log Level Configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            console_level = st.selectbox(
                "Console Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1,
                help="Minimum level for console output"
            )
        
        with col2:
            file_level = st.selectbox(
                "File Log Level", 
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=0,
                help="Minimum level for file output"
            )
        
        # Auto-refresh settings
        st.markdown("**Auto-Refresh Settings:**")
        
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=60,
            value=5,
            help="How often to refresh logs in auto-refresh mode"
        )
        
        # Log retention settings
        st.markdown("**Log Retention Settings:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_file_size = st.number_input(
                "Max File Size (MB)",
                min_value=1,
                max_value=100,
                value=10,
                help="Maximum size before log rotation"
            )
        
        with col2:
            backup_count = st.number_input(
                "Backup Files",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of backup files to keep"
            )
        
        # Export options
        st.markdown("---")
        st.markdown("**Export Options:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export All Logs"):
                self.export_logs("all")
        
        with col2:
            if st.button("📥 Export Errors Only"):
                self.export_logs("errors")
        
        with col3:
            if st.button("📥 Export Performance"):
                self.export_logs("performance")
        
        # Clear logs options
        st.markdown("---")
        st.markdown("**Maintenance:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Old Logs"):
                if st.session_state.get('confirm_clear_logs', False):
                    self.clear_old_logs()
                    st.success("✅ Old logs cleared")
                    st.session_state['confirm_clear_logs'] = False
                else:
                    st.session_state['confirm_clear_logs'] = True
                    st.warning("⚠️ Click again to confirm")
        
        with col2:
            if st.button("📊 Refresh Statistics"):
                st.rerun()
    
    def render_log_statistics(self):
        """Render quick log statistics"""
        
        st.markdown("**Quick Statistics:**")
        
        stats = self.central_logger.get_log_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Session Events", stats['events_logged'])
        
        with col2:
            main_log_info = stats['log_files'].get('main', {})
            if isinstance(main_log_info, dict):
                st.metric("Main Log Lines", main_log_info.get('line_count', 0))
        
        with col3:
            error_log_info = stats['log_files'].get('errors', {})
            if isinstance(error_log_info, dict):
                st.metric("Error Log Lines", error_log_info.get('line_count', 0))
        
        with col4:
            perf_log_info = stats['log_files'].get('performance', {})
            if isinstance(perf_log_info, dict):
                st.metric("Performance Entries", perf_log_info.get('line_count', 0))
    
    def export_logs(self, log_type: str):
        """Export logs to downloadable format"""
        
        try:
            if log_type == "all":
                # Export all logs
                all_logs = {}
                for lt in ["main", "errors", "performance", "events"]:
                    logs = self.central_logger.get_recent_logs(lt, 1000)
                    all_logs[lt] = logs
                
                export_data = json.dumps(all_logs, indent=2)
                filename = f"all_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            else:
                # Export specific log type
                logs = self.central_logger.get_recent_logs(log_type, 1000)
                export_data = "\n".join(logs)
                filename = f"{log_type}_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            st.download_button(
                f"📥 Download {log_type.title()} Logs",
                data=export_data,
                file_name=filename,
                mime="text/plain" if log_type != "all" else "application/json"
            )
        
        except Exception as e:
            st.error(f"Failed to export logs: {str(e)}")
    
    def clear_old_logs(self):
        """Clear old log files"""
        
        try:
            log_files = self.central_logger.get_log_files()
            
            for log_type, log_path in log_files.items():
                if os.path.exists(log_path):
                    # Keep only recent entries
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Keep last 100 lines
                    if len(lines) > 100:
                        with open(log_path, 'w', encoding='utf-8') as f:
                            f.writelines(lines[-100:])
            
            self.logger.info("Old logs cleared successfully")
        
        except Exception as e:
            st.error(f"Failed to clear logs: {str(e)}")
            self.logger.error(f"Log clearing failed: {str(e)}")

# Integration function for main app
def render_log_viewer():
    """
    Render log viewer in Streamlit app
    
    Usage in main app:
    from modules.log_viewer import render_log_viewer
    
    # In sidebar or main area
    render_log_viewer()
    """
    
    log_viewer = LogViewer()
    log_viewer.render_log_viewer_interface()

# Compact log viewer for sidebar
def render_compact_log_viewer():
    """
    Render compact log viewer for sidebar
    
    Usage:
    with st.sidebar:
        render_compact_log_viewer()
    """
    
    st.markdown("### 📊 System Status")
    
    central_logger = CentralizedLogger()
    stats = central_logger.get_log_statistics()
    
    # Quick metrics
    st.metric("Events", stats['events_logged'])
    
    # Recent errors
    error_logs = central_logger.get_recent_logs("errors", 5)
    if error_logs:
        st.warning(f"⚠️ {len(error_logs)} recent errors")
        if st.button("View Errors"):
            st.session_state['show_log_viewer'] = True
    else:
        st.success("✅ No recent errors")
    
    # Quick actions
    if st.button("📋 View Full Logs"):
        st.session_state['show_log_viewer'] = True

if __name__ == "__main__":
    # Test the log viewer
    st.set_page_config(page_title="Log Viewer Test", layout="wide")
    
    st.title("Log Viewer Test")
    
    # Generate some test logs
    from modules.logger import get_logger, log_event, log_performance, log_error
    
    logger = get_logger("test")
    logger.info("Test log entry")
    log_event("test_event", {"test": True}, "test")
    log_performance("test_operation", 1.5, {"items": 10}, "test")
    
    # Render log viewer
    render_log_viewer()

