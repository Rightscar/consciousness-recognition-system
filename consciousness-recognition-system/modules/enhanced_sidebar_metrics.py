"""
Enhanced Sidebar Metrics Module
===============================

Implements Core Enhancement 5: Sidebar Metrics Dashboard.
Provides comprehensive mini dashboard with key metrics and real-time updates.

Features:
- File upload information
- Processing statistics
- Quality metrics
- Export readiness indicators
- Cost tracking
- Performance monitoring
"""

import streamlit as st
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedSidebarMetrics:
    """Enhanced Sidebar Metrics Dashboard - Core Enhancement 5"""
    
    def __init__(self):
        self.metrics_data = {
            'uploaded_file': None,
            'file_size': 0,
            'total_examples': 0,
            'enhanced_examples': 0,
            'final_export_count': 0,
            'quality_distribution': {},
            'processing_time': 0,
            'enhancement_cost': 0.0,
            'session_start': time.time(),
            'last_update': None
        }
        
        # Initialize session state for metrics
        if 'sidebar_metrics' not in st.session_state:
            st.session_state.sidebar_metrics = self.metrics_data.copy()
    
    def render_sidebar(self, 
                      uploaded_file: Optional[Any] = None,
                      extracted_content: Optional[List[Dict[str, Any]]] = None,
                      enhanced_content: Optional[List[Dict[str, Any]]] = None,
                      export_content: Optional[List[Dict[str, Any]]] = None,
                      processing_stats: Optional[Dict[str, Any]] = None) -> None:
        """Render the complete enhanced sidebar with metrics - Core Enhancement 5 requirement"""
        
        # Update metrics data
        self._update_metrics(uploaded_file, extracted_content, enhanced_content, 
                           export_content, processing_stats)
        
        # Render sidebar sections
        with st.sidebar:
            self._render_header()
            self._render_file_info()
            self._render_processing_metrics()
            self._render_quality_metrics()
            self._render_export_metrics()
            self._render_performance_metrics()
            self._render_session_info()
    
    def _update_metrics(self, uploaded_file, extracted_content, enhanced_content, 
                       export_content, processing_stats):
        """Update internal metrics data"""
        
        # File information
        if uploaded_file:
            self.metrics_data['uploaded_file'] = uploaded_file.name
            self.metrics_data['file_size'] = uploaded_file.size if hasattr(uploaded_file, 'size') else 0
        
        # Content counts
        self.metrics_data['total_examples'] = len(extracted_content) if extracted_content else 0
        self.metrics_data['enhanced_examples'] = len([item for item in (enhanced_content or []) 
                                                     if item.get('enhanced', False)])
        self.metrics_data['final_export_count'] = len(export_content) if export_content else 0
        
        # Quality distribution
        if enhanced_content:
            quality_scores = [item.get('quality_score', 0) for item in enhanced_content 
                            if item.get('enhanced', False)]
            self.metrics_data['quality_distribution'] = self._calculate_quality_distribution(quality_scores)
        
        # Enhancement cost
        if enhanced_content:
            total_cost = sum(item.get('enhancement_cost', 0) for item in enhanced_content 
                           if item.get('enhanced', False))
            self.metrics_data['enhancement_cost'] = total_cost
        
        # Processing stats
        if processing_stats:
            self.metrics_data['processing_time'] = processing_stats.get('processing_time', 0)
        
        # Update timestamp
        self.metrics_data['last_update'] = datetime.now()
        
        # Update session state
        st.session_state.sidebar_metrics = self.metrics_data.copy()
    
    def _render_header(self):
        """Render sidebar header - Core Enhancement 5 requirement"""
        
        st.markdown("## 📊 **Dashboard**")
        st.markdown("---")
        
        # Status indicator
        if self.metrics_data['uploaded_file']:
            if self.metrics_data['final_export_count'] > 0:
                st.success("✅ Ready for Export")
            elif self.metrics_data['enhanced_examples'] > 0:
                st.info("🔄 Processing Complete")
            elif self.metrics_data['total_examples'] > 0:
                st.warning("⏳ Content Extracted")
            else:
                st.info("📁 File Uploaded")
        else:
            st.info("🚀 Ready to Start")
    
    def _render_file_info(self):
        """Render file information section - Core Enhancement 5 requirement"""
        
        st.markdown("### 📁 **File Info**")
        
        if self.metrics_data['uploaded_file']:
            # File name (truncated if too long)
            file_name = self.metrics_data['uploaded_file']
            if len(file_name) > 25:
                display_name = file_name[:22] + "..."
            else:
                display_name = file_name
            
            st.metric("📄 File", display_name)
            
            # File size
            file_size = self.metrics_data['file_size']
            if file_size > 1024 * 1024:  # MB
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            elif file_size > 1024:  # KB
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} B"
            
            st.metric("💾 Size", size_str)
        else:
            st.info("No file uploaded")
        
        st.markdown("---")
    
    def _render_processing_metrics(self):
        """Render processing metrics section - Core Enhancement 5 requirement"""
        
        st.markdown("### ⚙️ **Processing**")
        
        # Total examples
        st.metric(
            "📝 Total Examples", 
            self.metrics_data['total_examples'],
            help="Total examples extracted from uploaded content"
        )
        
        # Enhanced examples
        enhanced_count = self.metrics_data['enhanced_examples']
        total_count = self.metrics_data['total_examples']
        
        if total_count > 0:
            enhancement_rate = (enhanced_count / total_count) * 100
            delta_str = f"{enhancement_rate:.0f}% enhanced"
        else:
            delta_str = None
        
        st.metric(
            "✨ Enhanced", 
            enhanced_count,
            delta=delta_str,
            help="Examples processed through GPT enhancement"
        )
        
        # Processing time
        if self.metrics_data['processing_time'] > 0:
            time_str = f"{self.metrics_data['processing_time']:.1f}s"
            st.metric("⏱️ Process Time", time_str)
        
        st.markdown("---")
    
    def _render_quality_metrics(self):
        """Render quality metrics section - Core Enhancement 5 requirement"""
        
        st.markdown("### 🎯 **Quality**")
        
        quality_dist = self.metrics_data['quality_distribution']
        
        if quality_dist:
            # Quality breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                excellent = quality_dist.get('excellent', 0)
                good = quality_dist.get('good', 0)
                st.metric("🟢 High", excellent + good)
            
            with col2:
                fair = quality_dist.get('fair', 0)
                poor = quality_dist.get('poor', 0)
                st.metric("🟡 Low", fair + poor)
            
            # Average quality
            avg_quality = quality_dist.get('average', 0)
            if avg_quality > 0:
                st.metric("📊 Avg Quality", f"{avg_quality:.2f}")
        else:
            st.info("No quality data yet")
        
        st.markdown("---")
    
    def _render_export_metrics(self):
        """Render export metrics section - Core Enhancement 5 requirement"""
        
        st.markdown("### 📤 **Export**")
        
        # Final export count
        export_count = self.metrics_data['final_export_count']
        total_count = self.metrics_data['total_examples']
        
        if total_count > 0:
            export_rate = (export_count / total_count) * 100
            delta_str = f"{export_rate:.0f}% of total"
        else:
            delta_str = None
        
        st.metric(
            "📋 Ready to Export", 
            export_count,
            delta=delta_str,
            help="Examples ready for final export"
        )
        
        # Enhancement cost
        cost = self.metrics_data['enhancement_cost']
        if cost > 0:
            st.metric("💰 Enhancement Cost", f"${cost:.4f}")
        
        st.markdown("---")
    
    def _render_performance_metrics(self):
        """Render performance metrics section"""
        
        st.markdown("### ⚡ **Performance**")
        
        # Examples per second (if processing time available)
        if (self.metrics_data['processing_time'] > 0 and 
            self.metrics_data['enhanced_examples'] > 0):
            
            eps = self.metrics_data['enhanced_examples'] / self.metrics_data['processing_time']
            st.metric("🚀 Examples/sec", f"{eps:.1f}")
        
        # Cost per example
        if (self.metrics_data['enhancement_cost'] > 0 and 
            self.metrics_data['enhanced_examples'] > 0):
            
            cost_per_example = self.metrics_data['enhancement_cost'] / self.metrics_data['enhanced_examples']
            st.metric("💸 Cost/Example", f"${cost_per_example:.4f}")
        
        st.markdown("---")
    
    def _render_session_info(self):
        """Render session information"""
        
        st.markdown("### 🕐 **Session**")
        
        # Session duration
        session_duration = time.time() - self.metrics_data['session_start']
        if session_duration > 3600:  # Hours
            duration_str = f"{session_duration / 3600:.1f}h"
        elif session_duration > 60:  # Minutes
            duration_str = f"{session_duration / 60:.1f}m"
        else:  # Seconds
            duration_str = f"{session_duration:.0f}s"
        
        st.metric("⏰ Duration", duration_str)
        
        # Last update
        if self.metrics_data['last_update']:
            update_time = self.metrics_data['last_update'].strftime("%H:%M:%S")
            st.caption(f"Last update: {update_time}")
    
    def _calculate_quality_distribution(self, quality_scores: List[float]) -> Dict[str, Any]:
        """Calculate quality score distribution"""
        
        if not quality_scores:
            return {}
        
        excellent = sum(1 for score in quality_scores if score >= 0.8)
        good = sum(1 for score in quality_scores if 0.6 <= score < 0.8)
        fair = sum(1 for score in quality_scores if 0.4 <= score < 0.6)
        poor = sum(1 for score in quality_scores if score < 0.4)
        
        average = sum(quality_scores) / len(quality_scores)
        
        return {
            'excellent': excellent,
            'good': good,
            'fair': fair,
            'poor': poor,
            'average': average,
            'total': len(quality_scores)
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary for external use"""
        
        return {
            'file_uploaded': bool(self.metrics_data['uploaded_file']),
            'total_examples': self.metrics_data['total_examples'],
            'enhanced_examples': self.metrics_data['enhanced_examples'],
            'export_ready': self.metrics_data['final_export_count'],
            'enhancement_rate': (
                (self.metrics_data['enhanced_examples'] / max(self.metrics_data['total_examples'], 1)) * 100
                if self.metrics_data['total_examples'] > 0 else 0
            ),
            'export_rate': (
                (self.metrics_data['final_export_count'] / max(self.metrics_data['total_examples'], 1)) * 100
                if self.metrics_data['total_examples'] > 0 else 0
            ),
            'total_cost': self.metrics_data['enhancement_cost'],
            'session_duration': time.time() - self.metrics_data['session_start']
        }
    
    def render_compact_metrics(self):
        """Render compact metrics for main content area"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Examples", self.metrics_data['total_examples'])
        
        with col2:
            st.metric("✨ Enhanced", self.metrics_data['enhanced_examples'])
        
        with col3:
            st.metric("📤 Export Ready", self.metrics_data['final_export_count'])
        
        with col4:
            cost = self.metrics_data['enhancement_cost']
            st.metric("💰 Cost", f"${cost:.4f}" if cost > 0 else "$0.00")

