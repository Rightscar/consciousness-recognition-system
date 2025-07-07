"""
Enhanced Comparison Viewer Module
=================================

Implements Core Enhancement 4: Raw vs Enhanced Comparison Viewer.
Provides comprehensive side-by-side comparison of original and enhanced content.

Features:
- Toggle-based comparison mode
- Side-by-side raw vs enhanced display
- Quality improvement metrics
- Filtering and sorting options
- Performance optimized for large datasets
"""

import streamlit as st
import logging
from typing import List, Dict, Any, Optional, Tuple
import difflib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedComparisonViewer:
    """Enhanced Raw vs Enhanced Comparison Viewer - Core Enhancement 4"""
    
    def __init__(self):
        self.max_display_items = 20  # Increased limit for better UX
        self.comparison_enabled = False
        self.show_metrics = True
        self.show_diff_highlights = True
        
    def render_comparison_toggle(self) -> bool:
        """Render the comparison toggle with enhanced options - Core Enhancement 4 requirement"""
        
        st.markdown("### 🔍 **Raw vs Enhanced Comparison**")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("Compare original content with GPT-enhanced versions to analyze improvements:")
        
        with col2:
            self.comparison_enabled = st.toggle(
                "🔄 Compare Raw vs Enhanced",
                value=self.comparison_enabled,
                help="Toggle to see side-by-side comparison of original and enhanced content",
                key="comparison_toggle"
            )
        
        with col3:
            if self.comparison_enabled:
                self.show_metrics = st.checkbox(
                    "📊 Show Metrics",
                    value=self.show_metrics,
                    help="Display improvement metrics and statistics"
                )
        
        return self.comparison_enabled
    
    def render_comparison_view(self, enhanced_content: List[Dict[str, Any]], 
                             max_items: Optional[int] = None) -> None:
        """Render the side-by-side comparison view - Core Enhancement 4 requirement"""
        
        if not self.comparison_enabled:
            return
        
        if not enhanced_content:
            st.info("📝 No enhanced content available for comparison.")
            return
        
        # Filter content that has both original and enhanced versions
        comparable_content = [
            item for item in enhanced_content 
            if (item.get('enhanced', False) and 
                item.get('original_question') and 
                item.get('original_answer'))
        ]
        
        if not comparable_content:
            st.info("📝 No enhanced content with original versions found for comparison.")
            return
        
        # Comparison controls
        self._render_comparison_controls(comparable_content)
        
        # Limit items for performance
        display_limit = max_items or self.max_display_items
        display_items = comparable_content[:display_limit]
        
        st.markdown(f"### 📊 **Comparison Results** ({len(display_items)} of {len(comparable_content)} enhanced examples)")
        
        if len(comparable_content) > display_limit:
            st.info(f"📋 Showing first {display_limit} examples for performance. Total enhanced: {len(comparable_content)}")
        
        # Render each comparison
        for i, item in enumerate(display_items):
            self._render_single_comparison(item, i + 1)
    
    def _render_comparison_controls(self, comparable_content: List[Dict[str, Any]]) -> None:
        """Render comparison filtering and sorting controls"""
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sort_by = st.selectbox(
                "Sort By",
                options=["Quality Score ↓", "Quality Score ↑", "Improvement ↓", "Improvement ↑", "Length ↓", "Length ↑"],
                index=0,
                key="comparison_sort"
            )
        
        with col2:
            filter_tone = st.selectbox(
                "Filter by Tone",
                options=["All"] + list(set(item.get('enhancement_tone', 'Unknown') for item in comparable_content)),
                index=0,
                key="comparison_filter_tone"
            )
        
        with col3:
            min_quality = st.slider(
                "Min Quality Score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                key="comparison_min_quality"
            )
    
    def _render_single_comparison(self, item: Dict[str, Any], index: int) -> None:
        """Render a single comparison item - Core Enhancement 4 requirement"""
        
        # Get content
        original_question = item.get('original_question', '')
        original_answer = item.get('original_answer', '')
        enhanced_question = item.get('question', '')
        enhanced_answer = item.get('answer', '')
        
        # Get metadata
        enhancement_tone = item.get('enhancement_tone', 'Unknown')
        enhancement_mode = item.get('enhancement_mode', 'Unknown')
        quality_score = item.get('quality_score', 0)
        
        # Calculate improvements
        orig_total_len = len((original_question + ' ' + original_answer).split())
        enh_total_len = len((enhanced_question + ' ' + enhanced_answer).split())
        improvement_pct = ((enh_total_len - orig_total_len) / max(orig_total_len, 1)) * 100
        
        # Create expandable section
        with st.expander(
            f"**Example {index}** - {enhancement_tone.replace('_', ' ').title()} "
            f"(Quality: {quality_score:.2f}, Improvement: {improvement_pct:+.0f}%)",
            expanded=(index <= 2)  # Auto-expand first 2 items
        ):
            # Metadata row
            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
            
            with col_meta1:
                st.metric("Quality Score", f"{quality_score:.2f}")
            
            with col_meta2:
                st.metric("Enhancement Tone", enhancement_tone.replace('_', ' ').title())
            
            with col_meta3:
                st.metric("Total Words", f"{enh_total_len}", delta=f"{improvement_pct:+.0f}%")
            
            with col_meta4:
                if item.get('enhancement_cost'):
                    st.metric("Cost", f"${item['enhancement_cost']:.4f}")
                else:
                    st.metric("Mode", enhancement_mode.title())
            
            st.markdown("---")
            
            # Side-by-side comparison - Core Enhancement 4 requirement
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("#### 📝 **Raw Content**")
                
                # Original Question
                st.markdown("**Question:**")
                st.markdown(f'''
                <div style="
                    background-color: #f8f9fa; 
                    padding: 12px; 
                    border-radius: 8px; 
                    margin-bottom: 12px;
                    border-left: 4px solid #6c757d;
                    font-family: 'Segoe UI', sans-serif;
                ">
                    {original_question}
                </div>
                ''', unsafe_allow_html=True)
                
                # Original Answer
                st.markdown("**Answer:**")
                st.markdown(f'''
                <div style="
                    background-color: #f8f9fa; 
                    padding: 12px; 
                    border-radius: 8px;
                    border-left: 4px solid #6c757d;
                    font-family: 'Segoe UI', sans-serif;
                ">
                    {original_answer}
                </div>
                ''', unsafe_allow_html=True)
                
                # Original stats
                orig_q_len = len(original_question.split())
                orig_a_len = len(original_answer.split())
                st.caption(f"📊 Question: {orig_q_len} words | Answer: {orig_a_len} words | Total: {orig_q_len + orig_a_len}")
            
            with col_right:
                st.markdown("#### ✨ **Enhanced Content**")
                
                # Enhanced Question
                st.markdown("**Question:**")
                st.markdown(f'''
                <div style="
                    background-color: #e8f5e8; 
                    padding: 12px; 
                    border-radius: 8px; 
                    margin-bottom: 12px;
                    border-left: 4px solid #28a745;
                    font-family: 'Segoe UI', sans-serif;
                    box-shadow: 0 2px 4px rgba(40, 167, 69, 0.1);
                ">
                    {enhanced_question}
                </div>
                ''', unsafe_allow_html=True)
                
                # Enhanced Answer
                st.markdown("**Answer:**")
                st.markdown(f'''
                <div style="
                    background-color: #e8f5e8; 
                    padding: 12px; 
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                    font-family: 'Segoe UI', sans-serif;
                    box-shadow: 0 2px 4px rgba(40, 167, 69, 0.1);
                ">
                    {enhanced_answer}
                </div>
                ''', unsafe_allow_html=True)
                
                # Enhanced stats
                enh_q_len = len(enhanced_question.split())
                enh_a_len = len(enhanced_answer.split())
                st.caption(f"📊 Question: {enh_q_len} words | Answer: {enh_a_len} words | Total: {enh_q_len + enh_a_len}")
            
            # Detailed improvement metrics
            if self.show_metrics:
                st.markdown("#### 📈 **Detailed Improvement Analysis**")
                
                col_imp1, col_imp2, col_imp3, col_imp4 = st.columns(4)
                
                with col_imp1:
                    q_improvement = ((enh_q_len - orig_q_len) / max(orig_q_len, 1)) * 100
                    st.metric(
                        "Question Length", 
                        f"{enh_q_len} words",
                        delta=f"{q_improvement:+.0f}%"
                    )
                
                with col_imp2:
                    a_improvement = ((enh_a_len - orig_a_len) / max(orig_a_len, 1)) * 100
                    st.metric(
                        "Answer Length", 
                        f"{enh_a_len} words",
                        delta=f"{a_improvement:+.0f}%"
                    )
                
                with col_imp3:
                    # Calculate readability improvement (simple metric)
                    orig_avg_word_len = sum(len(word) for word in (original_question + ' ' + original_answer).split()) / max(orig_total_len, 1)
                    enh_avg_word_len = sum(len(word) for word in (enhanced_question + ' ' + enhanced_answer).split()) / max(enh_total_len, 1)
                    readability_change = ((enh_avg_word_len - orig_avg_word_len) / max(orig_avg_word_len, 1)) * 100
                    
                    st.metric(
                        "Avg Word Length",
                        f"{enh_avg_word_len:.1f} chars",
                        delta=f"{readability_change:+.1f}%"
                    )
                
                with col_imp4:
                    # Spiritual keyword density
                    spiritual_keywords = ['consciousness', 'awareness', 'meditation', 'spiritual', 'enlightenment', 'wisdom', 'truth']
                    orig_spiritual = sum(1 for word in spiritual_keywords if word in (original_question + ' ' + original_answer).lower())
                    enh_spiritual = sum(1 for word in spiritual_keywords if word in (enhanced_question + ' ' + enhanced_answer).lower())
                    
                    st.metric(
                        "Spiritual Keywords",
                        f"{enh_spiritual}",
                        delta=f"{enh_spiritual - orig_spiritual:+d}"
                    )
    
    def render_comparison_summary(self, enhanced_content: List[Dict[str, Any]]) -> None:
        """Render overall comparison summary statistics"""
        
        if not self.comparison_enabled:
            return
            
        # Filter comparable content
        comparable_content = [
            item for item in enhanced_content 
            if (item.get('enhanced', False) and 
                item.get('original_question') and 
                item.get('original_answer'))
        ]
        
        if not comparable_content:
            return
            
        st.markdown("### 📊 **Overall Enhancement Summary**")
        
        # Calculate aggregate metrics
        total_items = len(comparable_content)
        
        # Length improvements
        orig_lengths = []
        enh_lengths = []
        quality_scores = []
        
        for item in comparable_content:
            orig_len = len((item.get('original_question', '') + ' ' + item.get('original_answer', '')).split())
            enh_len = len((item.get('question', '') + ' ' + item.get('answer', '')).split())
            
            orig_lengths.append(orig_len)
            enh_lengths.append(enh_len)
            quality_scores.append(item.get('quality_score', 0))
        
        avg_orig_length = sum(orig_lengths) / len(orig_lengths) if orig_lengths else 0
        avg_enh_length = sum(enh_lengths) / len(enh_lengths) if enh_lengths else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_improvement = ((avg_enh_length - avg_orig_length) / max(avg_orig_length, 1)) * 100
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Enhanced Examples", total_items)
            
        with col2:
            st.metric("Avg Quality Score", f"{avg_quality:.2f}")
            
        with col3:
            st.metric("Avg Length Improvement", f"{avg_improvement:+.0f}%")
            
        with col4:
            total_cost = sum(item.get('enhancement_cost', 0) for item in comparable_content)
            st.metric("Total Enhancement Cost", f"${total_cost:.4f}")
        
        # Quality distribution
        quality_excellent = sum(1 for score in quality_scores if score >= 0.8)
        quality_good = sum(1 for score in quality_scores if 0.6 <= score < 0.8)
        quality_fair = sum(1 for score in quality_scores if 0.4 <= score < 0.6)
        quality_poor = sum(1 for score in quality_scores if score < 0.4)
        
        st.markdown("#### 🎯 **Quality Distribution**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 Excellent (0.8+)", quality_excellent)
        with col2:
            st.metric("🟡 Good (0.6-0.8)", quality_good)
        with col3:
            st.metric("🟠 Fair (0.4-0.6)", quality_fair)
        with col4:
            st.metric("🔴 Poor (<0.4)", quality_poor)
    
    def get_comparison_statistics(self, enhanced_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed comparison statistics for analysis"""
        
        comparable_content = [
            item for item in enhanced_content 
            if (item.get('enhanced', False) and 
                item.get('original_question') and 
                item.get('original_answer'))
        ]
        
        if not comparable_content:
            return {}
        
        # Calculate detailed statistics
        improvements = []
        quality_scores = []
        costs = []
        tones = []
        
        for item in comparable_content:
            orig_len = len((item.get('original_question', '') + ' ' + item.get('original_answer', '')).split())
            enh_len = len((item.get('question', '') + ' ' + item.get('answer', '')).split())
            
            if orig_len > 0:
                improvement = ((enh_len - orig_len) / orig_len) * 100
                improvements.append(improvement)
            
            quality_scores.append(item.get('quality_score', 0))
            costs.append(item.get('enhancement_cost', 0))
            tones.append(item.get('enhancement_tone', 'unknown'))
        
        return {
            'total_enhanced': len(comparable_content),
            'avg_improvement': sum(improvements) / len(improvements) if improvements else 0,
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'total_cost': sum(costs),
            'avg_cost_per_item': sum(costs) / len(costs) if costs else 0,
            'tone_distribution': {tone: tones.count(tone) for tone in set(tones)},
            'quality_distribution': {
                'excellent': sum(1 for score in quality_scores if score >= 0.8),
                'good': sum(1 for score in quality_scores if 0.6 <= score < 0.8),
                'fair': sum(1 for score in quality_scores if 0.4 <= score < 0.6),
                'poor': sum(1 for score in quality_scores if score < 0.4)
            },
            'improvement_range': {
                'min': min(improvements) if improvements else 0,
                'max': max(improvements) if improvements else 0,
                'median': sorted(improvements)[len(improvements)//2] if improvements else 0
            }
        }

