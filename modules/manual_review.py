"""
Enhanced Manual Review Module
============================

Provides comprehensive manual review capabilities for AI training data before export.
Includes editable fields, include/exclude checkboxes, and quality control features.

Core Enhancement 1: Manual Review Before Export
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime


class ManualReviewInterface:
    """Enhanced manual review interface for training data quality control"""
    
    def __init__(self):
        self.items_per_page = 10
        
    def render_review_interface(self, content: List[Dict[str, Any]]) -> None:
        """Render the complete manual review interface"""
        
        if not content:
            st.info("📝 No content available for review.")
            return
            
        st.markdown("### ✏️ **Manual Review & Quality Control**")
        st.markdown("Review each example and decide whether to include it in the final export.")
        
        # Initialize include flags if not present
        for item in content:
            if 'include' not in item:
                item['include'] = True
                
        # Review controls
        self._render_review_controls(content)
        
        # Pagination and filtering
        filtered_content = self._apply_filters(content)
        
        # Review interface
        self._render_review_items(filtered_content)
        
        # Bulk actions
        self._render_bulk_actions(content)
        
        # Review summary
        self._render_review_summary(content)
        
    def _render_review_controls(self, content: List[Dict[str, Any]]) -> None:
        """Render review control options"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.session_state.review_quality_filter = st.selectbox(
                "Quality Filter",
                options=['All', 'Excellent (0.8+)', 'Good (0.6+)', 'Fair (0.4+)', 'Poor (<0.4)', 'Needs Review'],
                index=0,
                key="review_quality_filter_select"
            )
        
        with col2:
            st.session_state.review_sort_by = st.selectbox(
                "Sort By",
                options=['Quality Score ↓', 'Quality Score ↑', 'Word Count ↓', 'Word Count ↑', 'Recently Modified'],
                index=0,
                key="review_sort_select"
            )
        
        with col3:
            st.session_state.review_show_enhanced_only = st.checkbox(
                "Enhanced Only",
                value=False,
                help="Show only AI-enhanced examples"
            )
            
        with col4:
            st.session_state.review_items_per_page = st.selectbox(
                "Items per Page",
                options=[5, 10, 20, 50],
                index=1,
                key="review_items_per_page_select"
            )
            
    def _apply_filters(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply filters and sorting to content"""
        
        filtered_content = content.copy()
        
        # Quality filter
        quality_filter = st.session_state.get('review_quality_filter', 'All')
        if quality_filter != 'All':
            if quality_filter == 'Excellent (0.8+)':
                filtered_content = [item for item in filtered_content if item.get('quality_score', 0) >= 0.8]
            elif quality_filter == 'Good (0.6+)':
                filtered_content = [item for item in filtered_content if 0.6 <= item.get('quality_score', 0) < 0.8]
            elif quality_filter == 'Fair (0.4+)':
                filtered_content = [item for item in filtered_content if 0.4 <= item.get('quality_score', 0) < 0.6]
            elif quality_filter == 'Poor (<0.4)':
                filtered_content = [item for item in filtered_content if item.get('quality_score', 0) < 0.4]
            elif quality_filter == 'Needs Review':
                filtered_content = [item for item in filtered_content if item.get('manually_edited', False) or item.get('quality_score', 0) < 0.5]
        
        # Enhanced only filter
        if st.session_state.get('review_show_enhanced_only', False):
            filtered_content = [item for item in filtered_content if item.get('enhanced', False)]
        
        # Sorting
        sort_by = st.session_state.get('review_sort_by', 'Quality Score ↓')
        if sort_by == 'Quality Score ↓':
            filtered_content.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        elif sort_by == 'Quality Score ↑':
            filtered_content.sort(key=lambda x: x.get('quality_score', 0))
        elif sort_by == 'Word Count ↓':
            filtered_content.sort(key=lambda x: len((x.get('question', '') + ' ' + x.get('answer', '')).split()), reverse=True)
        elif sort_by == 'Word Count ↑':
            filtered_content.sort(key=lambda x: len((x.get('question', '') + ' ' + x.get('answer', '')).split()))
        elif sort_by == 'Recently Modified':
            filtered_content.sort(key=lambda x: x.get('last_modified', 0), reverse=True)
        
        return filtered_content
        
    def _render_review_items(self, content: List[Dict[str, Any]]) -> None:
        """Render individual review items with editing capabilities"""
        
        items_per_page = st.session_state.get('review_items_per_page', 10)
        
        # Pagination
        total_items = len(content)
        total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 1
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                current_page = st.selectbox(
                    f"Page (1-{total_pages})",
                    options=list(range(1, total_pages + 1)),
                    index=0,
                    key="review_page_select"
                ) - 1
        else:
            current_page = 0
        
        # Calculate slice
        start_idx = current_page * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        page_content = content[start_idx:end_idx]
        
        # Render items
        for idx, item in enumerate(page_content):
            item_id = f"review_{start_idx + idx}"
            self._render_single_review_item(item, item_id, start_idx + idx + 1)
            
    def _render_single_review_item(self, item: Dict[str, Any], item_id: str, item_number: int) -> None:
        """Render a single review item with editing capabilities"""
        
        # Quality score and include checkbox
        quality_score = item.get('quality_score', 0)
        quality_class = self._get_quality_class(quality_score)
        
        # Container for the item
        with st.container():
            # Header with include checkbox and quality
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            
            with col1:
                # Include checkbox - Core Enhancement 1 requirement
                include_key = f"include_{item_id}"
                item['include'] = st.checkbox(
                    "Include",
                    value=item.get('include', True),
                    key=include_key,
                    help="Include this example in the final export"
                )
            
            with col2:
                st.markdown(f"**Item #{item_number}**")
                
            with col3:
                st.markdown(f"<span class='{quality_class}'>Quality: {quality_score:.2f}</span>", unsafe_allow_html=True)
                
            with col4:
                if item.get('enhanced', False):
                    st.markdown("🤖 **Enhanced**")
                if item.get('manually_edited', False):
                    st.markdown("✏️ **Edited**")
            
            # Editable content - Core Enhancement 1 requirement
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Question:**")
                original_question = item.get('question', '')
                edited_question = st.text_area(
                    "Question",
                    value=original_question,
                    key=f"question_{item_id}",
                    label_visibility="collapsed",
                    height=100,
                    help="Edit the question text"
                )
                
                # Update if changed
                if edited_question != original_question:
                    item['question'] = edited_question
                    item['manually_edited'] = True
                    item['last_modified'] = time.time()
            
            with col2:
                st.markdown("**Answer:**")
                original_answer = item.get('answer', '')
                edited_answer = st.text_area(
                    "Answer",
                    value=original_answer,
                    key=f"answer_{item_id}",
                    label_visibility="collapsed",
                    height=100,
                    help="Edit the answer text"
                )
                
                # Update if changed
                if edited_answer != original_answer:
                    item['answer'] = edited_answer
                    item['manually_edited'] = True
                    item['last_modified'] = time.time()
            
            # Metadata and actions
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                word_count = len((edited_question + ' ' + edited_answer).split())
                st.caption(f"📝 Words: {word_count}")
                
            with col2:
                if item.get('enhancement_tone'):
                    tone = item['enhancement_tone'].replace('_', ' ').title()
                    st.caption(f"🎭 Tone: {tone}")
                    
            with col3:
                if item.get('content_type'):
                    st.caption(f"📄 Type: {item['content_type']}")
                    
            with col4:
                # Quick actions
                if st.button("🗑️", key=f"delete_{item_id}", help="Remove this item"):
                    item['include'] = False
                    st.rerun()
            
            st.divider()
            
    def _render_bulk_actions(self, content: List[Dict[str, Any]]) -> None:
        """Render bulk action controls"""
        
        st.markdown("### 🔧 **Bulk Actions**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("✅ Include All", help="Include all items in export"):
                for item in content:
                    item['include'] = True
                st.success("✅ All items included!")
                st.rerun()
                
        with col2:
            if st.button("❌ Exclude All", help="Exclude all items from export"):
                for item in content:
                    item['include'] = False
                st.success("❌ All items excluded!")
                st.rerun()
                
        with col3:
            quality_threshold = st.number_input(
                "Quality Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Include only items above this quality score"
            )
            if st.button("🎯 Apply Threshold"):
                for item in content:
                    item['include'] = item.get('quality_score', 0) >= quality_threshold
                included_count = sum(1 for item in content if item['include'])
                st.success(f"✅ Applied threshold: {included_count} items included!")
                st.rerun()
                
        with col4:
            if st.button("🤖 Enhanced Only", help="Include only AI-enhanced items"):
                for item in content:
                    item['include'] = item.get('enhanced', False)
                included_count = sum(1 for item in content if item['include'])
                st.success(f"✅ Enhanced only: {included_count} items included!")
                st.rerun()
                
    def _render_review_summary(self, content: List[Dict[str, Any]]) -> None:
        """Render review summary statistics"""
        
        st.markdown("### 📊 **Review Summary**")
        
        total_items = len(content)
        included_items = sum(1 for item in content if item.get('include', True))
        excluded_items = total_items - included_items
        manually_edited = sum(1 for item in content if item.get('manually_edited', False))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Items", total_items)
            
        with col2:
            st.metric("✅ Included", included_items)
            
        with col3:
            st.metric("❌ Excluded", excluded_items)
            
        with col4:
            st.metric("✏️ Manually Edited", manually_edited)
            
        # Quality breakdown for included items
        if included_items > 0:
            included_content = [item for item in content if item.get('include', True)]
            
            quality_counts = {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
            for item in included_content:
                score = item.get('quality_score', 0)
                if score >= 0.8:
                    quality_counts['Excellent'] += 1
                elif score >= 0.6:
                    quality_counts['Good'] += 1
                elif score >= 0.4:
                    quality_counts['Fair'] += 1
                else:
                    quality_counts['Poor'] += 1
            
            st.markdown("#### 🎯 **Quality Breakdown (Included Items)**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🟢 Excellent", quality_counts['Excellent'])
            with col2:
                st.metric("🟡 Good", quality_counts['Good'])
            with col3:
                st.metric("🟠 Fair", quality_counts['Fair'])
            with col4:
                st.metric("🔴 Poor", quality_counts['Poor'])
                
    def _get_quality_class(self, score: float) -> str:
        """Get CSS class for quality score"""
        if score >= 0.8:
            return "quality-excellent"
        elif score >= 0.6:
            return "quality-good"
        elif score >= 0.4:
            return "quality-fair"
        else:
            return "quality-poor"
            
    def finalize_content_for_export(self, content: List[Dict[str, Any]], quality_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Finalize content for export based on manual review - Core Enhancement 1 requirement"""
        
        # Filter content based on include flags and quality threshold
        export_content = [
            item for item in content
            if item.get('quality_score', 0) >= quality_threshold and item.get('include', True)
        ]
        
        return export_content
        
    def get_review_statistics(self, content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get comprehensive review statistics"""
        
        total_items = len(content)
        included_items = sum(1 for item in content if item.get('include', True))
        excluded_items = total_items - included_items
        manually_edited = sum(1 for item in content if item.get('manually_edited', False))
        enhanced_items = sum(1 for item in content if item.get('enhanced', False))
        
        # Quality breakdown
        quality_counts = {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
        for item in content:
            if item.get('include', True):
                score = item.get('quality_score', 0)
                if score >= 0.8:
                    quality_counts['Excellent'] += 1
                elif score >= 0.6:
                    quality_counts['Good'] += 1
                elif score >= 0.4:
                    quality_counts['Fair'] += 1
                else:
                    quality_counts['Poor'] += 1
        
        return {
            'total_items': total_items,
            'included_items': included_items,
            'excluded_items': excluded_items,
            'manually_edited': manually_edited,
            'enhanced_items': enhanced_items,
            'quality_breakdown': quality_counts,
            'inclusion_rate': (included_items / total_items * 100) if total_items > 0 else 0,
            'edit_rate': (manually_edited / total_items * 100) if total_items > 0 else 0
        }

