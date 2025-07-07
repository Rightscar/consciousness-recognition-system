"""
Rework Marking System
====================

"Mark for rework" feature for any borderline or confusing data
to enable future refinement and quality improvement.

Features:
- Mark items for rework with reasons
- Categorize rework types
- Batch rework operations
- Rework queue management
- Progress tracking
- Export rework reports
- Integration with quality scoring
"""

import streamlit as st
from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime
import json
from modules.logger import get_logger, log_event, log_user_action
import pandas as pd

class ReworkMarkingSystem:
    """
    Comprehensive rework marking system for quality improvement
    
    Features:
    - Mark items for rework with detailed reasons
    - Categorize rework types and priorities
    - Batch operations for efficient processing
    - Rework queue management and tracking
    - Progress monitoring and reporting
    - Export capabilities for external processing
    """
    
    def __init__(self):
        self.logger = get_logger("rework_marking")
        
        # Initialize session state
        if 'rework_items' not in st.session_state:
            st.session_state['rework_items'] = {}
        
        if 'rework_categories' not in st.session_state:
            st.session_state['rework_categories'] = set()
        
        if 'rework_filters' not in st.session_state:
            st.session_state['rework_filters'] = {
                'category': 'All',
                'priority': 'All',
                'status': 'All'
            }
        
        # Predefined rework categories
        self.predefined_categories = {
            'content_quality': {
                'name': 'Content Quality',
                'description': 'Issues with content accuracy, completeness, or relevance',
                'examples': ['Incomplete answers', 'Factual errors', 'Irrelevant content'],
                'priority': 'high'
            },
            'language_issues': {
                'name': 'Language Issues',
                'description': 'Grammar, spelling, or language clarity problems',
                'examples': ['Grammar errors', 'Unclear phrasing', 'Spelling mistakes'],
                'priority': 'medium'
            },
            'formatting': {
                'name': 'Formatting',
                'description': 'Structure, formatting, or presentation issues',
                'examples': ['Poor structure', 'Inconsistent formatting', 'Missing elements'],
                'priority': 'low'
            },
            'tone_style': {
                'name': 'Tone & Style',
                'description': 'Inappropriate tone or style for the target use case',
                'examples': ['Too formal', 'Too casual', 'Inconsistent style'],
                'priority': 'medium'
            },
            'length_issues': {
                'name': 'Length Issues',
                'description': 'Content too long, too short, or poorly balanced',
                'examples': ['Too verbose', 'Too brief', 'Unbalanced Q&A'],
                'priority': 'medium'
            },
            'context_missing': {
                'name': 'Missing Context',
                'description': 'Lacks necessary context or background information',
                'examples': ['Unclear references', 'Missing background', 'Ambiguous terms'],
                'priority': 'high'
            },
            'bias_sensitivity': {
                'name': 'Bias & Sensitivity',
                'description': 'Potential bias, sensitivity, or ethical concerns',
                'examples': ['Cultural bias', 'Sensitive topics', 'Unfair representations'],
                'priority': 'high'
            },
            'technical_accuracy': {
                'name': 'Technical Accuracy',
                'description': 'Technical errors or inaccuracies in specialized content',
                'examples': ['Code errors', 'Technical mistakes', 'Outdated information'],
                'priority': 'high'
            },
            'enhancement_needed': {
                'name': 'Enhancement Needed',
                'description': 'Good content that could be significantly improved',
                'examples': ['Add examples', 'More detail needed', 'Better explanations'],
                'priority': 'medium'
            },
            'unclear_intent': {
                'name': 'Unclear Intent',
                'description': 'Unclear what the content is trying to achieve',
                'examples': ['Ambiguous questions', 'Unclear goals', 'Mixed purposes'],
                'priority': 'high'
            }
        }
        
        # Priority levels
        self.priority_levels = {
            'critical': {'name': 'Critical', 'color': '#FF4B4B', 'emoji': '🔴'},
            'high': {'name': 'High', 'color': '#FF8C00', 'emoji': '🟠'},
            'medium': {'name': 'Medium', 'color': '#FFD700', 'emoji': '🟡'},
            'low': {'name': 'Low', 'color': '#90EE90', 'emoji': '🟢'}
        }
        
        # Status types
        self.status_types = {
            'marked': {'name': 'Marked for Rework', 'emoji': '🚩'},
            'in_progress': {'name': 'In Progress', 'emoji': '🔄'},
            'completed': {'name': 'Rework Completed', 'emoji': '✅'},
            'deferred': {'name': 'Deferred', 'emoji': '⏸️'},
            'cancelled': {'name': 'Cancelled', 'emoji': '❌'}
        }
    
    def render_rework_interface(self, content_data: List[Dict]):
        """Render the complete rework marking interface"""
        
        st.subheader("🚩 Rework Marking System")
        
        if not content_data:
            st.warning("⚠️ No content available for rework marking")
            return
        
        # Rework overview
        self.render_rework_overview()
        
        # Rework controls
        self.render_rework_controls()
        
        # Content review interface
        self.render_content_review_interface(content_data)
        
        # Rework queue management
        self.render_rework_queue_management()
        
        # Export and reporting
        self.render_rework_export_options()
    
    def render_rework_overview(self):
        """Render rework overview and statistics"""
        
        st.markdown("**📊 Rework Overview:**")
        
        total_items = len(st.session_state.get('rework_items', {}))
        
        if total_items == 0:
            st.info("No items marked for rework yet")
            return
        
        # Calculate statistics
        stats = self.calculate_rework_statistics()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Marked", total_items)
        
        with col2:
            st.metric("High Priority", stats['priority_counts'].get('high', 0))
        
        with col3:
            st.metric("In Progress", stats['status_counts'].get('in_progress', 0))
        
        with col4:
            st.metric("Completed", stats['status_counts'].get('completed', 0))
        
        # Progress visualization
        if total_items > 0:
            completed = stats['status_counts'].get('completed', 0)
            progress = (completed / total_items) * 100
            st.progress(progress / 100)
            st.caption(f"Rework Progress: {completed}/{total_items} ({progress:.1f}%)")
    
    def render_rework_controls(self):
        """Render rework control panel"""
        
        st.markdown("---")
        st.markdown("**🔧 Rework Controls:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Category filter
            categories = ['All'] + list(self.predefined_categories.keys()) + list(st.session_state['rework_categories'])
            category_filter = st.selectbox(
                "Filter by Category",
                categories,
                index=categories.index(st.session_state['rework_filters']['category']) if st.session_state['rework_filters']['category'] in categories else 0
            )
            st.session_state['rework_filters']['category'] = category_filter
        
        with col2:
            # Priority filter
            priorities = ['All'] + list(self.priority_levels.keys())
            priority_filter = st.selectbox(
                "Filter by Priority",
                priorities,
                format_func=lambda x: x.title() if x != 'All' else x
            )
            st.session_state['rework_filters']['priority'] = priority_filter
        
        with col3:
            # Status filter
            statuses = ['All'] + list(self.status_types.keys())
            status_filter = st.selectbox(
                "Filter by Status",
                statuses,
                format_func=lambda x: self.status_types[x]['name'] if x != 'All' else x
            )
            st.session_state['rework_filters']['status'] = status_filter
        
        with col4:
            # Quick actions
            if st.button("🔄 Refresh View"):
                st.rerun()
        
        # Batch operations
        st.markdown("**⚡ Batch Operations:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚩 Mark All Low Quality"):
                self.batch_mark_low_quality()
        
        with col2:
            if st.button("✅ Complete Selected"):
                self.batch_update_status('completed')
        
        with col3:
            if st.button("⏸️ Defer Selected"):
                self.batch_update_status('deferred')
        
        with col4:
            if st.button("🗑️ Clear Completed"):
                self.clear_completed_rework()
    
    def render_content_review_interface(self, content_data: List[Dict]):
        """Render content review interface for marking items"""
        
        st.markdown("---")
        st.markdown("**📋 Content Review:**")
        
        # Pagination
        items_per_page = 5
        total_pages = (len(content_data) + items_per_page - 1) // items_per_page
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                current_page = st.selectbox(
                    "Page",
                    range(1, total_pages + 1),
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
        else:
            current_page = 1
        
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(content_data))
        
        # Display items for current page
        for i in range(start_idx, end_idx):
            self.render_single_item_review(content_data[i], i)
    
    def render_single_item_review(self, item: Dict, item_index: int):
        """Render review interface for a single content item"""
        
        # Check if item is already marked for rework
        is_marked = str(item_index) in st.session_state['rework_items']
        rework_info = st.session_state['rework_items'].get(str(item_index), {})
        
        # Item header
        status_emoji = "🚩" if is_marked else "📄"
        quality_score = item.get('quality_score', 0.75)
        
        with st.expander(
            f"{status_emoji} Item {item_index + 1} - Quality: {quality_score:.2f} {'(Marked for Rework)' if is_marked else ''}",
            expanded=not is_marked
        ):
            
            # Display content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show item content
                content_text = self.extract_text_content(item)
                st.text_area(
                    "Content",
                    value=content_text[:500] + "..." if len(content_text) > 500 else content_text,
                    height=100,
                    disabled=True,
                    key=f"content_display_{item_index}"
                )
            
            with col2:
                # Quality indicators
                if quality_score < 0.6:
                    st.error(f"🔴 Low Quality ({quality_score:.2f})")
                elif quality_score < 0.8:
                    st.warning(f"🟡 Medium Quality ({quality_score:.2f})")
                else:
                    st.success(f"🟢 High Quality ({quality_score:.2f})")
                
                # Show current rework status
                if is_marked:
                    status = rework_info.get('status', 'marked')
                    priority = rework_info.get('priority', 'medium')
                    
                    st.write(f"**Status:** {self.status_types[status]['emoji']} {self.status_types[status]['name']}")
                    st.write(f"**Priority:** {self.priority_levels[priority]['emoji']} {self.priority_levels[priority]['name']}")
            
            # Rework marking interface
            if is_marked:
                self.render_existing_rework_interface(item_index, rework_info)
            else:
                self.render_new_rework_interface(item_index, item)
    
    def render_new_rework_interface(self, item_index: int, item: Dict):
        """Render interface for marking new item for rework"""
        
        st.markdown("**🚩 Mark for Rework:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category selection
            categories = list(self.predefined_categories.keys()) + list(st.session_state['rework_categories'])
            selected_category = st.selectbox(
                "Rework Category",
                categories,
                format_func=lambda x: self.predefined_categories.get(x, {'name': x.title()})['name'],
                key=f"category_{item_index}"
            )
            
            # Priority selection
            priority = st.selectbox(
                "Priority",
                list(self.priority_levels.keys()),
                format_func=lambda x: f"{self.priority_levels[x]['emoji']} {self.priority_levels[x]['name']}",
                index=1,  # Default to 'high'
                key=f"priority_{item_index}"
            )
        
        with col2:
            # Custom category input
            custom_category = st.text_input(
                "Custom Category",
                placeholder="Enter custom category",
                key=f"custom_cat_{item_index}"
            )
            
            if custom_category:
                st.session_state['rework_categories'].add(custom_category.lower())
                selected_category = custom_category.lower()
            
            # Reason input
            reason = st.text_area(
                "Reason for Rework",
                placeholder="Describe why this item needs rework...",
                height=80,
                key=f"reason_{item_index}"
            )
        
        # Mark button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🚩 Mark for Rework", key=f"mark_{item_index}"):
                if reason.strip():
                    self.mark_item_for_rework(item_index, selected_category, priority, reason, item)
                    st.success("Item marked for rework!")
                    st.rerun()
                else:
                    st.warning("Please provide a reason for rework")
        
        with col2:
            # Quick mark buttons for common issues
            if st.button(f"⚡ Quick Mark: Low Quality", key=f"quick_mark_{item_index}"):
                self.mark_item_for_rework(
                    item_index, 
                    'content_quality', 
                    'high', 
                    'Automatically marked due to low quality score',
                    item
                )
                st.success("Item marked for rework!")
                st.rerun()
    
    def render_existing_rework_interface(self, item_index: int, rework_info: Dict):
        """Render interface for existing rework item"""
        
        st.markdown("**🔧 Rework Management:**")
        
        # Display current rework information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Category:** {rework_info.get('category', 'Unknown').title()}")
            st.write(f"**Marked:** {rework_info.get('marked_at', 'Unknown')}")
            st.write(f"**Reason:** {rework_info.get('reason', 'No reason provided')}")
        
        with col2:
            # Status update
            current_status = rework_info.get('status', 'marked')
            new_status = st.selectbox(
                "Update Status",
                list(self.status_types.keys()),
                index=list(self.status_types.keys()).index(current_status),
                format_func=lambda x: f"{self.status_types[x]['emoji']} {self.status_types[x]['name']}",
                key=f"status_update_{item_index}"
            )
            
            # Priority update
            current_priority = rework_info.get('priority', 'medium')
            new_priority = st.selectbox(
                "Update Priority",
                list(self.priority_levels.keys()),
                index=list(self.priority_levels.keys()).index(current_priority),
                format_func=lambda x: f"{self.priority_levels[x]['emoji']} {self.priority_levels[x]['name']}",
                key=f"priority_update_{item_index}"
            )
        
        # Notes
        notes = st.text_area(
            "Rework Notes",
            value=rework_info.get('notes', ''),
            placeholder="Add notes about rework progress...",
            key=f"notes_{item_index}"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"💾 Update", key=f"update_{item_index}"):
                self.update_rework_item(item_index, new_status, new_priority, notes)
                st.success("Rework item updated!")
                st.rerun()
        
        with col2:
            if st.button(f"✅ Complete", key=f"complete_{item_index}"):
                self.update_rework_item(item_index, 'completed', new_priority, notes)
                st.success("Rework marked as completed!")
                st.rerun()
        
        with col3:
            if st.button(f"🗑️ Remove", key=f"remove_{item_index}"):
                self.remove_rework_item(item_index)
                st.success("Item removed from rework queue!")
                st.rerun()
    
    def render_rework_queue_management(self):
        """Render rework queue management interface"""
        
        st.markdown("---")
        st.markdown("**📋 Rework Queue Management:**")
        
        rework_items = st.session_state.get('rework_items', {})
        
        if not rework_items:
            st.info("No items in rework queue")
            return
        
        # Filter rework items
        filtered_items = self.filter_rework_items(rework_items)
        
        if not filtered_items:
            st.info("No items match current filters")
            return
        
        # Display filtered items
        for item_id, rework_info in filtered_items.items():
            self.render_rework_queue_item(item_id, rework_info)
    
    def render_rework_queue_item(self, item_id: str, rework_info: Dict):
        """Render a single item in the rework queue"""
        
        status = rework_info.get('status', 'marked')
        priority = rework_info.get('priority', 'medium')
        category = rework_info.get('category', 'unknown')
        
        status_emoji = self.status_types[status]['emoji']
        priority_emoji = self.priority_levels[priority]['emoji']
        
        with st.expander(
            f"{status_emoji} {priority_emoji} Item {int(item_id) + 1} - {category.title()}",
            expanded=False
        ):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Category:** {category.title()}")
                st.write(f"**Priority:** {self.priority_levels[priority]['name']}")
                st.write(f"**Status:** {self.status_types[status]['name']}")
            
            with col2:
                st.write(f"**Marked:** {rework_info.get('marked_at', 'Unknown')}")
                if rework_info.get('updated_at'):
                    st.write(f"**Updated:** {rework_info['updated_at']}")
            
            st.write(f"**Reason:** {rework_info.get('reason', 'No reason provided')}")
            
            if rework_info.get('notes'):
                st.write(f"**Notes:** {rework_info['notes']}")
            
            # Quick actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"✅ Complete", key=f"queue_complete_{item_id}"):
                    self.update_rework_item(int(item_id), 'completed', priority, rework_info.get('notes', ''))
                    st.rerun()
            
            with col2:
                if st.button(f"🔄 In Progress", key=f"queue_progress_{item_id}"):
                    self.update_rework_item(int(item_id), 'in_progress', priority, rework_info.get('notes', ''))
                    st.rerun()
            
            with col3:
                if st.button(f"🗑️ Remove", key=f"queue_remove_{item_id}"):
                    self.remove_rework_item(int(item_id))
                    st.rerun()
    
    def render_rework_export_options(self):
        """Render rework export and reporting options"""
        
        st.markdown("---")
        st.markdown("**📥 Export & Reporting:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Rework Report"):
                self.export_rework_report()
        
        with col2:
            if st.button("📋 Export Rework Queue"):
                self.export_rework_queue()
        
        with col3:
            if st.button("📈 Export Statistics"):
                self.export_rework_statistics()
    
    def mark_item_for_rework(self, item_index: int, category: str, priority: str, reason: str, item: Dict):
        """Mark an item for rework"""
        
        rework_info = {
            'category': category,
            'priority': priority,
            'reason': reason,
            'status': 'marked',
            'marked_at': datetime.now().isoformat(),
            'quality_score': item.get('quality_score', 0.75),
            'content_preview': self.extract_text_content(item)[:200] + "..." if len(self.extract_text_content(item)) > 200 else self.extract_text_content(item),
            'notes': ''
        }
        
        st.session_state['rework_items'][str(item_index)] = rework_info
        
        log_user_action("item_marked_for_rework", {
            "item_index": item_index,
            "category": category,
            "priority": priority,
            "reason": reason[:100]  # Truncate for logging
        })
    
    def update_rework_item(self, item_index: int, status: str, priority: str, notes: str):
        """Update an existing rework item"""
        
        item_id = str(item_index)
        if item_id in st.session_state['rework_items']:
            st.session_state['rework_items'][item_id].update({
                'status': status,
                'priority': priority,
                'notes': notes,
                'updated_at': datetime.now().isoformat()
            })
            
            log_user_action("rework_item_updated", {
                "item_index": item_index,
                "status": status,
                "priority": priority
            })
    
    def remove_rework_item(self, item_index: int):
        """Remove an item from rework queue"""
        
        item_id = str(item_index)
        if item_id in st.session_state['rework_items']:
            del st.session_state['rework_items'][item_id]
            
            log_user_action("rework_item_removed", {
                "item_index": item_index
            })
    
    def batch_mark_low_quality(self):
        """Batch mark all low quality items for rework"""
        
        # This would need access to content data - placeholder implementation
        log_user_action("batch_mark_low_quality_requested")
        st.info("Batch marking feature - implementation depends on content access")
    
    def batch_update_status(self, new_status: str):
        """Batch update status for selected items"""
        
        # Update all items matching current filters
        filtered_items = self.filter_rework_items(st.session_state.get('rework_items', {}))
        
        for item_id in filtered_items:
            st.session_state['rework_items'][item_id]['status'] = new_status
            st.session_state['rework_items'][item_id]['updated_at'] = datetime.now().isoformat()
        
        log_user_action("batch_status_update", {
            "new_status": new_status,
            "item_count": len(filtered_items)
        })
        
        st.success(f"Updated {len(filtered_items)} items to {new_status}")
        st.rerun()
    
    def clear_completed_rework(self):
        """Clear all completed rework items"""
        
        completed_items = [
            item_id for item_id, info in st.session_state['rework_items'].items()
            if info.get('status') == 'completed'
        ]
        
        for item_id in completed_items:
            del st.session_state['rework_items'][item_id]
        
        log_user_action("completed_rework_cleared", {
            "item_count": len(completed_items)
        })
        
        st.success(f"Cleared {len(completed_items)} completed items")
        st.rerun()
    
    def filter_rework_items(self, rework_items: Dict) -> Dict:
        """Filter rework items based on current filters"""
        
        filtered = {}
        filters = st.session_state['rework_filters']
        
        for item_id, info in rework_items.items():
            # Category filter
            if filters['category'] != 'All' and info.get('category') != filters['category']:
                continue
            
            # Priority filter
            if filters['priority'] != 'All' and info.get('priority') != filters['priority']:
                continue
            
            # Status filter
            if filters['status'] != 'All' and info.get('status') != filters['status']:
                continue
            
            filtered[item_id] = info
        
        return filtered
    
    def calculate_rework_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive rework statistics"""
        
        rework_items = st.session_state.get('rework_items', {})
        
        stats = {
            'total_items': len(rework_items),
            'category_counts': {},
            'priority_counts': {},
            'status_counts': {},
            'completion_rate': 0
        }
        
        for info in rework_items.values():
            # Count categories
            category = info.get('category', 'unknown')
            stats['category_counts'][category] = stats['category_counts'].get(category, 0) + 1
            
            # Count priorities
            priority = info.get('priority', 'medium')
            stats['priority_counts'][priority] = stats['priority_counts'].get(priority, 0) + 1
            
            # Count statuses
            status = info.get('status', 'marked')
            stats['status_counts'][status] = stats['status_counts'].get(status, 0) + 1
        
        # Calculate completion rate
        if stats['total_items'] > 0:
            completed = stats['status_counts'].get('completed', 0)
            stats['completion_rate'] = (completed / stats['total_items']) * 100
        
        return stats
    
    def export_rework_report(self):
        """Export comprehensive rework report"""
        
        try:
            rework_items = st.session_state.get('rework_items', {})
            stats = self.calculate_rework_statistics()
            
            report_data = {
                'report_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_items': len(rework_items),
                    'completion_rate': stats['completion_rate']
                },
                'statistics': stats,
                'rework_items': rework_items,
                'categories': {
                    'predefined': self.predefined_categories,
                    'custom': list(st.session_state['rework_categories'])
                }
            }
            
            report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📊 Download Rework Report",
                data=report_json,
                file_name=f"rework_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            log_event("rework_report_exported", {
                "total_items": len(rework_items),
                "completion_rate": stats['completion_rate']
            }, "rework_marking")
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            self.logger.error(f"Rework report export failed: {str(e)}")
    
    def export_rework_queue(self):
        """Export current rework queue"""
        
        try:
            rework_items = st.session_state.get('rework_items', {})
            
            # Convert to list format for easier processing
            queue_data = []
            for item_id, info in rework_items.items():
                queue_item = {
                    'item_index': int(item_id),
                    **info
                }
                queue_data.append(queue_item)
            
            # Sort by priority and marked date
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            queue_data.sort(key=lambda x: (
                priority_order.get(x.get('priority', 'medium'), 2),
                x.get('marked_at', '')
            ))
            
            queue_json = json.dumps(queue_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📋 Download Rework Queue",
                data=queue_json,
                file_name=f"rework_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            self.logger.error(f"Rework queue export failed: {str(e)}")
    
    def export_rework_statistics(self):
        """Export rework statistics"""
        
        try:
            stats = self.calculate_rework_statistics()
            
            stats_data = {
                'timestamp': datetime.now().isoformat(),
                'statistics': stats,
                'summary': {
                    'total_marked': stats['total_items'],
                    'completion_rate': stats['completion_rate'],
                    'top_category': max(stats['category_counts'].items(), key=lambda x: x[1])[0] if stats['category_counts'] else None,
                    'high_priority_count': stats['priority_counts'].get('high', 0) + stats['priority_counts'].get('critical', 0)
                }
            }
            
            stats_json = json.dumps(stats_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📈 Download Statistics",
                data=stats_json,
                file_name=f"rework_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            self.logger.error(f"Rework statistics export failed: {str(e)}")
    
    def extract_text_content(self, item: Dict) -> str:
        """Extract text content from item for display"""
        
        text_parts = []
        
        # Common text fields
        for field in ['question', 'answer', 'content', 'text', 'instruction', 'input', 'output']:
            if field in item and isinstance(item[field], str):
                text_parts.append(item[field])
        
        return ' '.join(text_parts)

# Integration function for main app
def render_rework_marking_system(content_data: List[Dict]):
    """
    Render rework marking system in main app
    
    Usage:
    from modules.rework_marking_system import render_rework_marking_system
    
    render_rework_marking_system(enhanced_content)
    """
    
    rework_system = ReworkMarkingSystem()
    rework_system.render_rework_interface(content_data)

# Compact rework interface for sidebar
def render_compact_rework_interface():
    """
    Render compact rework interface for sidebar
    
    Usage:
    with st.sidebar:
        render_compact_rework_interface()
    """
    
    st.markdown("### 🚩 Rework Queue")
    
    rework_items = st.session_state.get('rework_items', {})
    total_items = len(rework_items)
    
    if total_items == 0:
        st.info("No items marked for rework")
        return
    
    # Quick statistics
    stats = {}
    for info in rework_items.values():
        status = info.get('status', 'marked')
        stats[status] = stats.get(status, 0) + 1
    
    st.metric("Total Marked", total_items)
    
    if stats.get('completed', 0) > 0:
        completion_rate = (stats['completed'] / total_items) * 100
        st.metric("Completed", f"{stats['completed']} ({completion_rate:.0f}%)")
    
    # Quick actions
    if st.button("🚩 View Rework Queue"):
        st.session_state['show_rework_system'] = True

if __name__ == "__main__":
    # Test the rework marking system
    st.set_page_config(page_title="Rework Marking System Test", layout="wide")
    
    st.title("Rework Marking System Test")
    
    # Sample data
    sample_data = [
        {
            "question": "What is AI?",
            "answer": "AI is artificial intelligence.",
            "quality_score": 0.45  # Low quality
        },
        {
            "question": "How does machine learning work?",
            "answer": "Machine learning uses algorithms to learn patterns from data and make predictions or decisions without being explicitly programmed for each specific task.",
            "quality_score": 0.85  # High quality
        },
        {
            "question": "Explain neural networks",
            "answer": "Neural networks are computing systems inspired by biological neural networks.",
            "quality_score": 0.65  # Medium quality
        }
    ]
    
    # Render rework marking system
    render_rework_marking_system(sample_data)

