"""
Visual Diff Viewer Module
=========================

Provides side-by-side comparison of original vs AI-enhanced content
to improve trust and manual review efficiency.

Features:
- Side-by-side diff visualization
- Highlighted changes and improvements
- Character and word-level differences
- Quality metrics overlay
- Export diff reports
- Interactive review interface
"""

import streamlit as st
import difflib
from typing import List, Dict, Any, Tuple, Optional
import re
import html
from modules.logger import get_logger, log_event, log_user_action
import pandas as pd
from datetime import datetime

class VisualDiffViewer:
    """
    Advanced visual diff viewer for content comparison
    
    Features:
    - Side-by-side original vs enhanced comparison
    - Highlighted additions, deletions, and modifications
    - Quality metrics and statistics
    - Interactive review and approval interface
    - Export capabilities for diff reports
    """
    
    def __init__(self):
        self.logger = get_logger("diff_viewer")
        
        # Initialize session state for diff viewer
        if 'diff_viewer_mode' not in st.session_state:
            st.session_state['diff_viewer_mode'] = 'side_by_side'
        
        if 'diff_viewer_show_stats' not in st.session_state:
            st.session_state['diff_viewer_show_stats'] = True
        
        if 'diff_approved_items' not in st.session_state:
            st.session_state['diff_approved_items'] = set()
    
    def render_diff_viewer_interface(self, original_content: List[Dict], enhanced_content: List[Dict]):
        """Render the complete visual diff viewer interface"""
        
        st.subheader("🔍 Visual Diff Viewer - Original vs Enhanced")
        
        if not original_content or not enhanced_content:
            st.warning("⚠️ No content available for comparison")
            return
        
        # Diff viewer controls
        self.render_diff_controls()
        
        # Statistics overview
        if st.session_state['diff_viewer_show_stats']:
            self.render_diff_statistics(original_content, enhanced_content)
        
        # Content comparison
        self.render_content_comparison(original_content, enhanced_content)
        
        # Bulk actions
        self.render_bulk_actions(original_content, enhanced_content)
    
    def render_diff_controls(self):
        """Render diff viewer control panel"""
        
        st.markdown("**Diff Viewer Controls:**")
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            view_mode = st.selectbox(
                "View Mode",
                ["side_by_side", "unified", "inline"],
                index=0,
                help="Choose how to display the differences"
            )
            st.session_state['diff_viewer_mode'] = view_mode
        
        with col2:
            show_stats = st.checkbox(
                "Show Statistics",
                value=st.session_state['diff_viewer_show_stats'],
                help="Display comparison statistics"
            )
            st.session_state['diff_viewer_show_stats'] = show_stats
        
        with col3:
            highlight_level = st.selectbox(
                "Highlight Level",
                ["word", "character", "line"],
                index=0,
                help="Granularity of difference highlighting"
            )
        
        with col4:
            if st.button("🔄 Refresh Comparison"):
                st.rerun()
    
    def render_diff_statistics(self, original_content: List[Dict], enhanced_content: List[Dict]):
        """Render comparison statistics"""
        
        st.markdown("---")
        st.markdown("**📊 Comparison Statistics:**")
        
        # Calculate statistics
        stats = self.calculate_diff_statistics(original_content, enhanced_content)
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Items", stats['total_items'])
        
        with col2:
            st.metric("Modified Items", stats['modified_items'])
        
        with col3:
            st.metric("Avg Length Change", f"{stats['avg_length_change']:+.1f}%")
        
        with col4:
            st.metric("Quality Score", f"{stats['avg_quality_score']:.2f}")
        
        with col5:
            approved_count = len(st.session_state['diff_approved_items'])
            st.metric("Approved", f"{approved_count}/{stats['total_items']}")
        
        # Detailed statistics
        with st.expander("📈 Detailed Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Content Changes:**")
                st.write(f"• Items with additions: {stats['items_with_additions']}")
                st.write(f"• Items with deletions: {stats['items_with_deletions']}")
                st.write(f"• Items unchanged: {stats['items_unchanged']}")
                st.write(f"• Average word count change: {stats['avg_word_change']:+.1f}")
            
            with col2:
                st.markdown("**Quality Metrics:**")
                st.write(f"• High quality items (>0.8): {stats['high_quality_items']}")
                st.write(f"• Medium quality items (0.6-0.8): {stats['medium_quality_items']}")
                st.write(f"• Low quality items (<0.6): {stats['low_quality_items']}")
                st.write(f"• Items needing review: {stats['items_needing_review']}")
    
    def calculate_diff_statistics(self, original_content: List[Dict], enhanced_content: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive diff statistics"""
        
        total_items = min(len(original_content), len(enhanced_content))
        modified_items = 0
        items_with_additions = 0
        items_with_deletions = 0
        items_unchanged = 0
        total_length_change = 0
        total_word_change = 0
        quality_scores = []
        high_quality_items = 0
        medium_quality_items = 0
        low_quality_items = 0
        items_needing_review = 0
        
        for i in range(total_items):
            original_item = original_content[i]
            enhanced_item = enhanced_content[i]
            
            # Get text content
            orig_text = self.extract_text_content(original_item)
            enh_text = self.extract_text_content(enhanced_item)
            
            # Calculate changes
            if orig_text != enh_text:
                modified_items += 1
                
                # Length change
                length_change = (len(enh_text) - len(orig_text)) / len(orig_text) * 100 if orig_text else 0
                total_length_change += length_change
                
                # Word count change
                orig_words = len(orig_text.split())
                enh_words = len(enh_text.split())
                word_change = enh_words - orig_words
                total_word_change += word_change
                
                # Check for additions/deletions
                diff = list(difflib.unified_diff(orig_text.splitlines(), enh_text.splitlines()))
                has_additions = any(line.startswith('+') for line in diff)
                has_deletions = any(line.startswith('-') for line in diff)
                
                if has_additions:
                    items_with_additions += 1
                if has_deletions:
                    items_with_deletions += 1
            else:
                items_unchanged += 1
            
            # Quality scoring (mock implementation - replace with actual quality metrics)
            quality_score = enhanced_item.get('quality_score', 0.75)
            quality_scores.append(quality_score)
            
            if quality_score > 0.8:
                high_quality_items += 1
            elif quality_score > 0.6:
                medium_quality_items += 1
            else:
                low_quality_items += 1
                items_needing_review += 1
        
        return {
            'total_items': total_items,
            'modified_items': modified_items,
            'items_with_additions': items_with_additions,
            'items_with_deletions': items_with_deletions,
            'items_unchanged': items_unchanged,
            'avg_length_change': total_length_change / modified_items if modified_items > 0 else 0,
            'avg_word_change': total_word_change / modified_items if modified_items > 0 else 0,
            'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'high_quality_items': high_quality_items,
            'medium_quality_items': medium_quality_items,
            'low_quality_items': low_quality_items,
            'items_needing_review': items_needing_review
        }
    
    def render_content_comparison(self, original_content: List[Dict], enhanced_content: List[Dict]):
        """Render side-by-side content comparison"""
        
        st.markdown("---")
        st.markdown("**📋 Content Comparison:**")
        
        total_items = min(len(original_content), len(enhanced_content))
        
        if total_items == 0:
            st.warning("No content to compare")
            return
        
        # Pagination for large datasets
        items_per_page = 5
        total_pages = (total_items + items_per_page - 1) // items_per_page
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                format_func=lambda x: f"Page {x} of {total_pages}"
            )
        
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        # Display items for current page
        for i in range(start_idx, end_idx):
            self.render_single_item_comparison(
                original_content[i], 
                enhanced_content[i], 
                item_index=i
            )
    
    def render_single_item_comparison(self, original_item: Dict, enhanced_item: Dict, item_index: int):
        """Render comparison for a single content item"""
        
        # Extract text content
        orig_text = self.extract_text_content(original_item)
        enh_text = self.extract_text_content(enhanced_item)
        
        # Check if item is approved
        is_approved = item_index in st.session_state['diff_approved_items']
        
        # Item header
        with st.expander(
            f"📄 Item {item_index + 1} {'✅' if is_approved else '⏳'} - {self.get_item_summary(original_item)}", 
            expanded=not is_approved
        ):
            
            # Quality indicators
            quality_score = enhanced_item.get('quality_score', 0.75)
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if quality_score > 0.8:
                    st.success(f"🟢 High Quality ({quality_score:.2f})")
                elif quality_score > 0.6:
                    st.warning(f"🟡 Medium Quality ({quality_score:.2f})")
                else:
                    st.error(f"🔴 Low Quality ({quality_score:.2f})")
            
            with col2:
                length_change = ((len(enh_text) - len(orig_text)) / len(orig_text) * 100) if orig_text else 0
                st.metric("Length Change", f"{length_change:+.1f}%")
            
            with col3:
                word_change = len(enh_text.split()) - len(orig_text.split())
                st.metric("Word Change", f"{word_change:+d}")
            
            # Content comparison based on view mode
            if st.session_state['diff_viewer_mode'] == 'side_by_side':
                self.render_side_by_side_diff(orig_text, enh_text)
            elif st.session_state['diff_viewer_mode'] == 'unified':
                self.render_unified_diff(orig_text, enh_text)
            else:  # inline
                self.render_inline_diff(orig_text, enh_text)
            
            # Item actions
            self.render_item_actions(item_index, original_item, enhanced_item)
    
    def render_side_by_side_diff(self, original_text: str, enhanced_text: str):
        """Render side-by-side diff view"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔵 Original Content:**")
            st.text_area(
                "Original",
                value=original_text,
                height=200,
                disabled=True,
                key=f"orig_{hash(original_text)}"
            )
        
        with col2:
            st.markdown("**✨ Enhanced Content:**")
            st.text_area(
                "Enhanced",
                value=enhanced_text,
                height=200,
                disabled=True,
                key=f"enh_{hash(enhanced_text)}"
            )
        
        # Highlight differences
        if original_text != enhanced_text:
            st.markdown("**🔍 Highlighted Differences:**")
            diff_html = self.generate_html_diff(original_text, enhanced_text)
            st.markdown(diff_html, unsafe_allow_html=True)
    
    def render_unified_diff(self, original_text: str, enhanced_text: str):
        """Render unified diff view"""
        
        st.markdown("**🔍 Unified Diff View:**")
        
        diff_lines = list(difflib.unified_diff(
            original_text.splitlines(keepends=True),
            enhanced_text.splitlines(keepends=True),
            fromfile="Original",
            tofile="Enhanced",
            lineterm=""
        ))
        
        if diff_lines:
            diff_text = ''.join(diff_lines)
            st.code(diff_text, language='diff')
        else:
            st.info("No differences found")
    
    def render_inline_diff(self, original_text: str, enhanced_text: str):
        """Render inline diff view"""
        
        st.markdown("**🔍 Inline Diff View:**")
        
        # Generate word-level diff
        orig_words = original_text.split()
        enh_words = enhanced_text.split()
        
        diff = list(difflib.unified_diff(orig_words, enh_words, lineterm=""))
        
        if diff:
            # Process diff to create inline view
            inline_html = self.generate_inline_diff_html(orig_words, enh_words)
            st.markdown(inline_html, unsafe_allow_html=True)
        else:
            st.info("No differences found")
    
    def generate_html_diff(self, original_text: str, enhanced_text: str) -> str:
        """Generate HTML diff with highlighting"""
        
        # Use difflib to generate HTML diff
        differ = difflib.HtmlDiff(wrapcolumn=60)
        
        orig_lines = original_text.splitlines()
        enh_lines = enhanced_text.splitlines()
        
        diff_html = differ.make_table(
            orig_lines,
            enh_lines,
            fromdesc="Original",
            todesc="Enhanced",
            context=True,
            numlines=2
        )
        
        # Customize styling
        styled_html = f"""
        <div style="font-size: 12px; max-height: 300px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px;">
            {diff_html}
        </div>
        """
        
        return styled_html
    
    def generate_inline_diff_html(self, orig_words: List[str], enh_words: List[str]) -> str:
        """Generate inline diff HTML"""
        
        # Simple word-level diff
        matcher = difflib.SequenceMatcher(None, orig_words, enh_words)
        
        html_parts = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                html_parts.append(' '.join(orig_words[i1:i2]))
            elif tag == 'delete':
                deleted_text = ' '.join(orig_words[i1:i2])
                html_parts.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{html.escape(deleted_text)}</span>')
            elif tag == 'insert':
                inserted_text = ' '.join(enh_words[j1:j2])
                html_parts.append(f'<span style="background-color: #ccffcc;">{html.escape(inserted_text)}</span>')
            elif tag == 'replace':
                deleted_text = ' '.join(orig_words[i1:i2])
                inserted_text = ' '.join(enh_words[j1:j2])
                html_parts.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{html.escape(deleted_text)}</span>')
                html_parts.append(f'<span style="background-color: #ccffcc;">{html.escape(inserted_text)}</span>')
        
        return f'<div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; line-height: 1.6;">{" ".join(html_parts)}</div>'
    
    def render_item_actions(self, item_index: int, original_item: Dict, enhanced_item: Dict):
        """Render actions for individual items"""
        
        st.markdown("**Actions:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            is_approved = item_index in st.session_state['diff_approved_items']
            
            if st.button(
                "✅ Approve" if not is_approved else "❌ Unapprove",
                key=f"approve_{item_index}"
            ):
                if is_approved:
                    st.session_state['diff_approved_items'].discard(item_index)
                    log_user_action("item_unapproved", {"item_index": item_index})
                else:
                    st.session_state['diff_approved_items'].add(item_index)
                    log_user_action("item_approved", {"item_index": item_index})
                st.rerun()
        
        with col2:
            if st.button("🔄 Regenerate", key=f"regen_{item_index}"):
                log_user_action("item_regenerate_requested", {"item_index": item_index})
                st.info("Regeneration requested - feature coming soon!")
        
        with col3:
            if st.button("📝 Edit", key=f"edit_{item_index}"):
                log_user_action("item_edit_requested", {"item_index": item_index})
                st.session_state[f'edit_mode_{item_index}'] = True
                st.rerun()
        
        with col4:
            if st.button("🚩 Flag for Review", key=f"flag_{item_index}"):
                log_user_action("item_flagged", {"item_index": item_index})
                st.warning("Item flagged for manual review")
        
        # Edit mode
        if st.session_state.get(f'edit_mode_{item_index}', False):
            self.render_edit_interface(item_index, enhanced_item)
    
    def render_edit_interface(self, item_index: int, enhanced_item: Dict):
        """Render inline edit interface"""
        
        st.markdown("**✏️ Edit Mode:**")
        
        # Extract current content
        current_text = self.extract_text_content(enhanced_item)
        
        # Edit interface
        edited_text = st.text_area(
            "Edit Enhanced Content",
            value=current_text,
            height=150,
            key=f"edit_text_{item_index}"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Changes", key=f"save_{item_index}"):
                # Update the enhanced item (this would need to be connected to your data structure)
                log_user_action("item_edited", {
                    "item_index": item_index,
                    "original_length": len(current_text),
                    "edited_length": len(edited_text)
                })
                st.session_state[f'edit_mode_{item_index}'] = False
                st.success("Changes saved!")
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel", key=f"cancel_{item_index}"):
                st.session_state[f'edit_mode_{item_index}'] = False
                st.rerun()
    
    def render_bulk_actions(self, original_content: List[Dict], enhanced_content: List[Dict]):
        """Render bulk actions for multiple items"""
        
        st.markdown("---")
        st.markdown("**🔧 Bulk Actions:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("✅ Approve All High Quality"):
                high_quality_items = [
                    i for i, item in enumerate(enhanced_content)
                    if item.get('quality_score', 0) > 0.8
                ]
                st.session_state['diff_approved_items'].update(high_quality_items)
                log_user_action("bulk_approve_high_quality", {"count": len(high_quality_items)})
                st.success(f"Approved {len(high_quality_items)} high quality items")
                st.rerun()
        
        with col2:
            if st.button("🚩 Flag All Low Quality"):
                low_quality_items = [
                    i for i, item in enumerate(enhanced_content)
                    if item.get('quality_score', 0) < 0.6
                ]
                log_user_action("bulk_flag_low_quality", {"count": len(low_quality_items)})
                st.warning(f"Flagged {len(low_quality_items)} low quality items for review")
        
        with col3:
            if st.button("📊 Export Diff Report"):
                self.export_diff_report(original_content, enhanced_content)
        
        with col4:
            if st.button("🔄 Reset All Approvals"):
                st.session_state['diff_approved_items'].clear()
                log_user_action("bulk_reset_approvals")
                st.info("All approvals reset")
                st.rerun()
        
        # Approval summary
        total_items = len(enhanced_content)
        approved_items = len(st.session_state['diff_approved_items'])
        
        if total_items > 0:
            approval_percentage = (approved_items / total_items) * 100
            st.progress(approval_percentage / 100)
            st.caption(f"Approval Progress: {approved_items}/{total_items} ({approval_percentage:.1f}%)")
    
    def export_diff_report(self, original_content: List[Dict], enhanced_content: List[Dict]):
        """Export comprehensive diff report"""
        
        try:
            # Generate report data
            report_data = []
            
            for i, (orig_item, enh_item) in enumerate(zip(original_content, enhanced_content)):
                orig_text = self.extract_text_content(orig_item)
                enh_text = self.extract_text_content(enh_item)
                
                report_data.append({
                    'Item_Index': i + 1,
                    'Original_Length': len(orig_text),
                    'Enhanced_Length': len(enh_text),
                    'Length_Change_Percent': ((len(enh_text) - len(orig_text)) / len(orig_text) * 100) if orig_text else 0,
                    'Word_Count_Original': len(orig_text.split()),
                    'Word_Count_Enhanced': len(enh_text.split()),
                    'Quality_Score': enh_item.get('quality_score', 0.75),
                    'Is_Approved': i in st.session_state['diff_approved_items'],
                    'Has_Changes': orig_text != enh_text,
                    'Original_Content': orig_text[:200] + "..." if len(orig_text) > 200 else orig_text,
                    'Enhanced_Content': enh_text[:200] + "..." if len(enh_text) > 200 else enh_text
                })
            
            # Create DataFrame
            df = pd.DataFrame(report_data)
            
            # Generate CSV
            csv_data = df.to_csv(index=False)
            
            # Download button
            st.download_button(
                "📥 Download Diff Report (CSV)",
                data=csv_data,
                file_name=f"diff_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            log_event("diff_report_exported", {
                "total_items": len(report_data),
                "approved_items": sum(1 for item in report_data if item['Is_Approved'])
            }, "diff_viewer")
        
        except Exception as e:
            st.error(f"Failed to export diff report: {str(e)}")
            self.logger.error(f"Diff report export failed: {str(e)}")
    
    def extract_text_content(self, item: Dict) -> str:
        """Extract text content from item dictionary"""
        
        # Handle different item structures
        if isinstance(item, str):
            return item
        
        # Try common keys
        for key in ['content', 'text', 'answer', 'response', 'question']:
            if key in item:
                content = item[key]
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict):
                    # Handle nested structures
                    return str(content)
        
        # Fallback to string representation
        return str(item)
    
    def get_item_summary(self, item: Dict) -> str:
        """Get a brief summary of the item for display"""
        
        text = self.extract_text_content(item)
        
        # Return first 50 characters
        if len(text) > 50:
            return text[:47] + "..."
        return text

# Integration function for main app
def render_visual_diff_viewer(original_content: List[Dict], enhanced_content: List[Dict]):
    """
    Render visual diff viewer in main app
    
    Usage:
    from modules.visual_diff_viewer import render_visual_diff_viewer
    
    render_visual_diff_viewer(original_data, enhanced_data)
    """
    
    diff_viewer = VisualDiffViewer()
    diff_viewer.render_diff_viewer_interface(original_content, enhanced_content)

# Compact diff summary for sidebar
def render_compact_diff_summary(original_content: List[Dict], enhanced_content: List[Dict]):
    """
    Render compact diff summary for sidebar
    
    Usage:
    with st.sidebar:
        render_compact_diff_summary(original_data, enhanced_data)
    """
    
    if not original_content or not enhanced_content:
        return
    
    st.markdown("### 🔍 Diff Summary")
    
    total_items = min(len(original_content), len(enhanced_content))
    approved_items = len(st.session_state.get('diff_approved_items', set()))
    
    # Quick metrics
    st.metric("Total Items", total_items)
    st.metric("Approved", f"{approved_items}/{total_items}")
    
    if total_items > 0:
        approval_rate = (approved_items / total_items) * 100
        st.progress(approval_rate / 100)
        st.caption(f"{approval_rate:.1f}% approved")
    
    if st.button("🔍 View Full Diff"):
        st.session_state['show_diff_viewer'] = True

if __name__ == "__main__":
    # Test the visual diff viewer
    st.set_page_config(page_title="Visual Diff Viewer Test", layout="wide")
    
    st.title("Visual Diff Viewer Test")
    
    # Sample data
    original_data = [
        {"content": "This is the original content that needs enhancement."},
        {"content": "Another piece of original text for testing."}
    ]
    
    enhanced_data = [
        {"content": "This is the significantly improved and enhanced content that provides much more value and clarity.", "quality_score": 0.85},
        {"content": "Another piece of original text for testing that has been refined.", "quality_score": 0.72}
    ]
    
    # Render diff viewer
    render_visual_diff_viewer(original_data, enhanced_data)

