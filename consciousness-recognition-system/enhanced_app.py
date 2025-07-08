"""
Enhanced Universal AI Training Data Creator
==========================================

Comprehensive application with all 5 core enhancements and optional add-ons:

Core Enhancements:
1. Manual Review Before Export
2. Dynamic Prompt Templates Per Tone  
3. Smart Q&A vs Monologue Detection
4. Raw vs Enhanced Comparison Viewer
5. Sidebar Metrics Dashboard

Optional Add-ons:
- Enhanced Theming System
- Enhanced ZIP Export
- Enhanced Hugging Face Upload

Features:
- Universal content extraction (PDF, TXT, DOCX)
- GPT-powered content enhancement
- Quality scoring and validation
- Multiple export formats
- Comprehensive analytics
- Professional UI/UX
"""

import streamlit as st
import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import sys

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import enhanced modules
try:
    from manual_review import ManualReviewInterface
    from dynamic_prompt_engine import DynamicPromptEngine
    from smart_content_detector import SmartContentDetector
    from enhanced_comparison_viewer import EnhancedComparisonViewer
    from enhanced_sidebar_metrics import EnhancedSidebarMetrics
    from enhanced_theming import EnhancedTheming
    from enhanced_zip_export import EnhancedZipExporter
    from enhanced_huggingface_upload import EnhancedHuggingFaceUploader
    from enhanced_universal_extractor import EnhancedUniversalExtractor
    from enhanced_custom_prompt_engine import EnhancedCustomPromptEngine
except ImportError as e:
    st.error(f"❌ **Module Import Error:** {e}")
    st.info("📝 Please ensure all enhanced modules are in the modules/ directory")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enhanced Universal AI Training Data Creator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnhancedUniversalAITrainer:
    """Enhanced Universal AI Training Data Creator - Complete System"""
    
    def __init__(self):
        # Initialize all enhanced components
        self.manual_review = ManualReviewInterface()
        self.prompt_engine = DynamicPromptEngine()
        self.content_detector = SmartContentDetector()
        self.comparison_viewer = EnhancedComparisonViewer()
        self.sidebar_metrics = EnhancedSidebarMetrics()
        self.theming = EnhancedTheming()
        self.zip_exporter = EnhancedZipExporter()
        self.hf_uploader = EnhancedHuggingFaceUploader()
        self.extractor = EnhancedUniversalExtractor()
        self.custom_prompt_engine = EnhancedCustomPromptEngine()
        
        # Initialize session state
        self._initialize_session_state()
        
        # Apply theming
        self.theming.apply_theme()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        
        defaults = {
            'uploaded_file': None,
            'extracted_content': [],
            'enhanced_content': [],
            'final_content': [],
            'processing_stats': {},
            'current_step': 1,
            'processing_complete': False,
            'enhancement_complete': False,
            'review_complete': False,
            'export_ready': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Main application entry point"""
        
        # Render theming controls
        self.theming.render_theme_selector()
        
        # Render sidebar metrics
        self.sidebar_metrics.render_sidebar(
            uploaded_file=st.session_state.uploaded_file,
            extracted_content=st.session_state.extracted_content,
            enhanced_content=st.session_state.enhanced_content,
            export_content=st.session_state.final_content,
            processing_stats=st.session_state.processing_stats
        )
        
        # Main content area
        self._render_main_content()
    
    def _render_main_content(self):
        """Render main content area with enhanced workflow"""
        
        # Header
        st.markdown("""
        # 🧠 **Enhanced Universal AI Training Data Creator**
        
        ### Transform any content into high-quality AI training data with advanced enhancements
        """)
        
        # Progress indicator
        self._render_progress_indicator()
        
        # Main workflow tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 **Upload & Extract**",
            "🔍 **Content Analysis**", 
            "✨ **Enhancement**",
            "📋 **Review & Validate**",
            "📤 **Export & Share**"
        ])
        
        with tab1:
            self._render_upload_extract_tab()
        
        with tab2:
            self._render_analysis_tab()
        
        with tab3:
            self._render_enhancement_tab()
        
        with tab4:
            self._render_review_tab()
        
        with tab5:
            self._render_export_tab()
    
    def _render_progress_indicator(self):
        """Render enhanced progress indicator"""
        
        steps = [
            ("📁", "Upload", st.session_state.uploaded_file is not None),
            ("🔍", "Extract", len(st.session_state.extracted_content) > 0),
            ("✨", "Enhance", st.session_state.enhancement_complete),
            ("📋", "Review", st.session_state.review_complete),
            ("📤", "Export", st.session_state.export_ready)
        ]
        
        cols = st.columns(len(steps))
        
        for i, (icon, label, completed) in enumerate(steps):
            with cols[i]:
                if completed:
                    st.success(f"{icon} **{label}** ✅")
                elif i == st.session_state.current_step - 1:
                    st.info(f"{icon} **{label}** 🔄")
                else:
                    st.write(f"{icon} **{label}**")
        
        st.markdown("---")
    
    def _render_upload_extract_tab(self):
        """Render upload and extraction tab - Core Enhancement 3 integration"""
        
        st.markdown("## 📁 **Step 1: Upload & Extract Content**")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📎 **Choose your content file**",
            type=['pdf', 'txt', 'docx', 'md'],
            help="Upload PDF, TXT, DOCX, or Markdown files containing spiritual/consciousness content"
        )
        
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            
            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 File Name", uploaded_file.name)
            with col2:
                file_size = uploaded_file.size / 1024 / 1024  # MB
                st.metric("💾 File Size", f"{file_size:.2f} MB")
            with col3:
                file_type = uploaded_file.name.split('.')[-1].upper()
                st.metric("📋 File Type", file_type)
            
            # Extract content button
            if st.button("🚀 **Extract Content**", type="primary"):
                with st.spinner("🔍 Analyzing and extracting content..."):
                    try:
                        # Use enhanced extractor with smart detection
                        extracted_content = self.extractor.extract_content(uploaded_file)
                        
                        # Apply smart content detection - Core Enhancement 3
                        detected_content = self.content_detector.detect_and_process_content(extracted_content)
                        
                        st.session_state.extracted_content = detected_content
                        st.session_state.current_step = 2
                        st.session_state.processing_complete = True
                        
                        st.success(f"✅ **Extraction complete!** Found {len(detected_content)} examples")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ **Extraction failed:** {str(e)}")
                        logger.error(f"Content extraction error: {e}")
        
        # Show extracted content preview
        if st.session_state.extracted_content:
            st.markdown("### 📊 **Extraction Results**")
            
            # Content type analysis - Core Enhancement 3
            content_analysis = self.content_detector.analyze_content_types(st.session_state.extracted_content)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📝 Total Examples", len(st.session_state.extracted_content))
            with col2:
                st.metric("💬 Q&A Pairs", content_analysis.get('qa_pairs', 0))
            with col3:
                st.metric("📖 Passages", content_analysis.get('passages', 0))
            with col4:
                st.metric("🔍 Content Type", content_analysis.get('primary_type', 'Mixed').title())
            
            # Preview first few examples
            with st.expander("👀 **Preview Extracted Content**", expanded=False):
                for i, item in enumerate(st.session_state.extracted_content[:3]):
                    st.markdown(f"**Example {i+1}:**")
                    st.markdown(f"**Q:** {item.get('question', 'N/A')}")
                    st.markdown(f"**A:** {item.get('answer', 'N/A')[:200]}...")
                    if item.get('content_type'):
                        st.caption(f"Type: {item['content_type']}")
                    st.markdown("---")
    
    def _render_analysis_tab(self):
        """Render content analysis tab"""
        
        st.markdown("## 🔍 **Step 2: Content Analysis**")
        
        if not st.session_state.extracted_content:
            st.info("📝 Please extract content first in the Upload & Extract tab.")
            return
        
        # Content statistics
        st.markdown("### 📊 **Content Statistics**")
        
        content = st.session_state.extracted_content
        total_words = sum(len((item.get('question', '') + ' ' + item.get('answer', '')).split()) for item in content)
        avg_words = total_words / len(content) if content else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Examples", len(content))
        with col2:
            st.metric("📊 Total Words", f"{total_words:,}")
        with col3:
            st.metric("📈 Avg Words/Example", f"{avg_words:.1f}")
        with col4:
            estimated_cost = len(content) * 0.002  # Rough estimate
            st.metric("💰 Est. Enhancement Cost", f"${estimated_cost:.3f}")
        
        # Content type breakdown - Core Enhancement 3
        st.markdown("### 🎯 **Content Type Analysis**")
        
        type_analysis = self.content_detector.analyze_content_types(content)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📋 Content Distribution:**")
            for content_type, count in type_analysis.get('type_distribution', {}).items():
                percentage = (count / len(content)) * 100
                st.write(f"• {content_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        with col2:
            st.markdown("**🎯 Quality Indicators:**")
            quality_indicators = type_analysis.get('quality_indicators', {})
            for indicator, value in quality_indicators.items():
                st.write(f"• {indicator.replace('_', ' ').title()}: {value}")
        
        # Sample content by type
        st.markdown("### 📖 **Content Samples by Type**")
        
        content_types = set(item.get('content_type', 'unknown') for item in content)
        
        for content_type in content_types:
            type_items = [item for item in content if item.get('content_type') == content_type]
            
            with st.expander(f"📋 **{content_type.replace('_', ' ').title()}** ({len(type_items)} items)", expanded=False):
                for i, item in enumerate(type_items[:2]):  # Show 2 examples per type
                    st.markdown(f"**Example {i+1}:**")
                    st.markdown(f"**Q:** {item.get('question', 'N/A')}")
                    st.markdown(f"**A:** {item.get('answer', 'N/A')[:300]}...")
                    st.markdown("---")
    
    def _render_enhancement_tab(self):
        """Render enhancement tab - Core Enhancement 2 integration"""
        
        st.markdown("## ✨ **Step 3: Content Enhancement**")
        
        if not st.session_state.extracted_content:
            st.info("📝 Please extract content first in the Upload & Extract tab.")
            return
        
        # Enhancement configuration - Core Enhancement 2
        st.markdown("### ⚙️ **Enhancement Configuration**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dynamic prompt selection - Core Enhancement 2
            available_tones = self.prompt_engine.get_available_tones()
            selected_tone = st.selectbox(
                "🎭 **Enhancement Tone**",
                options=available_tones,
                index=0,
                help="Choose the spiritual tone for content enhancement"
            )
            
            enhancement_strength = st.slider(
                "💪 **Enhancement Strength**",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="How much to enhance the content (0.1 = light, 1.0 = heavy)"
            )
        
        with col2:
            quality_threshold = st.slider(
                "🎯 **Quality Threshold**",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Minimum quality score for enhanced content"
            )
            
            max_items_to_enhance = st.number_input(
                "📊 **Max Items to Enhance**",
                min_value=1,
                max_value=len(st.session_state.extracted_content),
                value=min(50, len(st.session_state.extracted_content)),
                help="Limit enhancement for cost control"
            )
        
        # Show selected prompt preview - Core Enhancement 2
        with st.expander("👀 **Preview Selected Enhancement Tone**", expanded=False):
            prompt_preview = self.prompt_engine.get_prompt_preview(selected_tone)
            st.markdown(prompt_preview)
        
        # Enhancement button
        if st.button("🚀 **Start Enhancement Process**", type="primary"):
            with st.spinner("✨ Enhancing content with GPT..."):
                try:
                    start_time = time.time()
                    
                    # Load dynamic prompt - Core Enhancement 2
                    enhancement_prompt = self.prompt_engine.load_prompt_template(selected_tone)
                    
                    # Enhance content
                    enhanced_content = self.custom_prompt_engine.enhance_content_batch(
                        content=st.session_state.extracted_content[:max_items_to_enhance],
                        enhancement_prompt=enhancement_prompt,
                        quality_threshold=quality_threshold,
                        enhancement_strength=enhancement_strength
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Update session state
                    st.session_state.enhanced_content = enhanced_content
                    st.session_state.enhancement_complete = True
                    st.session_state.current_step = 4
                    st.session_state.processing_stats.update({
                        'enhancement_time': processing_time,
                        'enhanced_count': len([item for item in enhanced_content if item.get('enhanced', False)]),
                        'enhancement_tone': selected_tone
                    })
                    
                    st.success(f"✅ **Enhancement complete!** Enhanced {len([item for item in enhanced_content if item.get('enhanced', False)])} items in {processing_time:.1f}s")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ **Enhancement failed:** {str(e)}")
                    logger.error(f"Content enhancement error: {e}")
        
        # Show enhancement results
        if st.session_state.enhanced_content:
            st.markdown("### 📊 **Enhancement Results**")
            
            enhanced_items = [item for item in st.session_state.enhanced_content if item.get('enhanced', False)]
            total_cost = sum(item.get('enhancement_cost', 0) for item in enhanced_items)
            avg_quality = sum(item.get('quality_score', 0) for item in enhanced_items) / len(enhanced_items) if enhanced_items else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✨ Enhanced Items", len(enhanced_items))
            with col2:
                st.metric("💰 Total Cost", f"${total_cost:.4f}")
            with col3:
                st.metric("📊 Avg Quality", f"{avg_quality:.2f}")
            with col4:
                enhancement_rate = (len(enhanced_items) / len(st.session_state.extracted_content)) * 100
                st.metric("📈 Enhancement Rate", f"{enhancement_rate:.1f}%")
            
            # Raw vs Enhanced Comparison - Core Enhancement 4
            self.comparison_viewer.render_comparison_toggle()
            if self.comparison_viewer.comparison_enabled:
                self.comparison_viewer.render_comparison_view(st.session_state.enhanced_content)
                self.comparison_viewer.render_comparison_summary(st.session_state.enhanced_content)
    
    def _render_review_tab(self):
        """Render review tab - Core Enhancement 1 integration"""
        
        st.markdown("## 📋 **Step 4: Review & Validate**")
        
        if not st.session_state.enhanced_content:
            st.info("📝 Please enhance content first in the Enhancement tab.")
            return
        
        # Manual Review System - Core Enhancement 1
        st.markdown("### 🔍 **Manual Review Process**")
        
        review_results = self.manual_review.render_review_interface(
            st.session_state.enhanced_content
        )
        
        if review_results:
            # Update final content based on review
            st.session_state.final_content = review_results['approved_content']
            st.session_state.review_complete = True
            st.session_state.export_ready = True
            st.session_state.current_step = 5
            
            # Show review summary
            st.markdown("### 📊 **Review Summary**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ Approved", review_results['approved_count'])
            with col2:
                st.metric("❌ Rejected", review_results['rejected_count'])
            with col3:
                st.metric("✏️ Edited", review_results['edited_count'])
            with col4:
                approval_rate = (review_results['approved_count'] / max(review_results['total_reviewed'], 1)) * 100
                st.metric("📈 Approval Rate", f"{approval_rate:.1f}%")
        
        # Quality validation
        if st.session_state.final_content:
            st.markdown("### 🎯 **Quality Validation**")
            
            quality_scores = [item.get('quality_score', 0) for item in st.session_state.final_content]
            
            col1, col2 = st.columns(2)
            with col1:
                excellent = sum(1 for score in quality_scores if score >= 0.8)
                good = sum(1 for score in quality_scores if 0.6 <= score < 0.8)
                st.metric("🟢 High Quality (≥0.6)", excellent + good)
            
            with col2:
                fair = sum(1 for score in quality_scores if 0.4 <= score < 0.6)
                poor = sum(1 for score in quality_scores if score < 0.4)
                st.metric("🟡 Needs Review (<0.6)", fair + poor)
            
            if poor > 0:
                st.warning(f"⚠️ {poor} items have quality scores below 0.4 and may need additional review.")
    
    def _render_export_tab(self):
        """Render export tab with all optional add-ons"""
        
        st.markdown("## 📤 **Step 5: Export & Share**")
        
        if not st.session_state.final_content:
            st.info("📝 Please complete the review process first.")
            return
        
        # Export options tabs
        export_tab1, export_tab2, export_tab3 = st.tabs([
            "📋 **Standard Export**",
            "📦 **Enhanced ZIP Export**", 
            "🤗 **Hugging Face Upload**"
        ])
        
        with export_tab1:
            self._render_standard_export()
        
        with export_tab2:
            # Enhanced ZIP Export - Optional Add-on
            self.zip_exporter.render_export_ui(
                raw_content=st.session_state.extracted_content,
                enhanced_content=st.session_state.enhanced_content,
                final_content=st.session_state.final_content,
                session_stats=st.session_state.processing_stats
            )
        
        with export_tab3:
            # Enhanced Hugging Face Upload - Optional Add-on
            self.hf_uploader.render_upload_ui(st.session_state.final_content)
    
    def _render_standard_export(self):
        """Render standard export options"""
        
        st.markdown("### 📋 **Standard Export Options**")
        
        final_content = st.session_state.final_content
        
        # Export format selection
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "📄 **Export Format**",
                options=['JSON', 'JSONL', 'CSV'],
                index=1,  # Default to JSONL
                help="Choose the export format for your training data"
            )
        
        with col2:
            include_metadata = st.checkbox(
                "📊 **Include Metadata**",
                value=True,
                help="Include quality scores and enhancement information"
            )
        
        # Prepare export data
        export_data = []
        for item in final_content:
            export_item = {
                'question': item.get('question', ''),
                'answer': item.get('answer', '')
            }
            
            if include_metadata:
                export_item.update({
                    'quality_score': item.get('quality_score', 0),
                    'enhanced': item.get('enhanced', False),
                    'enhancement_tone': item.get('enhancement_tone', 'none')
                })
            
            export_data.append(export_item)
        
        # Generate export content
        if export_format == 'JSON':
            export_content = json.dumps(export_data, indent=2, ensure_ascii=False)
            mime_type = "application/json"
            file_extension = "json"
        elif export_format == 'JSONL':
            export_content = '\n'.join(json.dumps(item, ensure_ascii=False) for item in export_data)
            mime_type = "application/jsonl"
            file_extension = "jsonl"
        else:  # CSV
            import csv
            from io import StringIO
            
            csv_buffer = StringIO()
            if export_data:
                fieldnames = export_data[0].keys()
                writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_data)
            export_content = csv_buffer.getvalue()
            mime_type = "text/csv"
            file_extension = "csv"
        
        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_training_data_{timestamp}.{file_extension}"
        
        st.download_button(
            label=f"💾 **Download {export_format} File**",
            data=export_content,
            file_name=filename,
            mime=mime_type,
            help=f"Download your training data as {export_format} format"
        )
        
        # Export summary
        st.markdown("### 📊 **Export Summary**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Total Items", len(export_data))
        with col2:
            total_words = sum(len((item['question'] + ' ' + item['answer']).split()) for item in export_data)
            st.metric("📊 Total Words", f"{total_words:,}")
        with col3:
            st.metric("📄 Format", export_format)
        
        st.success(f"✅ **Ready to download:** {filename}")


def main():
    """Main application entry point"""
    
    try:
        # Initialize and run the enhanced app
        app = EnhancedUniversalAITrainer()
        app.run()
        
    except Exception as e:
        st.error(f"❌ **Application Error:** {str(e)}")
        logger.error(f"Application error: {e}")
        
        # Show error details in expander
        with st.expander("🔍 **Error Details**", expanded=False):
            st.code(str(e))
            st.info("💡 Please check that all required modules are installed and accessible.")


if __name__ == "__main__":
    main()

