#!/usr/bin/env python3
"""
Fine-Tune Data Refinement & Review System - Production NLP Enhanced
Advanced AI training data creation with intelligent theme discovery
"""

import os
import sys
import logging
import streamlit as st
from pathlib import Path

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set page config first
st.set_page_config(
    page_title="Fine-Tune Data System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Production environment setup
os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
os.environ.setdefault('LOW_MEM_MODE', 'true')

# Import modules with error handling
IMPORTS_SUCCESSFUL = True
import_errors = []

try:
    # Core modules
    from modules.input_validation import InputValidator, safe_env_bool, safe_env_str, safe_env_int
    from modules.render_optimization import initialize_render_optimizations, render_session_management, auto_save_current_session
    from modules.large_file_ocr_handler import LargeFileOCRHandler, OCRConfig, process_large_file_with_ui
    
    # NLP modules
    from modules.spacy_theme_discovery import SpacyThemeDiscovery, ThemeDiscoveryConfig, create_theme_discovery_ui, display_theme_matches, check_nlp_dependencies
    from modules.advanced_nlp_features import AdvancedNLPProcessor, create_nlp_analysis_ui, display_nlp_analysis_results, check_advanced_nlp_dependencies
    
    # Core processing modules
    from modules.enhanced_universal_extractor import EnhancedUniversalExtractor
    from modules.enhanced_tone_manager import EnhancedToneManager
    from modules.manual_review import ManualReviewInterface
    from modules.enhanced_zip_export import EnhancedZipExporter
    from modules.enhanced_sidebar_metrics import EnhancedSidebarMetrics
    from modules.enhanced_theming import EnhancedTheming
    from modules.enhanced_debugging import EnhancedDebugging
    from modules.async_enhancement_processor import AsyncEnhancementProcessor
    from modules.lean_session_manager import store_large_data, retrieve_data, get_memory_stats, render_memory_controls
    
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    import_errors.append(str(e))
    st.error(f"Import error: {e}")

class FineTuneDataSystemNLP:
    """Production NLP-Enhanced Fine-Tune Data System"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not IMPORTS_SUCCESSFUL:
            st.error("❌ System cannot start due to import errors")
            for error in import_errors:
                st.error(f"Import error: {error}")
            return
        
        # Initialize core components
        self.extractor = EnhancedUniversalExtractor()
        self.tone_manager = EnhancedToneManager()
        self.manual_review = ManualReviewInterface()
        self.zip_exporter = EnhancedZipExporter()
        self.theming = EnhancedTheming()
        self.debugging = EnhancedDebugging()
        self.async_processor = AsyncEnhancementProcessor()
        self.sidebar_metrics = EnhancedSidebarMetrics()
        
        # Initialize NLP components
        self.theme_discovery = SpacyThemeDiscovery()
        self.nlp_processor = AdvancedNLPProcessor()
        
        # Initialize session state
        self._initialize_session_state()
        
        # Initialize Render optimizations
        initialize_render_optimizations()
        
        self.logger.info("Fine-Tune Data System NLP Enhanced initialized")
    
    def _initialize_session_state(self):
        """Initialize session state for 6-step workflow"""
        essential_state = {
            'current_step': 1,
            'step_statuses': {i: False for i in range(1, 7)},
            'selected_theme': 'default',
            'debug_mode': safe_env_bool("DEBUG_MODE", False),
            'low_memory_mode': safe_env_bool("LOW_MEM_MODE", True),
            'total_files_processed': 0,
            'total_items_processed': 0,
            'session_initialized': True,
            'theme_discovery_completed': False,
            'nlp_analysis_completed': False,
            'discovered_themes': [],
            'approved_chunks': [],
            'nlp_insights': {}
        }
        
        for key, default_value in essential_state.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def run(self):
        """Main application entry point"""
        if not IMPORTS_SUCCESSFUL:
            st.error("Application cannot start due to import errors.")
            return
        
        # Apply theming
        self.theming.apply_theme()
        
        # Render header
        self._render_header()
        
        # Render memory controls in sidebar
        render_memory_controls()
        
        # Render debug controls if enabled
        if st.session_state.get('debug_mode', False):
            self.debugging.render_debug_panel()
        
        # Main navigation
        self._render_navigation()
        
        # Render current step
        current_step = st.session_state.get('current_step', 1)
        
        if current_step == 1:
            self._render_step_1_upload()
        elif current_step == 2:
            self._render_step_2_theme_discovery()
        elif current_step == 3:
            self._render_step_3_nlp_analysis()
        elif current_step == 4:
            self._render_step_4_enhancement()
        elif current_step == 5:
            self._render_step_5_review()
        elif current_step == 6:
            self._render_step_6_export()
        
        # Render sidebar metrics
        self.sidebar_metrics.render_metrics()
        
        # Auto-save session
        auto_save_current_session()
    
    def _render_header(self):
        """Render application header with system status"""
        st.title("🧠 Fine-Tune Data System - NLP Enhanced")
        st.markdown("*Production v5.0 - Theme Discovery • Advanced NLP • OCR • Memory Optimized*")
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            nlp_deps = check_nlp_dependencies()
            nlp_status = "✅" if any(nlp_deps.values()) else "⚠️"
            st.metric("NLP Support", nlp_status)
        
        with col2:
            memory_mode = "Low Memory" if st.session_state.get('low_memory_mode', False) else "Standard"
            st.metric("Memory Mode", memory_mode)
        
        with col3:
            theme_status = "✅ Complete" if st.session_state.get('theme_discovery_completed', False) else "Pending"
            st.metric("Theme Discovery", theme_status)
        
        with col4:
            debug_status = "🔍 ON" if st.session_state.get('debug_mode', False) else "OFF"
            st.metric("Debug Mode", debug_status)
    
    def _render_navigation(self):
        """Render step navigation with progress"""
        steps = [
            "📤 Upload Documents",
            "🔍 Theme Discovery", 
            "🧠 NLP Analysis",
            "✨ AI Enhancement",
            "📋 Manual Review",
            "📦 Export & Deploy"
        ]
        
        # Progress bar
        current_step = st.session_state.get('current_step', 1)
        progress = (current_step - 1) / (len(steps) - 1)
        st.progress(progress)
        
        # Step indicators
        cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                step_num = i + 1
                if step_num == current_step:
                    st.markdown(f"**🔄 {step}**")
                elif st.session_state['step_statuses'].get(step_num, False):
                    st.markdown(f"✅ {step}")
                else:
                    st.markdown(f"⏳ {step}")
    
    def _render_step_1_upload(self):
        """Step 1: Document Upload with OCR support"""
        st.header("📤 Step 1: Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents for processing",
            type=['pdf', 'docx', 'txt', 'md'],
            accept_multiple_files=True,
            help="Supports PDF (with OCR), Word documents, and text files"
        )
        
        if uploaded_files:
            all_extracted_content = []
            
            for uploaded_file in uploaded_files:
                with st.expander(f"Processing: {uploaded_file.name}", expanded=True):
                    try:
                        # Process file with OCR support
                        if uploaded_file.name.lower().endswith('.pdf'):
                            # Use large file OCR handler for PDFs
                            config = OCRConfig(
                                timeout_per_page=30,
                                max_total_timeout=1800,  # 30 minutes
                                memory_limit_mb=400,
                                languages=['eng']
                            )
                            
                            success, full_text, error_msg = process_large_file_with_ui(
                                uploaded_file, config
                            )
                            
                            if not success:
                                st.error(f"❌ Failed to extract content: {error_msg}")
                                continue
                        else:
                            # Standard text extraction
                            full_text = self.extractor.extract_content(uploaded_file)
                        
                        if full_text and len(full_text.strip()) > 50:
                            content_data = {
                                'filename': uploaded_file.name,
                                'content': full_text,
                                'char_count': len(full_text),
                                'extraction_method': 'OCR-enhanced' if uploaded_file.name.lower().endswith('.pdf') else 'standard'
                            }
                            
                            all_extracted_content.append(content_data)
                            
                            # Show preview
                            st.text_area(
                                "Content Preview",
                                full_text[:500] + "..." if len(full_text) > 500 else full_text,
                                height=100,
                                disabled=True
                            )
                            
                            st.success(f"✅ Extracted {len(full_text):,} characters")
                        else:
                            st.warning("⚠️ No content extracted or content too short")
                    
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        self.logger.error(f"File processing error: {e}")
            
            if all_extracted_content:
                store_large_data('extracted_content', all_extracted_content, force_disk=True)
                st.session_state['step_statuses'][1] = True
                st.session_state['total_files_processed'] = len(all_extracted_content)
                
                st.success(f"✅ Successfully processed {len(all_extracted_content)} files")
                
                if st.button("➡️ Proceed to Theme Discovery"):
                    st.session_state['current_step'] = 2
                    st.rerun()
    
    def _render_step_2_theme_discovery(self):
        """Step 2: Theme-Based Content Discovery"""
        st.header("🔍 Step 2: Theme-Based Content Discovery")
        
        extracted_content = retrieve_data('extracted_content', [])
        
        if not extracted_content:
            st.warning("No content found. Please upload documents first.")
            return
        
        # Check NLP dependencies
        nlp_status = check_nlp_dependencies()
        
        # Display NLP status
        with st.expander("🔧 NLP Dependencies Status", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                spacy_status = "✅ Available" if nlp_status.get('spacy', False) else "⚠️ Limited"
                st.metric("spaCy", spacy_status)
            
            with col2:
                nltk_status = "✅ Available" if nlp_status.get('nltk', False) else "⚠️ Limited"
                st.metric("NLTK", nltk_status)
            
            with col3:
                st_status = "✅ Available" if nlp_status.get('sentence_transformers', False) else "⚠️ Limited"
                st.metric("Sentence Transformers", st_status)
        
        # Combine content for theme discovery
        combined_content = "\n\n".join([
            f"=== {file_data['filename']} ===\n{file_data['content']}"
            for file_data in extracted_content
        ])
        
        st.write(f"**Total content:** {len(combined_content):,} characters from {len(extracted_content)} files")
        
        # Theme discovery UI
        themes, config = create_theme_discovery_ui()
        
        if themes and st.button("🔍 Discover Thematic Content"):
            with st.spinner("Analyzing content for thematic matches..."):
                
                # Perform theme discovery
                if config.similarity_threshold < 0.8:
                    matches = self.theme_discovery.enhance_themes_with_similarity(
                        combined_content, themes, config.similarity_threshold
                    )
                else:
                    matches = self.theme_discovery.discover_themes(combined_content, themes)
                
                # Store and display results
                if matches:
                    store_large_data('theme_matches', matches, force_disk=True)
                    
                    approved_chunks = display_theme_matches(matches, combined_content)
                    
                    if approved_chunks:
                        store_large_data('approved_chunks', approved_chunks, force_disk=True)
                        st.session_state['theme_discovery_completed'] = True
                        st.session_state['step_statuses'][2] = True
                        
                        st.success(f"✅ {len(approved_chunks)} chunks approved!")
                        
                        if st.button("➡️ Proceed to NLP Analysis"):
                            st.session_state['current_step'] = 3
                            st.rerun()
                else:
                    st.warning("No thematic matches found. Try different keywords.")
        
        # Show previous results or skip option
        elif st.session_state.get('theme_discovery_completed', False):
            st.success("✅ Theme discovery completed")
            if st.button("➡️ Proceed to NLP Analysis"):
                st.session_state['current_step'] = 3
                st.rerun()
        
        if st.button("⏭️ Skip Theme Discovery"):
            st.session_state['step_statuses'][2] = True
            st.session_state['current_step'] = 3
            st.rerun()
    
    def _render_step_3_nlp_analysis(self):
        """Step 3: Advanced NLP Analysis"""
        st.header("🧠 Step 3: Advanced NLP Analysis")
        
        # Get content for analysis
        approved_chunks = retrieve_data('approved_chunks', [])
        extracted_content = retrieve_data('extracted_content', [])
        
        if not extracted_content:
            st.warning("No content found. Please upload documents first.")
            return
        
        # Choose content source
        if approved_chunks:
            st.info(f"🎯 Analyzing {len(approved_chunks)} thematically relevant chunks")
            analysis_content = "\n\n".join([chunk.full_chunk for chunk in approved_chunks])
        else:
            st.info("📄 Analyzing full content")
            analysis_content = "\n\n".join([file_data['content'] for file_data in extracted_content])
        
        # NLP analysis UI
        analysis_options = create_nlp_analysis_ui()
        
        if st.button("🧠 Run NLP Analysis"):
            with st.spinner("Performing advanced NLP analysis..."):
                
                # Run comprehensive analysis
                analysis_result = self.nlp_processor.analyze_content_comprehensively(analysis_content)
                
                # Store results
                store_large_data('nlp_analysis_result', analysis_result, force_disk=True)
                st.session_state['nlp_analysis_completed'] = True
                st.session_state['step_statuses'][3] = True
                
                # Display results
                display_nlp_analysis_results(analysis_result, analysis_options)
                
                # Generate content suggestions
                if analysis_options.get('include_suggestions', True):
                    suggestions = self.nlp_processor.generate_content_suggestions(analysis_content)
                    
                    if suggestions:
                        st.subheader("💡 Content Improvement Suggestions")
                        for suggestion in suggestions:
                            priority_color = {
                                'high': '🔴',
                                'medium': '🟡', 
                                'low': '🟢'
                            }.get(suggestion['priority'], '⚪')
                            
                            st.write(f"{priority_color} **{suggestion['type'].title()}:** {suggestion['suggestion']}")
                            if suggestion.get('details'):
                                st.caption(suggestion['details'])
                
                st.success("✅ NLP analysis completed!")
                
                if st.button("➡️ Proceed to AI Enhancement"):
                    st.session_state['current_step'] = 4
                    st.rerun()
        
        # Show previous results
        elif st.session_state.get('nlp_analysis_completed', False):
            st.success("✅ NLP analysis completed")
            if st.button("➡️ Proceed to AI Enhancement"):
                st.session_state['current_step'] = 4
                st.rerun()
    
    def _render_step_4_enhancement(self):
        """Step 4: AI Enhancement"""
        st.header("✨ Step 4: AI Enhancement")
        
        # Get content for enhancement
        approved_chunks = retrieve_data('approved_chunks', [])
        extracted_content = retrieve_data('extracted_content', [])
        
        if not extracted_content:
            st.warning("No content found. Please complete previous steps first.")
            return
        
        # Prepare content items
        if approved_chunks:
            content_items = []
            for chunk in approved_chunks:
                content_items.append({
                    'type': 'thematic_chunk',
                    'content': chunk.full_chunk,
                    'theme': chunk.keyword,
                    'confidence': chunk.confidence
                })
        else:
            # Fallback to extracted content
            content_items = []
            for file_data in extracted_content:
                # Simple chunking for enhancement
                content = file_data['content']
                chunks = [content[i:i+2000] for i in range(0, len(content), 1500)]
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 100:
                        content_items.append({
                            'type': 'content_chunk',
                            'content': chunk,
                            'source': file_data['filename'],
                            'chunk_index': i
                        })
        
        if not content_items:
            st.warning("No content items prepared for enhancement.")
            return
        
        st.write(f"**Ready to enhance:** {len(content_items)} content items")
        
        # Enhancement options
        col1, col2 = st.columns(2)
        
        with col1:
            output_type = st.selectbox(
                "Output Type:",
                ["Question-Answer", "Summary", "Thematic Insight", "Dialogue", "Instruction"]
            )
        
        with col2:
            tone_options = self.tone_manager.get_available_tones()
            selected_tone = st.selectbox("Tone:", tone_options)
        
        if st.button("✨ Enhance Content"):
            with st.spinner("Enhancing content with AI..."):
                
                try:
                    # Process with async enhancement
                    enhanced_items = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, item in enumerate(content_items):
                        status_text.text(f"Enhancing item {i+1}/{len(content_items)}")
                        
                        # Get tone prompt
                        tone_prompt = self.tone_manager.get_tone_prompt(selected_tone)
                        
                        # Enhance content
                        enhanced_content = self.async_processor.enhance_content_item(
                            item, output_type, tone_prompt
                        )
                        
                        if enhanced_content:
                            enhanced_items.append(enhanced_content)
                        
                        progress_bar.progress((i + 1) / len(content_items))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if enhanced_items:
                        # Store enhanced content
                        store_large_data('enhanced_content', enhanced_items, force_disk=True)
                        st.session_state['step_statuses'][4] = True
                        
                        st.success(f"✅ Enhanced {len(enhanced_items)} items successfully!")
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Items Enhanced", len(enhanced_items))
                        with col2:
                            avg_length = sum(len(item.get('enhanced_content', '')) for item in enhanced_items) / len(enhanced_items)
                            st.metric("Avg Length", f"{avg_length:.0f} chars")
                        with col3:
                            st.metric("Tone Applied", selected_tone.replace('_', ' ').title())
                        
                        if st.button("➡️ Proceed to Manual Review"):
                            st.session_state['current_step'] = 5
                            st.rerun()
                    else:
                        st.error("Enhancement failed. Please try again.")
                
                except Exception as e:
                    st.error(f"Enhancement error: {e}")
                    self.logger.error(f"Enhancement error: {e}")
    
    def _render_step_5_review(self):
        """Step 5: Manual Review"""
        st.header("📋 Step 5: Manual Review")
        
        enhanced_content = retrieve_data('enhanced_content', [])
        
        if not enhanced_content:
            st.warning("No enhanced content found. Please complete AI enhancement first.")
            return
        
        # Render manual review interface
        reviewed_content = self.manual_review.render_review_interface(enhanced_content)
        
        if reviewed_content:
            store_large_data('reviewed_content', reviewed_content, force_disk=True)
            st.session_state['step_statuses'][5] = True
            
            if st.button("➡️ Proceed to Export"):
                st.session_state['current_step'] = 6
                st.rerun()
    
    def _render_step_6_export(self):
        """Step 6: Export & Deploy"""
        st.header("📦 Step 6: Export & Deploy")
        
        reviewed_content = retrieve_data('reviewed_content', [])
        
        if not reviewed_content:
            st.warning("No reviewed content found. Please complete manual review first.")
            return
        
        # Export options
        st.subheader("📁 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format:",
                ["JSONL (Training)", "JSON (Structured)", "TXT (Plain Text)", "ZIP (Complete)"]
            )
        
        with col2:
            include_metadata = st.checkbox("Include metadata", True)
        
        if st.button("📦 Generate Export"):
            with st.spinner("Preparing export..."):
                
                try:
                    # Generate export
                    export_data = self.zip_exporter.create_comprehensive_export(
                        reviewed_content,
                        format_type=export_format.split()[0].lower(),
                        include_metadata=include_metadata
                    )
                    
                    if export_data:
                        st.success("✅ Export generated successfully!")
                        
                        # Provide download
                        if export_format.startswith("ZIP"):
                            st.download_button(
                                label="📥 Download Complete Package",
                                data=export_data,
                                file_name="fine_tune_data_export.zip",
                                mime="application/zip"
                            )
                        else:
                            st.download_button(
                                label=f"📥 Download {export_format}",
                                data=export_data,
                                file_name=f"fine_tune_data.{export_format.split()[0].lower()}",
                                mime="application/octet-stream"
                            )
                        
                        st.session_state['step_statuses'][6] = True
                        st.balloons()
                        
                        st.success("🎉 Process completed successfully!")
                    else:
                        st.error("Export generation failed.")
                
                except Exception as e:
                    st.error(f"Export error: {e}")
                    self.logger.error(f"Export error: {e}")

# Health check endpoint
@st.cache_data
def health_check():
    """Health check for Render deployment"""
    return {
        "status": "healthy",
        "imports": IMPORTS_SUCCESSFUL,
        "nlp_available": any(check_nlp_dependencies().values()) if IMPORTS_SUCCESSFUL else False
    }

# Main application
def main():
    """Main application entry point"""
    
    # Handle health check
    if st.query_params.get("healthcheck"):
        st.json(health_check())
        return
    
    # Initialize and run app
    try:
        app = FineTuneDataSystemNLP()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logging.error(f"Application error: {e}")

if __name__ == "__main__":
    main()

