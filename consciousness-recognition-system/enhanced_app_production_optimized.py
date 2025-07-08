"""
Enhanced Universal AI Training Data Creator - Production Optimized
================================================================

Production-optimized version with:
- OCR support for image-only PDFs
- Async processing for large batches
- Memory-efficient session management
- Advanced monitoring and error handling

Version: 3.0 (Production Optimized)
"""

import os
import sys
import logging
import streamlit as st
from pathlib import Path

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import all modules
try:
    # Core modules
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
    
    # Production optimization modules
    from ocr_pdf_processor import OCRPDFProcessor, check_ocr_dependencies
    from async_enhancement_processor import AsyncEnhancementProcessor, StreamlitAsyncProcessor
    from lean_session_manager import LeanSessionManager, render_memory_controls, store_large_data, retrieve_data
    
    # Enhanced modules
    from enhanced_caching import EnhancedCaching
    from enhanced_debugging import EnhancedDebugging
    from enhanced_logging import EnhancedLogging
    from testing_support import TestingSupportModule
    from quality_control_enhanced import EnhancedQualityControl
    from ui_polish_enhanced import EnhancedUIPolish
    from metadata_schema_validator import MetadataSchemaValidator
    from export_confirmations import ExportConfirmations
    
    IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    st.error(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


class EnhancedUniversalAITrainingDataCreatorOptimized:
    """Production-optimized Fine-Tune Data Refinement & Review System"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not IMPORTS_SUCCESSFUL:
            st.error("Failed to import required modules. Please check dependencies.")
            return
        
        # Initialize core components
        self.manual_review = ManualReviewInterface()
        self.prompt_engine = DynamicPromptEngine()
        self.content_detector = SmartContentDetector()
        self.comparison_viewer = EnhancedComparisonViewer()
        self.sidebar_metrics = EnhancedSidebarMetrics()
        self.theming = EnhancedTheming()
        self.zip_exporter = EnhancedZipExporter()
        self.hf_uploader = EnhancedHuggingFaceUploader()
        self.universal_extractor = EnhancedUniversalExtractor()
        self.custom_prompt_engine = EnhancedCustomPromptEngine()
        
        # Initialize production optimization components
        self.ocr_processor = OCRPDFProcessor()
        self.async_processor = StreamlitAsyncProcessor()
        self.session_manager = LeanSessionManager()
        
        # Initialize enhanced components
        self.caching = EnhancedCaching()
        self.debugging = EnhancedDebugging()
        self.enhanced_logging = EnhancedLogging()
        self.testing_support = TestingSupportModule()
        self.quality_control = EnhancedQualityControl()
        self.ui_polish = EnhancedUIPolish()
        self.schema_validator = MetadataSchemaValidator()
        self.export_confirmations = ExportConfirmations()
        
        # Initialize session state
        self._initialize_session_state()
        
        self.logger.info("Enhanced Universal AI Training Data Creator (Optimized) initialized")
    
    def _initialize_session_state(self):
        """Initialize session state with memory optimization"""
        
        # Essential state only - large objects will be stored on disk
        essential_state = {
            'current_step': 1,
            'step_statuses': {i: False for i in range(1, 6)},
            'selected_theme': 'default',
            'debug_mode': False,
            'low_memory_mode': os.getenv("LOW_MEM_MODE", "1") == "1",
            'total_files_processed': 0,
            'total_items_processed': 0,
            'session_initialized': True
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
        self.theming.apply_theme(st.session_state.get('selected_theme', 'default'))
        
        # Render header with system status
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
            self._render_step_2_analysis()
        elif current_step == 3:
            self._render_step_3_enhancement()
        elif current_step == 4:
            self._render_step_4_review()
        elif current_step == 5:
            self._render_step_5_export()
        
        # Render sidebar metrics
        self.sidebar_metrics.render_metrics()
    
    def _render_header(self):
        """Render application header with system status"""
        
        st.title("🧠 Fine-Tune Data Refinement & Review System")
        st.markdown("*Production Optimized v3.0 - OCR • Async • Memory Efficient*")
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ocr_deps = check_ocr_dependencies()
            ocr_status = "✅" if ocr_deps.get('pytesseract', False) else "⚠️"
            st.metric("OCR Support", ocr_status)
        
        with col2:
            memory_mode = "Low Memory" if st.session_state.get('low_memory_mode', False) else "Standard"
            st.metric("Memory Mode", memory_mode)
        
        with col3:
            async_status = "✅ Available" if hasattr(self.async_processor, 'processor') else "⚠️ Limited"
            st.metric("Async Processing", async_status)
        
        with col4:
            debug_status = "🔍 ON" if st.session_state.get('debug_mode', False) else "OFF"
            st.metric("Debug Mode", debug_status)
    
    def _render_navigation(self):
        """Render step navigation with progress"""
        
        steps = [
            "📤 Upload Documents",
            "🔍 Content Analysis", 
            "✨ AI Enhancement",
            "📋 Manual Review",
            "📦 Export & Deploy"
        ]
        
        # Progress bar
        current_step = st.session_state.get('current_step', 1)
        progress = (current_step - 1) / (len(steps) - 1)
        st.progress(progress)
        
        # Step navigation
        cols = st.columns(len(steps))
        for i, (col, step_name) in enumerate(zip(cols, steps), 1):
            with col:
                if st.button(step_name, key=f"nav_step_{i}"):
                    st.session_state['current_step'] = i
                    st.rerun()
    
    def _render_step_1_upload(self):
        """Step 1: Document Upload with OCR support"""
        
        st.header("📤 Step 1: Upload Documents")
        
        # File upload with size validation
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX, MD)",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            help="Supports text extraction and OCR for image-only PDFs"
        )
        
        if uploaded_files:
            # Process files with OCR support
            all_extracted_content = []
            
            for uploaded_file in uploaded_files:
                with st.expander(f"Processing: {uploaded_file.name}", expanded=True):
                    
                    # File size check
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    st.write(f"**File size:** {file_size_mb:.1f} MB")
                    
                    if file_size_mb > 100:
                        st.warning("⚠️ Large file detected. Processing may take longer.")
                    
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        if uploaded_file.name.lower().endswith('.pdf'):
                            # Use OCR processor for PDFs
                            stats = self.ocr_processor.get_processing_stats(temp_path)
                            
                            st.write(f"**Processing mode:** {stats.get('recommended_mode', 'standard')}")
                            st.write(f"**Estimated time:** {stats.get('estimated_processing_time', 'Unknown')}")
                            
                            if stats.get('warnings'):
                                for warning in stats['warnings']:
                                    st.warning(f"⚠️ {warning}")
                            
                            # Process PDF with progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            extracted_pages = []
                            total_chars = 0
                            
                            for chunk_data in self.ocr_processor.process_pdf_streaming(temp_path, file_size_mb):
                                if "error" in chunk_data:
                                    st.error(f"Processing error: {chunk_data['error']}")
                                    break
                                
                                # Update progress
                                progress = chunk_data['chunk_end'] / chunk_data['total_pages']
                                progress_bar.progress(progress)
                                status_text.text(f"Processing pages {chunk_data['chunk_start']+1}-{chunk_data['chunk_end']} of {chunk_data['total_pages']}")
                                
                                # Collect pages
                                extracted_pages.extend(chunk_data['pages'])
                                total_chars += sum(page['char_count'] for page in chunk_data['pages'])
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Combine all text
                            full_text = "\n\n".join(page['text'] for page in extracted_pages)
                            
                            # Show processing summary
                            ocr_pages = sum(1 for page in extracted_pages if page.get('method') == 'OCR')
                            if ocr_pages > 0:
                                st.success(f"✅ Processed {len(extracted_pages)} pages ({ocr_pages} with OCR)")
                            else:
                                st.success(f"✅ Processed {len(extracted_pages)} pages")
                            
                        else:
                            # Use standard extractor for other formats
                            full_text = self.universal_extractor.extract_content(temp_path, uploaded_file.type)
                        
                        # Store extracted content efficiently
                        content_data = {
                            'filename': uploaded_file.name,
                            'content': full_text,
                            'file_size_mb': file_size_mb,
                            'char_count': len(full_text),
                            'extraction_method': 'OCR-enhanced' if uploaded_file.name.lower().endswith('.pdf') else 'standard'
                        }
                        
                        all_extracted_content.append(content_data)
                        
                        # Show preview
                        st.text_area(
                            "Content Preview",
                            full_text[:1000] + "..." if len(full_text) > 1000 else full_text,
                            height=150,
                            disabled=True
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        self.logger.error(f"File processing error: {e}")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            
            # Store all content efficiently
            if all_extracted_content:
                store_large_data('extracted_content', all_extracted_content, force_disk=True)
                st.session_state['step_statuses'][1] = True
                st.session_state['total_files_processed'] = len(all_extracted_content)
                
                st.success(f"✅ Successfully processed {len(all_extracted_content)} files")
                
                if st.button("➡️ Proceed to Content Analysis"):
                    st.session_state['current_step'] = 2
                    st.rerun()
    
    def _render_step_2_analysis(self):
        """Step 2: Content Analysis with enhanced detection"""
        
        st.header("🔍 Step 2: Content Analysis")
        
        extracted_content = retrieve_data('extracted_content', [])
        
        if not extracted_content:
            st.warning("No content found. Please upload documents first.")
            return
        
        # Analyze all content
        all_analyzed_items = []
        
        for file_data in extracted_content:
            with st.expander(f"Analyzing: {file_data['filename']}", expanded=True):
                
                # Detect content type and extract items
                content_items = self.content_detector.detect_and_extract(
                    file_data['content'], 
                    file_data['filename']
                )
                
                if content_items:
                    st.success(f"✅ Found {len(content_items)} content items")
                    
                    # Show content type distribution
                    content_types = {}
                    for item in content_items:
                        content_type = item.get('type', 'unknown')
                        content_types[content_type] = content_types.get(content_type, 0) + 1
                    
                    st.write("**Content Distribution:**")
                    for content_type, count in content_types.items():
                        st.write(f"- {content_type.title()}: {count} items")
                    
                    all_analyzed_items.extend(content_items)
                else:
                    st.warning("No structured content found in this file")
        
        if all_analyzed_items:
            # Store analyzed content
            store_large_data('analyzed_content', all_analyzed_items, force_disk=True)
            st.session_state['step_statuses'][2] = True
            st.session_state['total_items_processed'] = len(all_analyzed_items)
            
            # Show overall statistics
            st.subheader("📊 Analysis Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", len(all_analyzed_items))
            with col2:
                avg_length = sum(len(item.get('content', '')) for item in all_analyzed_items) / len(all_analyzed_items)
                st.metric("Avg Length", f"{avg_length:.0f} chars")
            with col3:
                quality_scores = [item.get('quality_score', 0) for item in all_analyzed_items]
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
                st.metric("Avg Quality", f"{avg_quality:.1f}/10")
            
            if st.button("➡️ Proceed to AI Enhancement"):
                st.session_state['current_step'] = 3
                st.rerun()
    
    def _render_step_3_enhancement(self):
        """Step 3: AI Enhancement with async processing"""
        
        st.header("✨ Step 3: AI Enhancement")
        
        analyzed_content = retrieve_data('analyzed_content', [])
        
        if not analyzed_content:
            st.warning("No analyzed content found. Please complete content analysis first.")
            return
        
        # Enhancement configuration
        col1, col2 = st.columns(2)
        
        with col1:
            selected_tone = st.selectbox(
                "Select Enhancement Tone",
                ["advaita_vedanta", "zen_buddhism", "christian_mysticism", 
                 "sufi_mysticism", "mindfulness_meditation", "universal_wisdom"]
            )
        
        with col2:
            # Show processing estimates
            from async_enhancement_processor import get_chunk_statistics
            stats = get_chunk_statistics(analyzed_content)
            
            st.write("**Processing Estimates:**")
            st.write(f"- Items: {stats['total_items']}")
            st.write(f"- Chunks: {stats['num_chunks']}")
            st.write(f"- Est. Time: {stats['estimated_time']}")
        
        # Enhancement options
        with st.expander("⚙️ Advanced Options", expanded=False):
            chunk_size = st.slider("Chunk Size", 5, 25, 15, help="Items per batch")
            max_concurrent = st.slider("Max Concurrent", 1, 10, 5, help="Parallel batches")
            
            # Show memory warning for large batches
            if len(analyzed_content) > 100:
                st.warning("⚠️ Large batch detected. Consider processing in smaller chunks.")
        
        # Start enhancement
        if st.button("🚀 Start AI Enhancement"):
            
            # Load prompt template
            prompt_template = self.prompt_engine.load_prompt_template(selected_tone)
            
            if not prompt_template:
                st.error("Failed to load prompt template")
                return
            
            # Process with async enhancement
            with st.spinner("Processing enhancement batches..."):
                try:
                    enhanced_results = self.async_processor.process_with_progress(
                        analyzed_content, prompt_template, selected_tone
                    )
                    
                    if enhanced_results:
                        # Store enhanced content
                        store_large_data('enhanced_content', enhanced_results, force_disk=True)
                        st.session_state['step_statuses'][3] = True
                        
                        st.success(f"✅ Enhanced {len(enhanced_results)} items successfully!")
                        
                        # Show enhancement summary
                        st.subheader("📊 Enhancement Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Enhanced Items", len(enhanced_results))
                        with col2:
                            avg_improvement = sum(
                                len(item.get('enhanced', '')) - len(item.get('original', {}).get('content', ''))
                                for item in enhanced_results
                            ) / len(enhanced_results)
                            st.metric("Avg Expansion", f"{avg_improvement:.0f} chars")
                        with col3:
                            st.metric("Tone Applied", selected_tone.replace('_', ' ').title())
                        
                        if st.button("➡️ Proceed to Manual Review"):
                            st.session_state['current_step'] = 4
                            st.rerun()
                    
                    else:
                        st.error("Enhancement failed. Please try again.")
                
                except Exception as e:
                    st.error(f"Enhancement error: {e}")
                    self.logger.error(f"Enhancement error: {e}")
    
    def _render_step_4_review(self):
        """Step 4: Manual Review with enhanced interface"""
        
        st.header("📋 Step 4: Manual Review")
        
        enhanced_content = retrieve_data('enhanced_content', [])
        
        if not enhanced_content:
            st.warning("No enhanced content found. Please complete AI enhancement first.")
            return
        
        # Render manual review interface
        reviewed_content = self.manual_review.render_review_interface(enhanced_content)
        
        if reviewed_content:
            # Store reviewed content
            store_large_data('reviewed_content', reviewed_content, force_disk=True)
            st.session_state['step_statuses'][4] = True
            
            if st.button("➡️ Proceed to Export"):
                st.session_state['current_step'] = 5
                st.rerun()
    
    def _render_step_5_export(self):
        """Step 5: Export with confirmations and validation"""
        
        st.header("📦 Step 5: Export & Deploy")
        
        reviewed_content = retrieve_data('reviewed_content', [])
        
        if not reviewed_content:
            st.warning("No reviewed content found. Please complete manual review first.")
            return
        
        # Export options with confirmations
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "JSONL", "CSV", "XLSX", "ZIP Package"]
        )
        
        # Show export preview and confirmation
        if self.export_confirmations.show_export_confirmation(reviewed_content, export_format):
            
            # Validate schema before export
            if self.schema_validator.validate_export_data(reviewed_content):
                
                # Perform export
                if export_format == "ZIP Package":
                    export_result = self.zip_exporter.create_enhanced_export(reviewed_content)
                else:
                    # Handle other formats
                    export_result = self._export_format(reviewed_content, export_format)
                
                if export_result:
                    st.session_state['step_statuses'][5] = True
                    st.success("✅ Export completed successfully!")
                    
                    # Offer Hugging Face upload
                    if st.checkbox("Upload to Hugging Face"):
                        self.hf_uploader.render_upload_interface(export_result)
            
            else:
                st.error("Export validation failed. Please check your data.")
    
    def _export_format(self, content, format_type):
        """Export content in specified format"""
        # Implementation for different export formats
        # This would integrate with existing export modules
        pass


def main():
    """Main application entry point"""
    
    # Set page config
    st.set_page_config(
        page_title="Fine-Tune Data System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and run application
    app = EnhancedUniversalAITrainingDataCreatorOptimized()
    app.run()


if __name__ == "__main__":
    main()

