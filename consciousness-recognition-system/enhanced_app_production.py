"""
Enhanced Universal AI Training Data Creator - Production Version
===============================================================

A comprehensive, production-grade application for creating high-quality AI training datasets
with advanced features including caching, debugging, quality control, and export validation.

Features:
- Advanced caching for performance optimization
- Hidden debug mode for troubleshooting
- Enhanced logging with detailed traces
- Testing support with backend function extraction
- Semantic similarity and hallucination detection
- Collapsible UI sections and breadcrumb navigation
- Pydantic schema validation for exports
- Export confirmations with file size validation
- Dark mode toggle and responsive design
- Comprehensive error handling and recovery

Author: Enhanced Universal AI Training Data Creator Team
Version: 2.0 (Production)
"""

import streamlit as st
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure page
st.set_page_config(
    page_title="Enhanced Universal AI Training Data Creator - Production",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

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
    
    # Production enhancement modules
    from enhanced_caching import EnhancedCaching, enhanced_caching
    from enhanced_debugging import EnhancedDebugging, enhanced_debugging
    from enhanced_logging import EnhancedLogging, enhanced_logging
    from testing_support import TestingSupport, testing_support
    from quality_control_enhanced import EnhancedQualityControl, enhanced_quality_control
    from ui_polish_enhanced import EnhancedUIPolish, ui_polish
    from metadata_schema_validator import MetadataSchemaValidator, metadata_validator
    from export_confirmations import ExportConfirmations, export_confirmations
    
except ImportError as e:
    st.error(f"❌ **Import Error:** {e}")
    st.error("Please ensure all required modules are installed and accessible.")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedUniversalAITrainingDataCreatorProduction:
    """Production-grade Enhanced Universal AI Training Data Creator"""
    
    def __init__(self):
        """Initialize the production application"""
        self.initialize_session_state()
        self.initialize_modules()
        self.setup_debug_mode()
        
        # Log application startup
        enhanced_logging.log_event("application_startup", {
            "version": "2.0",
            "session_id": st.session_state.get('session_id', 'unknown'),
            "debug_mode": st.session_state.get('debug_mode', False)
        })
    
    def initialize_session_state(self):
        """Initialize session state with production defaults"""
        defaults = {
            'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'session_start_time': datetime.now(),
            'debug_mode': st.secrets.get("debug_mode", False),
            'selected_theme': 'default',
            'current_step': 1,
            'step_statuses': {f'step_{i}': False for i in range(1, 6)},
            'uploaded_files': [],
            'extracted_content': None,
            'content_analysis': None,
            'enhanced_content': None,
            'quality_assessment': None,
            'manual_review_data': [],
            'export_data': None,
            'total_files_processed': 0,
            'total_items_processed': 0,
            'total_session_cost': 0.0,
            'processing_history': [],
            'cache_enabled': True,
            'auto_save_enabled': True
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def initialize_modules(self):
        """Initialize all application modules"""
        try:
            # Core modules
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
            
            # Production enhancement modules
            self.caching = enhanced_caching
            self.debugging = enhanced_debugging
            self.logging = enhanced_logging
            self.testing = testing_support
            self.quality_control = enhanced_quality_control
            self.ui_polish = ui_polish
            self.schema_validator = metadata_validator
            self.export_confirmations = export_confirmations
            
            logger.info("All modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Module initialization error: {e}")
            st.error(f"❌ **Module Initialization Error:** {e}")
            st.stop()
    
    def setup_debug_mode(self):
        """Setup debug mode if enabled"""
        if st.session_state.get('debug_mode', False):
            self.debugging.enable_debug_mode()
            st.sidebar.success("🐛 Debug Mode Enabled")
            
            # Add debug controls
            with st.sidebar.expander("🔧 Debug Controls", expanded=False):
                if st.button("📊 Show Session State"):
                    self.debugging.show_session_state()
                
                if st.button("📈 Show Performance Metrics"):
                    self.debugging.show_performance_metrics()
                
                if st.button("🧪 Run Test Suite"):
                    test_results = self.testing.run_test_suite()
                    st.json(test_results)
                
                if st.button("🗑️ Clear Cache"):
                    self.caching.clear_all_caches()
                    st.success("Cache cleared!")
    
    def render_header(self):
        """Render application header with branding and navigation"""
        # Apply theming
        self.theming.apply_theme(st.session_state.get('selected_theme', 'default'))
        
        # Header with dark mode toggle
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.title("🧠 Enhanced Universal AI Training Data Creator")
            st.markdown("*Production-Grade AI Training Dataset Creation Platform*")
        
        with col2:
            self.ui_polish.render_dark_mode_toggle()
        
        # Breadcrumb navigation
        self.ui_polish.render_breadcrumb_navigation(st.session_state.get('current_step', 1))
        
        # Session info in sidebar
        self.ui_polish.render_session_info()
    
    def render_step_1_upload_extract(self):
        """Step 1: Upload and Extract Content"""
        st.header("📤 Step 1: Upload & Extract Content")
        
        # Help tooltips
        self.ui_polish.render_help_tooltips('upload')
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload your files",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, MD"
        )
        
        if uploaded_files:
            st.session_state['uploaded_files'] = uploaded_files
            
            # Process files with caching
            if st.button("🔄 Extract Content", type="primary"):
                with st.spinner("Extracting content..."):
                    try:
                        # Use caching for extraction
                        extraction_results = self.caching.cached_content_extraction(
                            uploaded_files, self.extractor.extract_content
                        )
                        
                        st.session_state['extracted_content'] = extraction_results
                        st.session_state['step_statuses']['step_1'] = True
                        st.session_state['current_step'] = 2
                        st.session_state['total_files_processed'] += len(uploaded_files)
                        
                        # Log extraction
                        self.logging.log_extraction_event(
                            len(uploaded_files), 
                            sum(len(result.get('content', '')) for result in extraction_results)
                        )
                        
                        st.success(f"✅ Successfully extracted content from {len(uploaded_files)} files!")
                        st.rerun()
                        
                    except Exception as e:
                        self.logging.log_error("content_extraction", str(e))
                        st.error(f"❌ Extraction failed: {e}")
        
        # Display extracted content if available
        if st.session_state.get('extracted_content'):
            st.success("✅ Content extracted successfully!")
            
            # Show extraction summary
            content_data = st.session_state['extracted_content']
            total_chars = sum(len(item.get('content', '')) for item in content_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Processed", len(content_data))
            with col2:
                st.metric("Total Characters", f"{total_chars:,}")
            with col3:
                st.metric("Avg Chars/File", f"{total_chars // len(content_data) if content_data else 0:,}")
    
    def render_step_2_content_analysis(self):
        """Step 2: Content Analysis"""
        if not st.session_state.get('step_statuses', {}).get('step_1', False):
            st.info("⏳ Please complete Step 1 first.")
            return
        
        st.header("🔍 Step 2: Content Analysis")
        
        # Help tooltips
        self.ui_polish.render_help_tooltips('analysis')
        
        extracted_content = st.session_state.get('extracted_content', [])
        
        if st.button("🔍 Analyze Content", type="primary"):
            with st.spinner("Analyzing content..."):
                try:
                    # Analyze content with caching
                    analysis_results = []
                    
                    for item in extracted_content:
                        content = item.get('content', '')
                        
                        # Use testing support for backend analysis
                        stats = self.testing.compute_content_statistics(content)
                        content_type, confidence, metadata = self.testing.detect_content_type(content)
                        
                        analysis_results.append({
                            'content': content,
                            'statistics': stats,
                            'content_type': content_type,
                            'detection_confidence': confidence,
                            'detection_metadata': metadata
                        })
                    
                    st.session_state['content_analysis'] = analysis_results
                    st.session_state['step_statuses']['step_2'] = True
                    st.session_state['current_step'] = 3
                    
                    # Log analysis
                    self.logging.log_analysis_event(len(analysis_results))
                    
                    st.success("✅ Content analysis completed!")
                    st.rerun()
                    
                except Exception as e:
                    self.logging.log_error("content_analysis", str(e))
                    st.error(f"❌ Analysis failed: {e}")
        
        # Display analysis results
        if st.session_state.get('content_analysis'):
            analysis_data = st.session_state['content_analysis']
            
            # Aggregate statistics
            total_items = len(analysis_data)
            content_types = {}
            total_stats = {'total_chars': 0, 'total_words': 0, 'total_lines': 0}
            
            for item in analysis_data:
                content_type = item.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                stats = item.get('statistics', {})
                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)
            
            # Render advanced analysis with collapsible sections
            self.ui_polish.render_advanced_analysis_section(total_stats)
            
            # Content type distribution
            st.markdown("### 📊 Content Type Distribution")
            for content_type, count in content_types.items():
                percentage = (count / total_items) * 100
                st.write(f"**{content_type.title()}:** {count} items ({percentage:.1f}%)")
    
    def render_step_3_enhancement(self):
        """Step 3: Content Enhancement"""
        if not st.session_state.get('step_statuses', {}).get('step_2', False):
            st.info("⏳ Please complete Step 2 first.")
            return
        
        st.header("✨ Step 3: Content Enhancement")
        
        # Help tooltips
        self.ui_polish.render_help_tooltips('enhancement')
        
        # Tone selection
        available_tones = self.prompt_engine.get_available_tones()
        selected_tone = st.selectbox(
            "Select Spiritual Tone",
            available_tones,
            help="Choose the spiritual tone for content enhancement"
        )
        
        # Enhancement options
        col1, col2 = st.columns(2)
        
        with col1:
            enhancement_strength = st.slider(
                "Enhancement Strength",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Control the intensity of enhancement"
            )
        
        with col2:
            preserve_structure = st.checkbox(
                "Preserve Original Structure",
                value=True,
                help="Maintain the original content structure"
            )
        
        if st.button("✨ Enhance Content", type="primary"):
            with st.spinner("Enhancing content with AI..."):
                try:
                    analysis_data = st.session_state.get('content_analysis', [])
                    enhanced_results = []
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, item in enumerate(analysis_data):
                        content = item.get('content', '')
                        content_type = item.get('content_type', 'unknown')
                        
                        # Update progress
                        progress = (i + 1) / len(analysis_data)
                        progress_bar.progress(progress)
                        status_text.text(f"Enhancing item {i + 1} of {len(analysis_data)}")
                        
                        # Enhance with caching
                        enhanced_content = self.caching.cached_content_enhancement(
                            content, selected_tone, self.prompt_engine.enhance_content
                        )
                        
                        # Quality assessment
                        quality_assessment = self.quality_control.comprehensive_quality_assessment(
                            content, enhanced_content, selected_tone
                        )
                        
                        enhanced_results.append({
                            'original': content,
                            'enhanced': enhanced_content,
                            'tone': selected_tone,
                            'content_type': content_type,
                            'quality_assessment': quality_assessment,
                            'enhancement_metadata': {
                                'strength': enhancement_strength,
                                'preserve_structure': preserve_structure,
                                'timestamp': datetime.now().isoformat()
                            }
                        })
                        
                        # Update session cost tracking
                        st.session_state['total_session_cost'] += 0.001  # Estimate
                    
                    st.session_state['enhanced_content'] = enhanced_results
                    st.session_state['step_statuses']['step_3'] = True
                    st.session_state['current_step'] = 4
                    st.session_state['total_items_processed'] += len(enhanced_results)
                    
                    # Log enhancement
                    self.logging.log_enhancement_event(len(enhanced_results), selected_tone)
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ Content enhancement completed!")
                    st.rerun()
                    
                except Exception as e:
                    self.logging.log_error("content_enhancement", str(e))
                    st.error(f"❌ Enhancement failed: {e}")
        
        # Display enhancement progress if available
        if st.session_state.get('enhanced_content'):
            enhanced_data = st.session_state['enhanced_content']
            
            # Progress overview
            progress_data = {
                'total_items': len(enhanced_data),
                'processed_items': len(enhanced_data),
                'enhancement_stats': {
                    selected_tone: {
                        'count': len(enhanced_data),
                        'avg_time': 2.5,  # Estimate
                        'total_cost': st.session_state.get('total_session_cost', 0)
                    }
                }
            }
            
            self.ui_polish.render_enhancement_progress(progress_data)
    
    def render_step_4_review_validate(self):
        """Step 4: Review and Validate"""
        if not st.session_state.get('step_statuses', {}).get('step_3', False):
            st.info("⏳ Please complete Step 3 first.")
            return
        
        st.header("📋 Step 4: Review & Validate")
        
        # Help tooltips
        self.ui_polish.render_help_tooltips('review')
        
        enhanced_data = st.session_state.get('enhanced_content', [])
        
        if enhanced_data:
            # Quality dashboard
            st.markdown("### 🎯 Quality Assessment Dashboard")
            
            # Aggregate quality metrics
            quality_scores = [item.get('quality_assessment', {}).get('overall_score', 0) for item in enhanced_data]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            passed_items = sum(1 for item in enhanced_data 
                             if item.get('quality_assessment', {}).get('passed_threshold', False))
            pass_rate = passed_items / len(enhanced_data) if enhanced_data else 0
            
            # Quality overview
            quality_data = {
                'overall_score': avg_quality,
                'passed_threshold': pass_rate > 0.8,
                'flags': [],
                'metrics': {
                    'semantic_similarity': sum(item.get('quality_assessment', {}).get('metrics', {}).get('semantic_similarity', 0) for item in enhanced_data) / len(enhanced_data),
                    'hallucination_score': sum(item.get('quality_assessment', {}).get('metrics', {}).get('hallucination_score', 0) for item in enhanced_data) / len(enhanced_data),
                    'readability_score': sum(item.get('quality_assessment', {}).get('metrics', {}).get('readability_score', 0) for item in enhanced_data) / len(enhanced_data),
                    'coherence_score': sum(item.get('quality_assessment', {}).get('metrics', {}).get('coherence_score', 0) for item in enhanced_data) / len(enhanced_data)
                },
                'recommendations': []
            }
            
            if avg_quality < 0.7:
                quality_data['flags'].append('low_average_quality')
                quality_data['recommendations'].append('Consider manual review of low-quality items')
            
            if pass_rate < 0.8:
                quality_data['flags'].append('low_pass_rate')
                quality_data['recommendations'].append('Review items that failed quality thresholds')
            
            self.ui_polish.render_quality_dashboard(quality_data)
            
            # Manual review interface
            st.markdown("### ✏️ Manual Review")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_only_failed = st.checkbox("Show only failed items", value=False)
            
            with col2:
                show_only_flagged = st.checkbox("Show only flagged items", value=False)
            
            with col3:
                items_per_page = st.selectbox("Items per page", [5, 10, 20, 50], index=1)
            
            # Filter data
            filtered_data = enhanced_data
            
            if show_only_failed:
                filtered_data = [item for item in filtered_data 
                               if not item.get('quality_assessment', {}).get('passed_threshold', False)]
            
            if show_only_flagged:
                filtered_data = [item for item in filtered_data 
                               if item.get('quality_assessment', {}).get('flags', [])]
            
            # Pagination
            total_items = len(filtered_data)
            total_pages = (total_items + items_per_page - 1) // items_per_page
            
            if total_pages > 1:
                page = st.selectbox("Page", range(1, total_pages + 1)) - 1
            else:
                page = 0
            
            start_idx = page * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            page_data = filtered_data[start_idx:end_idx]
            
            # Render manual review interface
            review_results = self.manual_review.render_review_interface(page_data)
            
            if review_results:
                st.session_state['manual_review_data'] = review_results
                st.session_state['step_statuses']['step_4'] = True
                st.session_state['current_step'] = 5
                st.success("✅ Review completed!")
    
    def render_step_5_export_share(self):
        """Step 5: Export and Share"""
        if not st.session_state.get('step_statuses', {}).get('step_4', False):
            st.info("⏳ Please complete Step 4 first.")
            return
        
        st.header("📦 Step 5: Export & Share")
        
        # Help tooltips
        self.ui_polish.render_help_tooltips('export')
        
        enhanced_data = st.session_state.get('enhanced_content', [])
        review_data = st.session_state.get('manual_review_data', [])
        
        if enhanced_data:
            # Prepare export data
            export_data = []
            
            for item in enhanced_data:
                # Check if item was manually reviewed
                reviewed_item = next((r for r in review_data if r.get('original') == item.get('original')), None)
                
                if reviewed_item and reviewed_item.get('approved', True):
                    # Use reviewed version
                    export_item = {
                        'input': reviewed_item.get('original', ''),
                        'output': reviewed_item.get('enhanced', ''),
                        'content_type': item.get('content_type', ''),
                        'spiritual_tone': item.get('tone', ''),
                        'quality_metrics': item.get('quality_assessment', {}).get('metrics', {}),
                        'processing_metadata': {
                            'enhancement_model': 'gpt-4',
                            'processing_time': 2.5,
                            'created_at': datetime.now().isoformat(),
                            'schema_version': '1.0'
                        },
                        'manual_review': {
                            'reviewed': True,
                            'approved': True,
                            'reviewer_notes': reviewed_item.get('notes', '')
                        }
                    }
                elif not reviewed_item:
                    # Use original enhanced version
                    export_item = {
                        'input': item.get('original', ''),
                        'output': item.get('enhanced', ''),
                        'content_type': item.get('content_type', ''),
                        'spiritual_tone': item.get('tone', ''),
                        'quality_metrics': item.get('quality_assessment', {}).get('metrics', {}),
                        'processing_metadata': {
                            'enhancement_model': 'gpt-4',
                            'processing_time': 2.5,
                            'created_at': datetime.now().isoformat(),
                            'schema_version': '1.0'
                        }
                    }
                
                if 'export_item' in locals():
                    export_data.append(export_item)
            
            st.session_state['export_data'] = export_data
            
            # Export options
            st.markdown("### 📋 Export Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox(
                    "Export Format",
                    ['JSON', 'JSONL', 'CSV', 'XLSX', 'TXT'],
                    help="Choose the export format"
                )
            
            with col2:
                export_destination = st.selectbox(
                    "Export Destination",
                    ['Local Download', 'Hugging Face', 'Cloud Storage'],
                    help="Choose where to export the data"
                )
            
            # Schema validation
            st.markdown("### ✅ Schema Validation")
            
            if st.button("🔍 Validate Export Data"):
                with st.spinner("Validating data schema..."):
                    validation_results = self.schema_validator.validate_batch_training_data(export_data)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Items", validation_results['total_items'])
                    
                    with col2:
                        st.metric("Valid Items", validation_results['valid_items'])
                    
                    with col3:
                        validation_rate = validation_results['validation_rate'] * 100
                        st.metric("Validation Rate", f"{validation_rate:.1f}%")
                    
                    if validation_results['validation_errors']:
                        st.error("❌ Validation Errors Found:")
                        for error in validation_results['validation_errors'][:5]:  # Show first 5
                            st.write(f"Item {error['item_index']}: {error['errors']}")
                    else:
                        st.success("✅ All data passed validation!")
            
            # Export confirmation and execution
            st.markdown("### 🚀 Export Execution")
            
            if st.button("📦 Prepare Export", type="primary"):
                # Validate export safety
                destination_map = {
                    'Local Download': 'local',
                    'Hugging Face': 'huggingface',
                    'Cloud Storage': 'cloud'
                }
                
                validation_result = self.export_confirmations.validate_export_safety(
                    export_data, export_format.lower(), destination_map[export_destination]
                )
                
                # Show confirmation dialog
                if validation_result['safe_to_export']:
                    user_confirmed = self.export_confirmations.render_export_confirmation_dialog(
                        validation_result, export_format, export_destination
                    )
                    
                    if user_confirmed:
                        # Execute export
                        with st.spinner("Exporting data..."):
                            try:
                                if export_destination == 'Local Download':
                                    # Create download
                                    export_file = self.zip_exporter.create_comprehensive_export(
                                        export_data, export_format.lower()
                                    )
                                    
                                    with open(export_file, 'rb') as f:
                                        st.download_button(
                                            label="📥 Download Export",
                                            data=f.read(),
                                            file_name=os.path.basename(export_file),
                                            mime='application/zip'
                                        )
                                
                                elif export_destination == 'Hugging Face':
                                    # Upload to Hugging Face
                                    upload_result = self.hf_uploader.upload_dataset(export_data)
                                    
                                    if upload_result['success']:
                                        st.success(f"✅ Successfully uploaded to Hugging Face: {upload_result['url']}")
                                    else:
                                        st.error(f"❌ Upload failed: {upload_result['error']}")
                                
                                # Log export
                                self.logging.log_export_event(len(export_data), export_format, export_destination)
                                
                                st.session_state['step_statuses']['step_5'] = True
                                st.success("✅ Export completed successfully!")
                                
                            except Exception as e:
                                self.logging.log_error("export_execution", str(e))
                                st.error(f"❌ Export failed: {e}")
                else:
                    st.error("❌ Export validation failed. Please review the errors above.")
    
    def render_sidebar_controls(self):
        """Render sidebar controls and metrics"""
        with st.sidebar:
            st.markdown("## 🎛️ Controls")
            
            # Step completion status
            step_statuses = st.session_state.get('step_statuses', {})
            self.ui_polish.render_step_completion_status(step_statuses)
            
            st.markdown("---")
            
            # Reprocess controls
            self.ui_polish.render_reprocess_controls(step_statuses)
            
            st.markdown("---")
            
            # Theme selection
            st.markdown("### 🎨 Theme")
            themes = self.theming.get_available_themes()
            selected_theme = st.selectbox(
                "Select Theme",
                themes,
                index=themes.index(st.session_state.get('selected_theme', 'default'))
            )
            
            if selected_theme != st.session_state.get('selected_theme'):
                st.session_state['selected_theme'] = selected_theme
                st.rerun()
            
            st.markdown("---")
            
            # Performance metrics
            if st.session_state.get('debug_mode', False):
                st.markdown("### 📊 Performance")
                
                # Cache statistics
                cache_stats = self.caching.get_cache_statistics()
                st.write(f"**Cache Hits:** {cache_stats.get('hits', 0)}")
                st.write(f"**Cache Misses:** {cache_stats.get('misses', 0)}")
                
                # Quality trends
                quality_trends = self.quality_control.get_quality_trends()
                if 'avg_overall_score' in quality_trends:
                    st.write(f"**Avg Quality:** {quality_trends['avg_overall_score']:.3f}")
            
            st.markdown("---")
            
            # Keyboard shortcuts
            self.ui_polish.render_keyboard_shortcuts()
    
    def run(self):
        """Run the main application"""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar
            self.render_sidebar_controls()
            
            # Main content area
            current_step = st.session_state.get('current_step', 1)
            
            # Render appropriate step
            if current_step == 1:
                self.render_step_1_upload_extract()
            elif current_step == 2:
                self.render_step_2_content_analysis()
            elif current_step == 3:
                self.render_step_3_enhancement()
            elif current_step == 4:
                self.render_step_4_review_validate()
            elif current_step == 5:
                self.render_step_5_export_share()
            
            # Debug information
            if st.session_state.get('debug_mode', False):
                self.debugging.render_debug_panel()
        
        except Exception as e:
            self.logging.log_error("application_runtime", str(e))
            st.error(f"❌ **Application Error:** {e}")
            
            if st.session_state.get('debug_mode', False):
                st.exception(e)


# Main application entry point
if __name__ == "__main__":
    try:
        app = EnhancedUniversalAITrainingDataCreatorProduction()
        app.run()
    except Exception as e:
        st.error(f"❌ **Critical Error:** {e}")
        st.error("Please check your configuration and try again.")
        
        # Emergency debug mode
        if st.button("🐛 Enable Emergency Debug"):
            st.session_state['debug_mode'] = True
            st.rerun()

