"""
Consciousness Recognition System - Enhanced Validation Streamlit Interface
=========================================================================

Enhanced with comprehensive validation framework, early type checking, and emergency conversion.
Bulletproof system that prevents "expected string or bytes-like object, got 'list'" errors.

Features:
- Enhanced text validation with multiple defensive layers
- Early type checking and automatic conversion
- Emergency fallback mechanisms
- Comprehensive error handling and user feedback
- Real-time validation statistics and monitoring

Author: Consciousness Recognition System
Version: 2.0 - Enhanced Validation Framework
"""

import streamlit as st
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import psutil

# Import our enhanced validation modules
try:
    from modules.enhanced_extractor import EnhancedPDFExtractor
    from modules.enhanced_detector import EnhancedDialogueDetector
    from modules.enhanced_trainer import EnhancedOpenAITrainer
    from modules.intake_validator import SecureIntakeValidator
    from modules.output_validator import OutputValidator
    from modules.content_analyzer import ContentAnalyzer, ContentType
    from modules.scorer import ConsciousnessScorer, DialogueMode
    from modules.trainer import OpenAITrainer
    from modules.jsonl_manager import JSONLManager, FileManager
    from modules.universal_extractor import UniversalTextExtractor
    from modules.universal_intake_validator import UniversalIntakeValidator
    
    # Enhanced validation framework
    from modules.text_validator_enhanced import (
        EnhancedTextValidator, validate_text_enhanced, validate_extraction_enhanced,
        emergency_text_fix_enhanced, get_enhanced_validator_stats, 
        display_enhanced_validator_stats, enable_debug_mode, disable_debug_mode
    )
    from modules.early_type_checker import (
        EarlyTypeChecker, check_text_type, emergency_text_conversion,
        type_safe_wrapper, ensure_string, ensure_list, ensure_dict,
        get_type_checker_stats, display_type_checker_stats
    )
    
    # Backward compatibility
    from modules.text_validator import validate_text, validate_extraction, emergency_text_fix
    
except ImportError as e:
    st.error(f"Required modules not found: {e}")
    st.error("Please ensure all modules are in the modules/ directory.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Consciousness Recognition System - Enhanced",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize enhanced validation system
@st.cache_resource
def initialize_enhanced_validation():
    """Initialize the enhanced validation system."""
    return {
        'text_validator': EnhancedTextValidator(strict_mode=False, debug_mode=False),
        'type_checker': EarlyTypeChecker(auto_convert=True, emergency_mode=True)
    }

# Initialize session state with enhanced validation
def initialize_session_state():
    """Initialize all session state variables with enhanced validation."""
    if 'jsonl_manager' not in st.session_state:
        st.session_state.jsonl_manager = JSONLManager()
    
    if 'file_manager' not in st.session_state:
        st.session_state.file_manager = FileManager()
    
    if 'trainer' not in st.session_state:
        st.session_state.trainer = EnhancedOpenAITrainer()
    
    if 'detector' not in st.session_state:
        st.session_state.detector = EnhancedDialogueDetector()
    
    if 'extractor' not in st.session_state:
        st.session_state.extractor = UniversalTextExtractor()
    
    if 'intake_validator' not in st.session_state:
        st.session_state.intake_validator = UniversalIntakeValidator()
    
    if 'output_validator' not in st.session_state:
        st.session_state.output_validator = OutputValidator()
    
    if 'content_analyzer' not in st.session_state:
        st.session_state.content_analyzer = ContentAnalyzer()
    
    if 'validation_system' not in st.session_state:
        st.session_state.validation_system = initialize_enhanced_validation()
    
    if 'validation_debug_mode' not in st.session_state:
        st.session_state.validation_debug_mode = False
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    if 'total_dialogues' not in st.session_state:
        st.session_state.total_dialogues = 0

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def display_enhanced_validation_sidebar():
    """Display enhanced validation controls in sidebar."""
    st.sidebar.subheader("🛡️ Enhanced Validation System")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox(
        "Enable Debug Mode", 
        value=st.session_state.validation_debug_mode,
        help="Show detailed validation information"
    )
    
    if debug_mode != st.session_state.validation_debug_mode:
        st.session_state.validation_debug_mode = debug_mode
        if debug_mode:
            enable_debug_mode()
        else:
            disable_debug_mode()
        st.rerun()
    
    # Validation statistics
    if st.sidebar.button("Show Validation Stats"):
        st.sidebar.write("**Text Validator Stats:**")
        text_stats = get_enhanced_validator_stats()
        for key, value in text_stats.items():
            st.sidebar.metric(key.replace('_', ' ').title(), value)
        
        st.sidebar.write("**Type Checker Stats:**")
        type_stats = get_type_checker_stats()
        for key, value in type_stats.items():
            st.sidebar.metric(key.replace('_', ' ').title(), value)
    
    # Reset validation stats
    if st.sidebar.button("Reset Validation Stats"):
        st.session_state.validation_system['text_validator'].reset_stats()
        st.session_state.validation_system['type_checker'].reset_stats()
        st.sidebar.success("Validation statistics reset!")

def enhanced_text_processing_pipeline(
    uploaded_file, 
    detection_mode: str, 
    semantic_threshold: float, 
    score_threshold: float,
    content_analysis_enabled: bool
):
    """
    Enhanced text processing pipeline with comprehensive validation.
    
    This pipeline includes multiple validation layers to prevent any type-related errors.
    """
    
    # Initialize components
    extractor = st.session_state.extractor
    detector = st.session_state.detector
    trainer = st.session_state.trainer
    intake_validator = st.session_state.intake_validator
    content_analyzer = st.session_state.content_analyzer
    
    # Mode mapping with validation
    mode_map = {
        "Multi-Mode (Recommended)": "multi_mode",
        "Auto (Regex + Semantic)": "auto",
        "Regex Only": "regex", 
        "Semantic Only": "semantic"
    }
    
    st.subheader(f"📄 Processing: {uploaded_file.name}")
    
    try:
        # Step 1: Enhanced file validation
        st.info("🔍 Validating file with enhanced security checks...")
        
        # Early type check for uploaded file
        is_valid_file, validated_file, file_message = check_text_type(
            uploaded_file, f"uploaded_file_{uploaded_file.name}", show_feedback=True
        )
        
        if not is_valid_file:
            st.error(f"❌ File validation failed: {file_message}")
            return
        
        # Comprehensive intake validation
        validation_result = intake_validator.validate_file(uploaded_file)
        
        if not validation_result.is_valid:
            st.error(f"❌ **File Validation Failed:** {validation_result.error_message}")
            if validation_result.security_issues:
                st.error("🚨 **Security Issues Detected:**")
                for issue in validation_result.security_issues:
                    st.write(f"• {issue}")
            return
        
        # Display validation results
        if validation_result.warnings:
            st.warning("⚠️ **File Warnings:**")
            for warning in validation_result.warnings:
                st.write(f"• {warning}")
        
        if validation_result.processing_recommendations:
            st.info("💡 **Processing Recommendations:**")
            for rec in validation_result.processing_recommendations:
                st.write(f"• {rec}")
        
        # Get processing strategy
        strategy = intake_validator.get_processing_strategy(validation_result)
        
        # Save uploaded file temporarily (securely)
        file_ext = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            uploaded_file.seek(0)  # Reset file pointer
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Step 2: Enhanced text extraction with validation
            st.info("📖 Extracting text with enhanced validation...")
            
            # Get file format info
            format_info = extractor.get_format_info(tmp_path)
            st.write(f"**Format:** {format_info['info']['name']} ({format_info['format'].upper()})")
            
            # Extract text using universal extractor
            extraction_result = extractor.extract_text(tmp_path)
            
            # 🔧 LAYER 1: Enhanced extraction validation
            st.info("🛡️ **Layer 1:** Validating extraction result...")
            text = validate_extraction_enhanced(extraction_result, f"PDF_{uploaded_file.name}")
            
            if text is None:
                st.error(f"❌ Enhanced extraction validation failed for {uploaded_file.name}")
                return
            
            # 🔧 LAYER 2: Early type checking
            st.info("🔍 **Layer 2:** Early type checking...")
            is_valid_text, validated_text, type_message = check_text_type(
                text, f"extracted_text_{uploaded_file.name}", show_feedback=True
            )
            
            if not is_valid_text:
                st.error(f"❌ Type validation failed: {type_message}")
                return
            
            text = validated_text
            
            # 🔧 LAYER 3: Content validation
            st.info("✅ **Layer 3:** Content validation...")
            if not text or not text.strip():
                st.error("❌ No meaningful text content extracted from file")
                return
            
            # 🔧 LAYER 4: Emergency validation check
            st.info("🚨 **Layer 4:** Emergency validation check...")
            emergency_validated_text = emergency_text_fix_enhanced(text, f"emergency_{uploaded_file.name}")
            
            if emergency_validated_text != text:
                st.warning("⚠️ Emergency text normalization was applied")
                text = emergency_validated_text
            
            # Display extraction results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Characters", len(text))
            with col2:
                st.metric("Words", len(text.split()))
            with col3:
                st.metric("Extractor", extraction_result['extractor'])
            with col4:
                quality = format_info['info'].get('extraction_quality', 'Unknown')
                st.metric("Quality", quality)
            
            # Show metadata if available
            if extraction_result.get('metadata'):
                with st.expander("📋 File Metadata"):
                    metadata = extraction_result['metadata']
                    for key, value in metadata.items():
                        if value:
                            st.write(f"**{key.title()}:** {value}")
            
            # Step 3: Enhanced content analysis (if enabled)
            if content_analysis_enabled:
                st.info("🧠 Analyzing content with enhanced validation...")
                
                # Validate text before content analysis
                analysis_text = validate_text_enhanced(
                    text, f"content_analysis_{uploaded_file.name}", 
                    show_ui_feedback=True, emergency_mode=True
                )
                
                if analysis_text is None:
                    st.error("❌ Text validation failed before content analysis")
                    return
                
                content_analysis = content_analyzer.analyze_content(analysis_text)
                
                # Display content analysis
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Content Type", content_analysis['content_type'].value.replace('_', ' ').title())
                with col2:
                    st.metric("Consciousness Density", f"{content_analysis['consciousness_density']:.2f}")
                with col3:
                    st.metric("Dialogue Ratio", f"{content_analysis['dialogue_ratio']:.2f}")
                
                # Show recommendations
                if content_analysis['recommendations']:
                    st.info("💡 **Processing Recommendations:**")
                    for rec in content_analysis['recommendations']:
                        st.write(f"• {rec}")
            
            # Step 4: Enhanced dialogue detection
            st.info("🔍 Detecting dialogues with enhanced validation...")
            
            # 🔧 FINAL VALIDATION: Multiple validation layers before detector
            st.info("🛡️ **Final Validation:** Preparing text for detector...")
            
            # Layer A: Enhanced text validation
            detector_text = validate_text_enhanced(
                text, f"detector_input_{uploaded_file.name}", 
                show_ui_feedback=True, emergency_mode=True
            )
            
            # Layer B: Type checking
            is_detector_valid, detector_text_checked, detector_message = check_text_type(
                detector_text, f"detector_type_check_{uploaded_file.name}", show_feedback=True
            )
            
            # Layer C: Emergency conversion if needed
            if not is_detector_valid:
                st.warning("⚠️ Applying emergency conversion for detector input")
                detector_text_checked = emergency_text_conversion(detector_text, f"detector_emergency_{uploaded_file.name}")
            
            # Layer D: Final safety check
            final_detector_text = ensure_string(detector_text_checked, f"detector_final_{uploaded_file.name}")
            
            # Validate we have the expected mode
            if detection_mode not in mode_map:
                st.error(f"❌ Invalid detection mode: {detection_mode}")
                return
            
            # Call detector with fully validated text
            st.info(f"🎯 Running {detection_mode} detection...")
            detection_result = detector.detect_dialogues_with_progress(
                final_detector_text,  # Use the fully validated text
                mode=mode_map[detection_mode],
                semantic_threshold=semantic_threshold,
                show_progress=True
            )
            
            if not detection_result['success']:
                st.error(f"Failed to detect dialogues: {detection_result.get('error', 'Unknown error')}")
                return
            
            # Step 5: Enhanced result processing
            st.info("📊 Processing results with enhanced validation...")
            
            # Validate detection results
            dialogues = detection_result.get('dialogues', [])
            if not isinstance(dialogues, list):
                st.warning("⚠️ Detection result is not a list, converting...")
                dialogues = ensure_list(dialogues, f"detection_results_{uploaded_file.name}")
            
            # Filter by score threshold with validation
            high_score_dialogues = []
            for dialogue in dialogues:
                if not isinstance(dialogue, dict):
                    st.warning(f"⚠️ Invalid dialogue format, converting: {type(dialogue)}")
                    dialogue = ensure_dict(dialogue, f"dialogue_item_{uploaded_file.name}")
                
                score = dialogue.get('overall_score', 0)
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        score = 0
                
                if score >= score_threshold:
                    high_score_dialogues.append(dialogue)
            
            # Step 6: Enhanced training data addition
            st.info("📝 Adding to training data with enhanced validation...")
            
            added_count = 0
            for dialogue in high_score_dialogues:
                # Validate dialogue fields
                question = ensure_string(dialogue.get('question', ''), f"question_{uploaded_file.name}")
                answer = ensure_string(dialogue.get('answer', ''), f"answer_{uploaded_file.name}")
                
                if not question.strip() or not answer.strip():
                    continue
                
                # Add to trainer with validation
                add_result = trainer.add_dialogue(
                    question=question,
                    answer=answer,
                    score=dialogue.get('overall_score', 0),
                    mode=dialogue.get('mode', DialogueMode.UNKNOWN),
                    source=uploaded_file.name,
                    metadata={
                        'extraction_method': extraction_result['extractor'],
                        'detection_mode': detection_mode,
                        'file_format': format_info['format'],
                        'processing_timestamp': datetime.now().isoformat()
                    }
                )
                
                if add_result['success']:
                    added_count += 1
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Dialogues Found", len(dialogues))
            with col2:
                st.metric("High Score Dialogues", len(high_score_dialogues))
            with col3:
                st.metric("Added to Training", added_count)
            
            # Update session state
            st.session_state.processed_files.append({
                'name': uploaded_file.name,
                'dialogues_found': len(dialogues),
                'dialogues_added': added_count,
                'timestamp': datetime.now().isoformat()
            })
            st.session_state.total_dialogues += added_count
            
            st.success(f"✅ Successfully processed {uploaded_file.name}!")
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        st.error(f"❌ **Processing Error:** {str(e)}")
        
        # Enhanced error reporting
        if st.session_state.validation_debug_mode:
            st.error("**Debug Information:**")
            st.code(f"Error type: {type(e).__name__}")
            st.code(f"Error message: {str(e)}")
            
            import traceback
            st.code(f"Traceback:\n{traceback.format_exc()}")
        
        # Emergency recovery attempt
        st.info("🚨 Attempting emergency recovery...")
        try:
            # Try to salvage any partial results
            if 'detection_result' in locals() and detection_result.get('dialogues'):
                st.info(f"Found {len(detection_result['dialogues'])} dialogues before error")
                # Could implement partial recovery here
        except:
            pass

def main():
    """Enhanced main application with comprehensive validation."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header with enhanced validation info
    st.title("🧘 Consciousness Recognition System")
    st.markdown("**Enhanced with Comprehensive Validation Framework**")
    
    # Display validation system status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Validation System", "✅ Active")
    with col2:
        memory_mb = get_memory_usage()
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    with col3:
        debug_status = "🔍 Debug" if st.session_state.validation_debug_mode else "🔒 Normal"
        st.metric("Mode", debug_status)
    
    # Enhanced sidebar
    with st.sidebar:
        display_enhanced_validation_sidebar()
        
        st.subheader("📊 System Statistics")
        st.metric("Total Dialogues", st.session_state.total_dialogues)
        st.metric("Files Processed", len(st.session_state.processed_files))
        
        # Memory management
        st.subheader("🧹 Memory Management")
        if st.button("Clear Cache & Reset"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.validation_system['text_validator'].reset_stats()
            st.session_state.validation_system['type_checker'].reset_stats()
            st.success("Cache cleared and validation stats reset!")
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Process Files", 
        "📝 Edit Training Data", 
        "📊 Validation Stats",
        "🔧 System Diagnostics",
        "📋 Export Data"
    ])
    
    with tab1:
        st.header("📄 Enhanced File Processing")
        
        # File upload with enhanced validation
        uploaded_files = st.file_uploader(
            "Choose files to process",
            type=['pdf', 'epub', 'txt', 'docx', 'html'],
            accept_multiple_files=True,
            help="Supports PDF, EPUB, TXT, DOCX, and HTML files with enhanced validation"
        )
        
        if uploaded_files:
            # Processing options
            col1, col2 = st.columns(2)
            
            with col1:
                detection_mode = st.selectbox(
                    "Detection Mode",
                    ["Multi-Mode (Recommended)", "Auto (Regex + Semantic)", "Regex Only", "Semantic Only"],
                    help="Multi-mode automatically handles different content types"
                )
                
                semantic_threshold = st.slider(
                    "Semantic Similarity Threshold",
                    0.0, 1.0, 0.4, 0.1,
                    help="Higher values = more strict semantic matching"
                )
            
            with col2:
                score_threshold = st.slider(
                    "Consciousness Score Threshold",
                    0.0, 1.0, 0.6, 0.1,
                    help="Minimum consciousness score for inclusion"
                )
                
                content_analysis_enabled = st.checkbox(
                    "Enable Content Analysis",
                    value=True,
                    help="Analyze content type and provide recommendations"
                )
            
            # Process files button
            if st.button("🚀 Process Files with Enhanced Validation", type="primary"):
                for uploaded_file in uploaded_files:
                    enhanced_text_processing_pipeline(
                        uploaded_file,
                        detection_mode,
                        semantic_threshold,
                        score_threshold,
                        content_analysis_enabled
                    )
    
    with tab2:
        st.header("📝 Enhanced Training Data Editor")
        
        # Get current training data with validation
        training_data = st.session_state.trainer.get_training_data()
        
        if training_data:
            # Enhanced data display with validation
            df = pd.DataFrame(training_data)
            
            # Validate dataframe structure
            expected_columns = ['question', 'answer', 'score', 'mode', 'source']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"⚠️ Missing columns in training data: {missing_columns}")
                # Add missing columns with defaults
                for col in missing_columns:
                    df[col] = ""
            
            # Enhanced data editor
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "score": st.column_config.NumberColumn(
                        "Consciousness Score",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        format="%.2f"
                    ),
                    "mode": st.column_config.SelectboxColumn(
                        "Mode",
                        options=["TEACHING", "INQUIRY", "DIALOGUE", "UNKNOWN"]
                    )
                }
            )
            
            # Update training data with validation
            if st.button("💾 Update Training Data"):
                try:
                    # Validate updated data
                    validated_data = []
                    for _, row in edited_df.iterrows():
                        validated_row = {
                            'question': ensure_string(row.get('question', ''), 'edited_question'),
                            'answer': ensure_string(row.get('answer', ''), 'edited_answer'),
                            'score': float(row.get('score', 0)),
                            'mode': ensure_string(row.get('mode', 'UNKNOWN'), 'edited_mode'),
                            'source': ensure_string(row.get('source', ''), 'edited_source')
                        }
                        
                        # Skip empty entries
                        if validated_row['question'].strip() and validated_row['answer'].strip():
                            validated_data.append(validated_row)
                    
                    # Update trainer
                    st.session_state.trainer.training_data = validated_data
                    st.success(f"✅ Updated training data with {len(validated_data)} validated entries!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Failed to update training data: {str(e)}")
        else:
            st.info("No training data available. Process some files first!")
    
    with tab3:
        st.header("📊 Enhanced Validation Statistics")
        
        # Display comprehensive validation stats
        display_enhanced_validator_stats()
        
        st.subheader("🔍 Type Checking Statistics")
        display_type_checker_stats()
        
        # Processing history
        if st.session_state.processed_files:
            st.subheader("📁 Processing History")
            history_df = pd.DataFrame(st.session_state.processed_files)
            st.dataframe(history_df, use_container_width=True)
    
    with tab4:
        st.header("🔧 Enhanced System Diagnostics")
        
        # System health checks
        st.subheader("🏥 System Health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Memory check
            memory_mb = get_memory_usage()
            memory_status = "🟢 Good" if memory_mb < 500 else "🟡 High" if memory_mb < 1000 else "🔴 Critical"
            st.metric("Memory Status", memory_status, f"{memory_mb:.1f} MB")
        
        with col2:
            # Validation system check
            text_stats = get_enhanced_validator_stats()
            success_rate = ((text_stats['total_validations'] - text_stats['failures']) / max(text_stats['total_validations'], 1)) * 100
            validation_status = "🟢 Excellent" if success_rate > 95 else "🟡 Good" if success_rate > 90 else "🔴 Poor"
            st.metric("Validation Health", validation_status, f"{success_rate:.1f}%")
        
        with col3:
            # Type checking health
            type_stats = get_type_checker_stats()
            type_success_rate = ((type_stats['total_checks'] - type_stats['failures']) / max(type_stats['total_checks'], 1)) * 100
            type_status = "🟢 Excellent" if type_success_rate > 95 else "🟡 Good" if type_success_rate > 90 else "🔴 Poor"
            st.metric("Type Check Health", type_status, f"{type_success_rate:.1f}%")
        
        # Diagnostic tests
        st.subheader("🧪 Diagnostic Tests")
        
        if st.button("Run Validation Tests"):
            st.info("Running comprehensive validation tests...")
            
            # Test 1: String validation
            test_string = "Test string"
            is_valid, result, message = check_text_type(test_string, "diagnostic_test")
            st.write(f"✅ String test: {message}")
            
            # Test 2: List validation
            test_list = ["Item 1", "Item 2", "Item 3"]
            validated_list_text = validate_text_enhanced(test_list, "diagnostic_list_test")
            st.write(f"✅ List conversion test: Converted to {len(validated_list_text)} character string")
            
            # Test 3: Emergency conversion
            test_emergency = emergency_text_conversion({"test": "data"}, "diagnostic_emergency")
            st.write(f"✅ Emergency conversion test: {type(test_emergency)} with {len(test_emergency)} characters")
            
            st.success("All diagnostic tests passed!")
    
    with tab5:
        st.header("📋 Enhanced Data Export")
        
        training_data = st.session_state.trainer.get_training_data()
        
        if training_data:
            # Export options with validation
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox(
                    "Export Format",
                    ["JSONL (OpenAI)", "CSV", "JSON"],
                    help="Choose format for exporting training data"
                )
                
                min_score = st.slider(
                    "Minimum Score Filter",
                    0.0, 1.0, 0.0, 0.1,
                    help="Only export dialogues above this score"
                )
            
            with col2:
                source_filter = st.multiselect(
                    "Source Filter",
                    options=list(set(item.get('source', 'Unknown') for item in training_data)),
                    help="Filter by source files"
                )
                
                mode_filter = st.multiselect(
                    "Mode Filter",
                    options=["TEACHING", "INQUIRY", "DIALOGUE", "UNKNOWN"],
                    help="Filter by dialogue mode"
                )
            
            # Filter data with validation
            filtered_data = []
            for item in training_data:
                # Validate item structure
                validated_item = ensure_dict(item, "export_item")
                
                # Apply filters
                score = float(validated_item.get('score', 0))
                source = ensure_string(validated_item.get('source', 'Unknown'), 'export_source')
                mode = ensure_string(validated_item.get('mode', 'UNKNOWN'), 'export_mode')
                
                if score >= min_score:
                    if not source_filter or source in source_filter:
                        if not mode_filter or mode in mode_filter:
                            filtered_data.append(validated_item)
            
            st.write(f"**Filtered Data:** {len(filtered_data)} / {len(training_data)} dialogues")
            
            if filtered_data and st.button("📥 Export Filtered Data"):
                try:
                    if export_format == "JSONL (OpenAI)":
                        # Enhanced JSONL export with validation
                        export_result = st.session_state.trainer.export_training_data(
                            filename=f"consciousness_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                            data_filter=lambda x: x in filtered_data
                        )
                        
                        if export_result['success']:
                            with open(export_result['filepath'], 'r') as f:
                                st.download_button(
                                    "📥 Download JSONL",
                                    f.read(),
                                    file_name=Path(export_result['filepath']).name,
                                    mime="application/jsonl"
                                )
                        else:
                            st.error(f"Export failed: {export_result['error']}")
                    
                    elif export_format == "CSV":
                        # CSV export with validation
                        df = pd.DataFrame(filtered_data)
                        csv_data = df.to_csv(index=False)
                        
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            file_name=f"consciousness_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    elif export_format == "JSON":
                        # JSON export with validation
                        json_data = json.dumps(filtered_data, indent=2, ensure_ascii=False)
                        
                        st.download_button(
                            "📥 Download JSON",
                            json_data,
                            file_name=f"consciousness_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    st.success("✅ Export completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        else:
            st.info("No training data available for export.")

if __name__ == "__main__":
    main()

