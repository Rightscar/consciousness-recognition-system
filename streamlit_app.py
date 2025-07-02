"""
Consciousness Recognition System - Optimized Streamlit Interface

Enhanced with chunking, lazy loading, batch editing, file validation, and memory management.
Designed for processing spiritual texts and creating consciousness recognition AI training data.

Author: Consciousness Recognition System
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
# Import our modules
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
except ImportError:
    st.error("Required modules not found. Please ensure all modules are in the modules/ directory.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Consciousness Recognition System",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    if 'jsonl_manager' not in st.session_state:
        st.session_state.jsonl_manager = JSONLManager()
    
    if 'file_manager' not in st.session_state:
        st.session_state.file_manager = FileManager("./output")
    
    if 'current_dialogues' not in st.session_state:
        st.session_state.current_dialogues = []
    
    if 'filtered_dialogues' not in st.session_state:
        st.session_state.filtered_dialogues = []
    
    if 'marked_for_export' not in st.session_state:
        st.session_state.marked_for_export = set()
    
    if 'current_dialogue_index' not in st.session_state:
        st.session_state.current_dialogue_index = 0
    
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    
    if 'page_size' not in st.session_state:
        st.session_state.page_size = 50  # For pagination
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0


def validate_file_size(uploaded_file, max_size_mb: int = 50) -> Dict[str, Any]:
    """
    Validate uploaded file size and provide warnings.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_size_mb: Maximum recommended size in MB
        
    Returns:
        Dict with validation results
    """
    if uploaded_file is None:
        return {'valid': False, 'error': 'No file provided'}
    
    file_size = uploaded_file.size
    size_mb = file_size / (1024 * 1024)
    
    result = {
        'valid': True,
        'size_bytes': file_size,
        'size_mb': size_mb,
        'is_large': size_mb > max_size_mb,
        'warning': None,
        'recommendation': None
    }
    
    if size_mb > max_size_mb:
        result['warning'] = f"Large file detected ({size_mb:.1f}MB). Processing may be slow."
        
        if size_mb > 100:
            result['recommendation'] = "Consider splitting this PDF into smaller files for better performance."
        elif size_mb > max_size_mb:
            result['recommendation'] = "Processing will use chunking for better memory management."
    
    return result


def render_memory_management_sidebar():
    """Render memory management controls in sidebar."""
    st.sidebar.header("🧠 Memory Management")
    
    # Get current memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    st.sidebar.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    # Clear cache buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🧹 Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset All"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("System reset!")
            st.rerun()
    
    # Memory usage warning
    if memory_mb > 1000:  # 1GB
        st.sidebar.warning("⚠️ High memory usage detected. Consider clearing cache or resetting.")


def render_file_manager_sidebar():
    """Render file manager in sidebar."""
    st.sidebar.header("📁 File Manager")
    
    # Output directory configuration
    output_dir = st.sidebar.text_input(
        "Output Directory", 
        value=str(st.session_state.file_manager.base_directory),
        help="Directory for saving processed files"
    )
    
    if output_dir != str(st.session_state.file_manager.base_directory):
        st.session_state.file_manager = FileManager(output_dir)
    
    # List files
    files = st.session_state.file_manager.list_files()
    
    if files:
        st.sidebar.subheader("📄 Recent Files")
        
        for file_info in files[:5]:  # Show last 5 files
            with st.sidebar.expander(f"{file_info['name'][:20]}..."):
                st.write(f"**Size:** {file_info['size_human']}")
                st.write(f"**Modified:** {file_info['modified_human']}")
                
                if file_info['extension'] == '.jsonl':
                    st.write(f"**Entries:** {file_info.get('entry_count', 'Unknown')}")
                    if file_info.get('modes'):
                        st.write(f"**Modes:** {', '.join(file_info['modes'])}")
                
                # File actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Open", key=f"open_{file_info['name']}"):
                        if file_info['extension'] == '.jsonl':
                            # Load JSONL file in editor
                            result = st.session_state.jsonl_manager.load_jsonl(file_info['path'])
                            if result['success']:
                                st.session_state.current_dialogues = result['data']
                                st.session_state.filtered_dialogues = result['data'].copy()
                                st.session_state.current_page = 0  # Reset pagination
                                st.success(f"Loaded {len(result['data'])} dialogues")
                            else:
                                st.error(f"Error loading file: {result['error']}")
                
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{file_info['name']}"):
                        result = st.session_state.file_manager.delete_file(file_info['path'])
                        if result['success']:
                            st.success("File deleted")
                            st.rerun()
                        else:
                            st.error(f"Error: {result['error']}")
    else:
        st.sidebar.info("No files found in output directory")


def render_upload_tab():
    """Render the Upload tab with enhanced file validation."""
    st.header("📤 Upload & Extract")
    
    # Configuration section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Upload Spiritual Texts")
        
        # Show supported formats
        with st.expander("📋 Supported Formats", expanded=False):
            try:
                from modules.universal_intake_validator import UniversalIntakeValidator
                universal_validator = UniversalIntakeValidator()
                format_info = universal_validator.get_supported_formats_info()
                
                st.write("**Currently Available:**")
                for fmt, info in format_info.items():
                    if info['available']:
                        st.write(f"• **{fmt.upper()}** - {info['name']} (Max: {info['max_size_mb']:.0f}MB)")
                        st.write(f"  _{info['description']}_")
                
                unavailable = [fmt for fmt, info in format_info.items() if not info['available']]
                if unavailable:
                    st.write("**Requires Additional Dependencies:**")
                    for fmt in unavailable:
                        info = format_info[fmt]
                        st.write(f"• **{fmt.upper()}** - {info['name']} _(Install dependencies)_")
                        
            except ImportError:
                st.write("PDF, TXT, and other basic formats supported")
        
        uploaded_files = st.file_uploader(
            "Choose spiritual texts in any supported format",
            type=['pdf', 'epub', 'azw3', 'mobi', 'txt', 'docx', 'doc', 'rtf', 'html', 'htm'],
            accept_multiple_files=True,
            help="Upload spiritual texts from teachers like Nisargadatta, Ramana, Rupert Spira, etc."
        )
        
        # File validation
        if uploaded_files:
            st.subheader("📊 File Validation")
            for uploaded_file in uploaded_files:
                validation = validate_file_size(uploaded_file)
                
                col_name, col_size, col_status = st.columns([2, 1, 1])
                
                with col_name:
                    st.write(f"**{uploaded_file.name}**")
                
                with col_size:
                    st.write(f"{validation['size_mb']:.1f} MB")
                
                with col_status:
                    if validation['is_large']:
                        st.warning("⚠️ Large")
                    else:
                        st.success("✅ OK")
                
                if validation.get('warning'):
                    st.warning(validation['warning'])
                
                if validation.get('recommendation'):
                    st.info(validation['recommendation'])
    
    with col2:
        st.subheader("⚙️ Detection Settings")
        
        detection_mode = st.selectbox(
            "Detection Mode",
            ["Multi-Mode (Recommended)", "Auto (Regex + Semantic)", "Regex Only", "Semantic Only"],
            help="Multi-Mode handles both dialogues and non-dialogue content"
        )
        
        semantic_threshold = st.slider(
            "Semantic Threshold",
            0.1, 1.0, 0.4, 0.1,
            help="Minimum similarity to non-dual teachings"
        )
        
        score_threshold = st.slider(
            "Score Threshold", 
            0.1, 1.0, 0.7, 0.1,
            help="Minimum consciousness recognition score"
        )
        
        # Chunking settings for large files
        st.subheader("🔧 Processing Options")
        
        chunk_size = st.slider(
            "Pages per Chunk",
            5, 50, 20, 5,
            help="For large PDFs, process this many pages at a time"
        )
        
        use_chunking = st.checkbox(
            "Force Chunking",
            value=False,
            help="Use chunking even for small files (for testing)"
        )
    
    # Processing section
    if uploaded_files and st.button("🚀 Start Processing", type="primary"):
        
        # Initialize components with universal format support
        from modules.universal_extractor import UniversalTextExtractor
        from modules.universal_intake_validator import UniversalIntakeValidator
        
        extractor = UniversalTextExtractor()
        detector = EnhancedDialogueDetector()
        trainer = EnhancedOpenAITrainer()  # Use enhanced trainer
        intake_validator = UniversalIntakeValidator()  # Use universal validator
        
        all_results = []
        
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")
            
            # Step 1: Comprehensive intake validation
            st.info("🔍 Validating file...")
            validation_result = intake_validator.validate_upload(uploaded_file)
            
            if not validation_result.is_valid:
                st.error(f"❌ File validation failed: {validation_result.error_message}")
                if validation_result.security_warnings:
                    st.warning("⚠️ Security warnings:")
                    for warning in validation_result.security_warnings:
                        st.write(f"• {warning}")
                continue
            
            # Display validation results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("File Size", f"{validation_result.file_size / 1024 / 1024:.1f}MB")
            with col2:
                st.metric("Pages", validation_result.page_count)
            with col3:
                st.metric("Has Text", "✅" if validation_result.has_text else "❌")
            with col4:
                st.metric("Encrypted", "🔒" if validation_result.is_encrypted else "🔓")
            
            # Show processing recommendations
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
                # Step 2: Universal text extraction
                st.info("📖 Extracting text...")
                
                # Get file format info
                format_info = extractor.get_format_info(tmp_path)
                st.write(f"**Format:** {format_info['info']['name']} ({format_info['format'].upper()})")
                
                # Extract text using universal extractor
                extraction_result = extractor.extract_text(tmp_path)
                
                if not extraction_result['success']:
                    st.error(f"❌ Text extraction failed: {extraction_result['error']}")
                    continue
                
                text = extraction_result['text']
                
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
                    metadata = extraction_result['metadata']
                    st.info("📋 **Metadata:**")
                    for key, value in metadata.items():
                        if value and key not in ['extractor']:
                            st.write(f"• **{key.title()}:** {value}")
                
                # Debug: Check text type before detection
                if not isinstance(text, str):
                    st.error(f"❌ Text extraction returned {type(text)} instead of string. Content: {str(text)[:100]}...")
                    continue
                
                if len(text.strip()) < 100:
                    st.warning("⚠️ Very little text extracted. File may be image-based or corrupted.")
                    continue
                
                # Step 3: Dialogue detection
                st.info("🔍 Detecting dialogues...")
                
                # Define mode mapping
                mode_map = {
                    "Multi-Mode (Recommended)": "multi_mode",
                    "Auto (Regex + Semantic)": "auto",
                    "Regex Only": "regex", 
                    "Semantic Only": "semantic"
                }
                
                # Show content analysis for multi-mode
                if detection_mode == "Multi-Mode (Recommended)":
                    st.info("🔍 Analyzing content type...")
                    content_analyzer = ContentAnalyzer()
                    content_analysis = content_analyzer.analyze_content(text)
                    
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
                
                detection_result = detector.detect_dialogues_with_progress(
                    text,  # Use the extracted text directly, not extraction_result['text']
                    mode=mode_map[detection_mode],
                    semantic_threshold=semantic_threshold,
                    show_progress=True
                )
                
                if not detection_result['success']:
                    st.error(f"Failed to detect dialogues: {detection_result.get('error', 'Unknown error')}")
                    continue
                
                # Filter by score threshold
                high_score_dialogues = [
                    d for d in detection_result['dialogues'] 
                    if d.get('overall_score', 0) >= score_threshold
                ]
                
                # Add to trainer with validation
                for dialogue in high_score_dialogues:
                    add_result = trainer.add_dialogue(
                        question=dialogue['question'],
                        answer=dialogue['answer'],
                        score=dialogue.get('overall_score', 0),
                        mode=dialogue.get('mode', 'unknown'),
                        source=uploaded_file.name,
                        metadata=dialogue
                    )
                    
                    # Track validation results
                    if not add_result['success']:
                        st.warning(f"⚠️ Dialogue rejected: {add_result['issues'][0] if add_result['issues'] else 'Unknown reason'}")
                    elif add_result['warnings']:
                        st.info(f"ℹ️ Dialogue added with warnings: {len(add_result['warnings'])} warnings")
                
                # Store results
                file_result = {
                    'filename': uploaded_file.name,
                    'total_dialogues': len(detection_result['dialogues']),
                    'high_score_dialogues': len(high_score_dialogues),
                    'extraction_info': extraction_result,
                    'detection_info': detection_result
                }
                all_results.append(file_result)
                
                # Display results
                if detection_mode == "Multi-Mode (Recommended)" and 'content_analysis' in detection_result:
                    # Enhanced results for multi-mode
                    st.success(f"✅ **{uploaded_file.name}** processed successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Content", len(detection_result['dialogues']))
                    with col2:
                        st.metric("Traditional Q&A", detection_result.get('traditional_dialogues', 0))
                    with col3:
                        st.metric("From Passages", detection_result.get('synthetic_dialogues', 0))
                    with col4:
                        st.metric("High Score", len(high_score_dialogues))
                    
                    # Show content type and processing method
                    content_type = detection_result.get('content_type', 'unknown')
                    st.info(f"📖 **Content Type:** {content_type.replace('_', ' ').title()}")
                    
                else:
                    # Standard results display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Dialogues", len(detection_result['dialogues']))
                    with col2:
                        st.metric("High Score", len(high_score_dialogues))
                    with col3:
                        avg_score = sum(d.get('overall_score', 0) for d in detection_result['dialogues']) / max(1, len(detection_result['dialogues']))
                        st.metric("Avg Score", f"{avg_score:.2f}")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # Save results and update session state
        if all_results:
            st.session_state.processing_results = all_results
            st.session_state.current_dialogues = trainer.dialogues
            st.session_state.filtered_dialogues = trainer.dialogues.copy()
            st.session_state.current_page = 0  # Reset pagination
            
            # Export results with comprehensive validation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSONL with validation
            jsonl_path = f"./output/consciousness_dialogues_{timestamp}.jsonl"
            os.makedirs("./output", exist_ok=True)
            
            st.info("📤 Exporting with validation...")
            export_result = trainer.export_to_jsonl(jsonl_path, validate_output=True)
            
            if export_result['success']:
                st.success(f"✅ Exported {export_result['count']} dialogues to {jsonl_path}")
                
                # Display validation results
                if export_result.get('validation'):
                    validation = export_result['validation']
                    
                    # Validation metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Valid Items", validation.valid_items)
                    with col2:
                        st.metric("Invalid Items", validation.invalid_items)
                    with col3:
                        st.metric("Quality Score", f"{validation.quality_score:.2f}")
                    with col4:
                        st.metric("OpenAI Compliant", "✅" if validation.openai_compliance else "❌")
                    
                    # Show issues and warnings
                    if validation.issues:
                        st.error("🚨 **Validation Issues:**")
                        for issue in validation.issues[:5]:  # Show first 5
                            st.write(f"• {issue}")
                    
                    if validation.warnings:
                        st.warning("⚠️ **Validation Warnings:**")
                        for warning in validation.warnings[:5]:  # Show first 5
                            st.write(f"• {warning}")
                    
                    if validation.recommendations:
                        st.info("💡 **Recommendations:**")
                        for rec in validation.recommendations:
                            st.write(f"• {rec}")
                
                # Show quality report
                quality_report = trainer.get_quality_report()
                st.info(f"📊 **Quality Report:** {quality_report['acceptance_rate']:.1%} acceptance rate, {quality_report['rejected']} dialogues rejected")
                
                # Export rejected dialogues for analysis
                if quality_report['rejected'] > 0:
                    rejected_path = f"./output/rejected_dialogues_{timestamp}.jsonl"
                    rejected_result = trainer.export_rejected_dialogues(rejected_path)
                    if rejected_result['success']:
                        st.info(f"📋 Rejected dialogues saved to {rejected_path} for analysis")
                
            else:
                st.error(f"Export failed: {export_result.get('error', 'Unknown error')}")
                
                # Show validation details if available
                if export_result.get('validation'):
                    validation = export_result['validation']
                    st.error("❌ **Export failed validation:**")
                    for issue in validation.issues:
                        st.write(f"• {issue}")
            
            # Save CSV summary
            csv_path = f"./output/consciousness_summary_{timestamp}.csv"
            csv_result = trainer.export_to_csv(csv_path)
            
            if csv_result['success']:
                st.success(f"✅ Summary saved to {csv_path}")


def render_viewer_editor_tab():
    """Render the Viewer & Editor tab with pagination for large datasets."""
    st.header("👁️ Viewer & Editor")
    
    if not st.session_state.current_dialogues:
        st.info("No dialogues loaded. Please upload and process PDFs first, or load a JSONL file.")
        return
    
    # Search and filter section
    st.subheader("🔍 Search & Filter")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search_keyword = st.text_input("Keyword Search", help="Search in questions and answers")
    
    with col2:
        score_range = st.slider(
            "Score Range",
            0.0, 1.0, (0.0, 1.0), 0.1,
            help="Filter by consciousness recognition score"
        )
    
    with col3:
        mode_filter = st.selectbox(
            "Mode Filter",
            ["All", "consciousness", "inquiry", "teaching", "mixed"],
            help="Filter by dialogue mode"
        )
    
    with col4:
        source_filter = st.selectbox(
            "Source Filter",
            ["All"] + list(set(d.get('source', 'Unknown') for d in st.session_state.current_dialogues)),
            help="Filter by source file"
        )
    
    # Apply filters
    filtered = st.session_state.current_dialogues.copy()
    
    if search_keyword:
        filtered = [
            d for d in filtered 
            if search_keyword.lower() in d.get('question', '').lower() 
            or search_keyword.lower() in d.get('answer', '').lower()
        ]
    
    filtered = [
        d for d in filtered 
        if score_range[0] <= d.get('overall_score', 0) <= score_range[1]
    ]
    
    if mode_filter != "All":
        filtered = [d for d in filtered if d.get('mode') == mode_filter]
    
    if source_filter != "All":
        filtered = [d for d in filtered if d.get('source') == source_filter]
    
    st.session_state.filtered_dialogues = filtered
    
    # Pagination controls
    total_dialogues = len(filtered)
    if total_dialogues == 0:
        st.warning("No dialogues match the current filters.")
        return
    
    st.subheader(f"📊 Results ({total_dialogues} dialogues)")
    
    # Page size control
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        page_size = st.selectbox(
            "Items per page",
            [10, 25, 50, 100, 200],
            index=2,  # Default to 50
            key="page_size_selector"
        )
        st.session_state.page_size = page_size
    
    with col2:
        total_pages = (total_dialogues - 1) // page_size + 1
        current_page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.current_page + 1,
            key="page_selector"
        ) - 1
        st.session_state.current_page = current_page
    
    with col3:
        st.metric("Total Pages", total_pages)
    
    # Calculate page bounds
    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, total_dialogues)
    page_dialogues = filtered[start_idx:end_idx]
    
    # Display dialogues for current page
    st.subheader(f"Page {current_page + 1} (Items {start_idx + 1}-{end_idx})")
    
    # Batch actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Mark All on Page for Export"):
            for i, dialogue in enumerate(page_dialogues):
                dialogue_id = start_idx + i
                st.session_state.marked_for_export.add(dialogue_id)
            st.success(f"Marked {len(page_dialogues)} dialogues for export")
    
    with col2:
        if st.button("❌ Unmark All on Page"):
            for i, dialogue in enumerate(page_dialogues):
                dialogue_id = start_idx + i
                st.session_state.marked_for_export.discard(dialogue_id)
            st.success(f"Unmarked {len(page_dialogues)} dialogues")
    
    with col3:
        if st.button("🗑️ Delete All on Page"):
            # Remove from filtered list (this affects the original list)
            for dialogue in page_dialogues:
                if dialogue in st.session_state.current_dialogues:
                    st.session_state.current_dialogues.remove(dialogue)
            
            # Update filtered list
            st.session_state.filtered_dialogues = [
                d for d in st.session_state.filtered_dialogues 
                if d not in page_dialogues
            ]
            
            # Adjust current page if necessary
            new_total = len(st.session_state.filtered_dialogues)
            new_total_pages = (new_total - 1) // page_size + 1 if new_total > 0 else 1
            if current_page >= new_total_pages:
                st.session_state.current_page = max(0, new_total_pages - 1)
            
            st.success(f"Deleted {len(page_dialogues)} dialogues")
            st.rerun()
    
    # Display individual dialogues
    for i, dialogue in enumerate(page_dialogues):
        dialogue_id = start_idx + i
        
        with st.expander(
            f"Dialogue {dialogue_id + 1} | Score: {dialogue.get('overall_score', 0):.2f} | "
            f"Mode: {dialogue.get('mode', 'unknown')} | "
            f"{'✅' if dialogue_id in st.session_state.marked_for_export else '⭕'}"
        ):
            # Edit dialogue content
            col1, col2 = st.columns(2)
            
            with col1:
                new_question = st.text_area(
                    "Question",
                    value=dialogue.get('question', ''),
                    key=f"question_{dialogue_id}",
                    height=100
                )
            
            with col2:
                new_answer = st.text_area(
                    "Answer", 
                    value=dialogue.get('answer', ''),
                    key=f"answer_{dialogue_id}",
                    height=100
                )
            
            # Metadata editing
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_score = st.number_input(
                    "Score",
                    0.0, 1.0,
                    value=dialogue.get('overall_score', 0),
                    step=0.1,
                    key=f"score_{dialogue_id}"
                )
            
            with col2:
                new_mode = st.selectbox(
                    "Mode",
                    ["consciousness", "inquiry", "teaching", "mixed"],
                    index=["consciousness", "inquiry", "teaching", "mixed"].index(
                        dialogue.get('mode', 'consciousness')
                    ),
                    key=f"mode_{dialogue_id}"
                )
            
            with col3:
                new_source = st.text_input(
                    "Source",
                    value=dialogue.get('source', ''),
                    key=f"source_{dialogue_id}"
                )
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Changes", key=f"save_{dialogue_id}"):
                    # Update dialogue in place
                    dialogue['question'] = new_question
                    dialogue['answer'] = new_answer
                    dialogue['overall_score'] = new_score
                    dialogue['mode'] = new_mode
                    dialogue['source'] = new_source
                    st.success("Changes saved!")
            
            with col2:
                if dialogue_id in st.session_state.marked_for_export:
                    if st.button("❌ Unmark", key=f"unmark_{dialogue_id}"):
                        st.session_state.marked_for_export.discard(dialogue_id)
                        st.rerun()
                else:
                    if st.button("✅ Mark for Export", key=f"mark_{dialogue_id}"):
                        st.session_state.marked_for_export.add(dialogue_id)
                        st.rerun()
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{dialogue_id}"):
                    # Remove from both lists
                    if dialogue in st.session_state.current_dialogues:
                        st.session_state.current_dialogues.remove(dialogue)
                    if dialogue in st.session_state.filtered_dialogues:
                        st.session_state.filtered_dialogues.remove(dialogue)
                    
                    # Remove from marked set
                    st.session_state.marked_for_export.discard(dialogue_id)
                    
                    st.success("Dialogue deleted!")
                    st.rerun()
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_page > 0:
            if st.button("⬅️ Previous Page"):
                st.session_state.current_page = current_page - 1
                st.rerun()
    
    with col3:
        if current_page < total_pages - 1:
            if st.button("➡️ Next Page"):
                st.session_state.current_page = current_page + 1
                st.rerun()
    
    # Export marked dialogues
    if st.session_state.marked_for_export:
        st.subheader("📤 Export Marked Dialogues")
        
        marked_count = len(st.session_state.marked_for_export)
        st.info(f"{marked_count} dialogues marked for export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Export to JSONL"):
                marked_dialogues = [
                    st.session_state.current_dialogues[i] 
                    for i in st.session_state.marked_for_export
                    if i < len(st.session_state.current_dialogues)
                ]
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"./output/marked_dialogues_{timestamp}.jsonl"
                
                # Create trainer and export
                trainer = OpenAITrainer()
                for dialogue in marked_dialogues:
                    trainer.add_dialogue(
                        question=dialogue['question'],
                        answer=dialogue['answer'],
                        score=dialogue.get('overall_score', 0),
                        mode=dialogue.get('mode', 'unknown'),
                        source=dialogue.get('source', 'manual'),
                        metadata=dialogue
                    )
                
                result = trainer.export_to_jsonl(export_path)
                
                if result['success']:
                    st.success(f"✅ Exported {result['count']} dialogues to {export_path}")
                else:
                    st.error(f"Export failed: {result.get('error', 'Unknown error')}")
        
        with col2:
            if st.button("🧹 Clear All Marks"):
                st.session_state.marked_for_export.clear()
                st.success("All marks cleared!")
                st.rerun()


def render_merge_jsonl_tab():
    """Render the Merge JSONL Files tab with enhanced filtering."""
    st.header("🔗 Merge JSONL Files")
    
    # File selection section
    st.subheader("📁 Select Files to Merge")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Upload JSONL Files:**")
        uploaded_jsonl_files = st.file_uploader(
            "Choose JSONL files",
            type=['jsonl'],
            accept_multiple_files=True,
            help="Upload multiple JSONL files to merge"
        )
    
    with col2:
        st.write("**Select from Output Folder:**")
        
        # List available JSONL files
        try:
            output_files = list(Path("./output").glob("*.jsonl"))
            if output_files:
                selected_files = st.multiselect(
                    "Available JSONL files",
                    options=[str(f) for f in output_files],
                    help="Select files from the output directory"
                )
            else:
                st.info("No JSONL files found in output directory")
                selected_files = []
        except:
            st.warning("Output directory not found")
            selected_files = []
    
    # Combine file sources
    all_files = []
    
    # Add uploaded files
    if uploaded_jsonl_files:
        for uploaded_file in uploaded_jsonl_files:
            # Save temporarily
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.jsonl', delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                all_files.append({
                    'name': uploaded_file.name,
                    'path': tmp_file.name,
                    'source': 'uploaded',
                    'temp': True
                })
    
    # Add selected files
    for file_path in selected_files:
        all_files.append({
            'name': Path(file_path).name,
            'path': file_path,
            'source': 'local',
            'temp': False
        })
    
    if not all_files:
        st.info("Please select or upload JSONL files to merge.")
        return
    
    # Load and preview files
    st.subheader("📊 File Preview")
    
    all_dialogues = []
    file_stats = []
    
    for file_info in all_files:
        try:
            result = st.session_state.jsonl_manager.load_jsonl(file_info['path'])
            
            if result['success']:
                dialogues = result['data']
                
                # Add source information to each dialogue
                for dialogue in dialogues:
                    dialogue['merge_source'] = file_info['name']
                
                all_dialogues.extend(dialogues)
                
                # Calculate stats
                modes = [d.get('mode', 'unknown') for d in dialogues]
                scores = [d.get('overall_score', 0) for d in dialogues]
                
                file_stats.append({
                    'File': file_info['name'],
                    'Dialogues': len(dialogues),
                    'Avg Score': f"{sum(scores) / len(scores):.2f}" if scores else "0.00",
                    'Modes': ', '.join(set(modes)),
                    'Source': file_info['source']
                })
            else:
                st.error(f"Failed to load {file_info['name']}: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error processing {file_info['name']}: {str(e)}")
    
    if file_stats:
        st.dataframe(pd.DataFrame(file_stats), use_container_width=True)
    
    if not all_dialogues:
        st.warning("No valid dialogues found in selected files.")
        return
    
    # Filtering section
    st.subheader("🔍 Apply Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_threshold = st.slider(
            "Minimum Score",
            0.0, 1.0, 0.7, 0.1,
            help="Filter dialogues by minimum consciousness score"
        )
    
    with col2:
        mode_selection = st.multiselect(
            "Select Modes",
            options=["consciousness", "inquiry", "teaching", "mixed"],
            default=["consciousness", "inquiry"],
            help="Include dialogues with these modes"
        )
    
    with col3:
        source_selection = st.multiselect(
            "Select Sources",
            options=list(set(d.get('merge_source', 'Unknown') for d in all_dialogues)),
            default=list(set(d.get('merge_source', 'Unknown') for d in all_dialogues)),
            help="Include dialogues from these files"
        )
    
    with col4:
        max_dialogues = st.number_input(
            "Max Dialogues",
            min_value=1,
            max_value=len(all_dialogues),
            value=min(1000, len(all_dialogues)),
            help="Maximum number of dialogues to include"
        )
    
    # Apply filters
    filtered_dialogues = all_dialogues.copy()
    
    # Score filter
    filtered_dialogues = [
        d for d in filtered_dialogues 
        if d.get('overall_score', 0) >= score_threshold
    ]
    
    # Mode filter
    if mode_selection:
        filtered_dialogues = [
            d for d in filtered_dialogues 
            if d.get('mode', 'unknown') in mode_selection
        ]
    
    # Source filter
    if source_selection:
        filtered_dialogues = [
            d for d in filtered_dialogues 
            if d.get('merge_source', 'Unknown') in source_selection
        ]
    
    # Limit number
    if len(filtered_dialogues) > max_dialogues:
        # Sort by score and take top dialogues
        filtered_dialogues.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        filtered_dialogues = filtered_dialogues[:max_dialogues]
    
    # Preview merged dataset
    st.subheader("📈 Merged Dataset Preview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Dialogues", len(filtered_dialogues))
    
    with col2:
        avg_score = sum(d.get('overall_score', 0) for d in filtered_dialogues) / max(1, len(filtered_dialogues))
        st.metric("Average Score", f"{avg_score:.2f}")
    
    with col3:
        modes = [d.get('mode', 'unknown') for d in filtered_dialogues]
        mode_counts = {mode: modes.count(mode) for mode in set(modes)}
        st.metric("Unique Modes", len(mode_counts))
    
    with col4:
        sources = [d.get('merge_source', 'Unknown') for d in filtered_dialogues]
        source_counts = {source: sources.count(source) for source in set(sources)}
        st.metric("Source Files", len(source_counts))
    
    # Distribution charts
    if filtered_dialogues:
        col1, col2 = st.columns(2)
        
        with col1:
            # Mode distribution
            mode_df = pd.DataFrame(list(mode_counts.items()), columns=['Mode', 'Count'])
            fig_mode = px.pie(mode_df, values='Count', names='Mode', title="Mode Distribution")
            st.plotly_chart(fig_mode, use_container_width=True)
        
        with col2:
            # Source distribution
            source_df = pd.DataFrame(list(source_counts.items()), columns=['Source', 'Count'])
            fig_source = px.bar(source_df, x='Source', y='Count', title="Source Distribution")
            st.plotly_chart(fig_source, use_container_width=True)
        
        # Score distribution
        scores = [d.get('overall_score', 0) for d in filtered_dialogues]
        fig_scores = px.histogram(
            x=scores, 
            nbins=20, 
            title="Score Distribution",
            labels={'x': 'Consciousness Recognition Score', 'y': 'Count'}
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # Export section
    st.subheader("💾 Export Merged Dataset")
    
    if filtered_dialogues:
        col1, col2 = st.columns(2)
        
        with col1:
            export_filename = st.text_input(
                "Export Filename",
                value="final_openai_dataset.jsonl",
                help="Name for the merged dataset file"
            )
        
        with col2:
            if st.button("🚀 Export Merged Dataset", type="primary"):
                try:
                    # Create trainer and add dialogues
                    trainer = OpenAITrainer()
                    
                    for dialogue in filtered_dialogues:
                        trainer.add_dialogue(
                            question=dialogue.get('question', ''),
                            answer=dialogue.get('answer', ''),
                            score=dialogue.get('overall_score', 0),
                            mode=dialogue.get('mode', 'unknown'),
                            source=dialogue.get('merge_source', 'merged'),
                            metadata=dialogue
                        )
                    
                    # Export
                    export_path = f"./output/{export_filename}"
                    os.makedirs("./output", exist_ok=True)
                    
                    result = trainer.export_to_jsonl(export_path)
                    
                    if result['success']:
                        st.success(f"✅ Successfully exported {result['count']} dialogues to {export_path}")
                        
                        # Provide download link
                        with open(export_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Merged Dataset",
                                data=f.read(),
                                file_name=export_filename,
                                mime="application/json"
                            )
                    else:
                        st.error(f"Export failed: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"Error during export: {str(e)}")
    
    else:
        st.warning("No dialogues match the current filters. Adjust your filter settings.")
    
    # Cleanup temporary files
    for file_info in all_files:
        if file_info.get('temp', False):
            try:
                os.unlink(file_info['path'])
            except:
                pass


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar components
    render_memory_management_sidebar()
    render_file_manager_sidebar()
    
    # Main content area
    st.title("🧘 Consciousness Recognition System")
    st.markdown("*Enhanced with chunking, lazy loading, and batch processing*")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload", 
        "👁️ Viewer & Editor", 
        "🔗 Merge JSONL",
        "📊 Export"
    ])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_viewer_editor_tab()
    
    with tab3:
        render_merge_jsonl_tab()
    
    with tab4:
        st.header("📊 Export & Analytics")
        st.info("Export functionality is integrated into other tabs. Use the Viewer & Editor tab to export marked dialogues, or the Merge JSONL tab to create final datasets.")
        
        # Show current session statistics
        if st.session_state.current_dialogues:
            st.subheader("📈 Current Session Statistics")
            
            total_dialogues = len(st.session_state.current_dialogues)
            marked_count = len(st.session_state.marked_for_export)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Dialogues", total_dialogues)
            
            with col2:
                st.metric("Marked for Export", marked_count)
            
            with col3:
                if total_dialogues > 0:
                    avg_score = sum(d.get('overall_score', 0) for d in st.session_state.current_dialogues) / total_dialogues
                    st.metric("Average Score", f"{avg_score:.2f}")
                else:
                    st.metric("Average Score", "0.00")
            
            # Mode distribution
            modes = [d.get('mode', 'unknown') for d in st.session_state.current_dialogues]
            mode_counts = {mode: modes.count(mode) for mode in set(modes)}
            
            if mode_counts:
                mode_df = pd.DataFrame(list(mode_counts.items()), columns=['Mode', 'Count'])
                fig = px.pie(mode_df, values='Count', names='Mode', title="Mode Distribution")
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

