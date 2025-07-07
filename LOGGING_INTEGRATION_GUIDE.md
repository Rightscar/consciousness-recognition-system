# 📊 Logging Integration Guide

## 🎯 **Complete Integration of Centralized Logging System**

This guide shows how to integrate the centralized logging system into all application modules for comprehensive error and event tracking with timestamps.

---

## 📋 **Quick Start Integration**

### **1. Basic Module Integration**

```python
# In any module (e.g., extractor.py, enhancer.py, validator.py)
from modules.logger import get_logger, log_event, log_performance, log_error

# Get module-specific logger
logger = get_logger("extractor")  # Replace with your module name

# Basic logging
logger.info("Content extraction started")
logger.warning("Large file detected, processing may take longer")
logger.error("Failed to extract content from file")

# Event logging
log_event("file_uploaded", {"filename": "data.txt", "size": 1024}, "extractor")

# Performance logging
log_performance("content_extraction", 5.2, {"items": 100}, "extractor")

# Error logging with context
try:
    risky_operation()
except Exception as e:
    log_error(e, {"operation": "file_processing", "file": "data.txt"}, "extractor")
```

### **2. Class-Based Integration**

```python
from modules.logger import ModuleLoggerMixin, log_timing

class ContentExtractor(ModuleLoggerMixin):
    def __init__(self):
        super().__init__()
        self.setup_module_logging("extractor")
    
    @log_timing("content_extraction", "extractor")
    def extract_content(self, file_path):
        self.log_operation_start("content_extraction", {"file": file_path})
        
        try:
            # Your extraction logic here
            content = self.perform_extraction(file_path)
            
            self.log_operation_success("content_extraction", 3.2, {
                "file": file_path,
                "content_length": len(content)
            })
            
            return content
            
        except Exception as e:
            self.log_operation_error("content_extraction", e, {"file": file_path})
            raise
```

### **3. Context Manager Integration**

```python
from modules.logger import LoggedOperation

def process_file(file_path):
    with LoggedOperation("file_processing", "processor") as op:
        op.add_detail("file_path", file_path)
        
        # Step 1
        op.log_progress("Reading file")
        content = read_file(file_path)
        op.add_detail("file_size", len(content))
        
        # Step 2
        op.log_progress("Processing content")
        processed = process_content(content)
        op.add_detail("processed_items", len(processed))
        
        return processed
```

---

## 🔧 **Module-Specific Integration Examples**

### **Content Extractor Module**

```python
# modules/enhanced_universal_extractor.py
from modules.logger import get_logger, log_event, log_performance, log_error, log_timing
import time

class EnhancedUniversalExtractor:
    def __init__(self):
        self.logger = get_logger("extractor")
        self.logger.info("Enhanced Universal Extractor initialized")
    
    @log_timing("text_extraction", "extractor")
    def extract_text_content(self, file_path, file_type):
        """Extract text content with comprehensive logging"""
        
        self.logger.info(f"Starting text extraction from {file_path} (type: {file_type})")
        
        # Log extraction start event
        log_event("extraction_started", {
            "file_path": file_path,
            "file_type": file_type,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }, "extractor")
        
        try:
            start_time = time.time()
            
            if file_type == "txt":
                content = self.extract_txt(file_path)
            elif file_type == "pdf":
                content = self.extract_pdf(file_path)
            elif file_type == "docx":
                content = self.extract_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            duration = time.time() - start_time
            
            # Log successful extraction
            self.logger.info(f"Successfully extracted {len(content)} characters in {duration:.2f}s")
            
            log_event("extraction_completed", {
                "file_path": file_path,
                "content_length": len(content),
                "success": True
            }, "extractor")
            
            log_performance("text_extraction", duration, {
                "file_type": file_type,
                "content_length": len(content),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }, "extractor")
            
            return content
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {str(e)}")
            
            log_error(e, {
                "operation": "text_extraction",
                "file_path": file_path,
                "file_type": file_type
            }, "extractor")
            
            log_event("extraction_failed", {
                "file_path": file_path,
                "error": str(e),
                "success": False
            }, "extractor")
            
            raise
    
    def extract_pdf(self, file_path):
        """Extract PDF content with detailed logging"""
        
        self.logger.debug(f"Attempting PDF extraction from {file_path}")
        
        try:
            # Your PDF extraction logic
            content = "extracted content"
            
            self.logger.debug(f"PDF extraction successful, {len(content)} characters extracted")
            return content
            
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {str(e)}")
            raise
```

### **Content Enhancer Module**

```python
# modules/enhanced_custom_prompt_engine.py
from modules.logger import get_logger, log_ai_operation, log_event, log_error
import time

class EnhancedCustomPromptEngine:
    def __init__(self):
        self.logger = get_logger("enhancer")
        self.logger.info("Enhanced Custom Prompt Engine initialized")
    
    def enhance_content(self, content, tone, custom_instructions):
        """Enhance content with AI and comprehensive logging"""
        
        self.logger.info(f"Starting content enhancement with tone: {tone}")
        
        # Log enhancement start
        log_event("enhancement_started", {
            "content_length": len(content),
            "tone": tone,
            "has_custom_instructions": bool(custom_instructions)
        }, "enhancer")
        
        try:
            start_time = time.time()
            
            # Prepare prompt
            prompt = self.build_enhancement_prompt(content, tone, custom_instructions)
            self.logger.debug(f"Built enhancement prompt, length: {len(prompt)}")
            
            # Call AI API
            enhanced_content = self.call_ai_api(prompt)
            
            duration = time.time() - start_time
            
            # Log AI operation
            log_ai_operation(
                operation="content_enhancement",
                input_size=len(content),
                output_size=len(enhanced_content),
                duration=duration,
                success=True
            )
            
            # Log enhancement completion
            self.logger.info(f"Content enhancement completed in {duration:.2f}s")
            self.logger.info(f"Enhanced content length: {len(enhanced_content)} characters")
            
            log_event("enhancement_completed", {
                "original_length": len(content),
                "enhanced_length": len(enhanced_content),
                "tone": tone,
                "success": True
            }, "enhancer")
            
            return enhanced_content
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.logger.error(f"Content enhancement failed after {duration:.2f}s: {str(e)}")
            
            # Log failed AI operation
            log_ai_operation(
                operation="content_enhancement",
                input_size=len(content),
                output_size=0,
                duration=duration,
                success=False
            )
            
            log_error(e, {
                "operation": "content_enhancement",
                "content_length": len(content),
                "tone": tone
            }, "enhancer")
            
            log_event("enhancement_failed", {
                "content_length": len(content),
                "tone": tone,
                "error": str(e),
                "success": False
            }, "enhancer")
            
            raise
```

### **Quality Validator Module**

```python
# modules/quality_threshold_handler.py
from modules.logger import get_logger, log_event, log_performance, log_error

class QualityThresholdHandler:
    def __init__(self):
        self.logger = get_logger("validator")
        self.logger.info("Quality Threshold Handler initialized")
    
    def validate_content_quality(self, original_content, enhanced_content):
        """Validate content quality with comprehensive logging"""
        
        self.logger.info("Starting content quality validation")
        
        try:
            start_time = time.time()
            
            # Calculate quality metrics
            coherence_score = self.calculate_coherence(original_content, enhanced_content)
            length_ratio = len(enhanced_content) / len(original_content) if original_content else 0
            
            # Determine quality status
            quality_passed = coherence_score >= 0.75 and length_ratio <= 1.8
            
            duration = time.time() - start_time
            
            # Log validation results
            self.logger.info(f"Quality validation completed in {duration:.2f}s")
            self.logger.info(f"Coherence score: {coherence_score:.3f}")
            self.logger.info(f"Length ratio: {length_ratio:.2f}")
            self.logger.info(f"Quality check: {'PASSED' if quality_passed else 'FAILED'}")
            
            # Log validation event
            log_event("quality_validation", {
                "coherence_score": coherence_score,
                "length_ratio": length_ratio,
                "quality_passed": quality_passed,
                "original_length": len(original_content),
                "enhanced_length": len(enhanced_content)
            }, "validator")
            
            # Log performance
            log_performance("quality_validation", duration, {
                "coherence_score": coherence_score,
                "length_ratio": length_ratio
            }, "validator")
            
            if not quality_passed:
                self.logger.warning("Content failed quality validation - flagged for manual review")
                
                log_event("quality_failure", {
                    "reason": "coherence_or_length_threshold",
                    "coherence_score": coherence_score,
                    "length_ratio": length_ratio
                }, "validator")
            
            return {
                "quality_passed": quality_passed,
                "coherence_score": coherence_score,
                "length_ratio": length_ratio,
                "requires_manual_review": not quality_passed
            }
            
        except Exception as e:
            self.logger.error(f"Quality validation failed: {str(e)}")
            
            log_error(e, {
                "operation": "quality_validation",
                "original_length": len(original_content),
                "enhanced_length": len(enhanced_content)
            }, "validator")
            
            raise
```

### **Export Utilities Module**

```python
# modules/export_utils.py
from modules.logger import get_logger, log_event, log_performance, log_error

class RobustExportManager:
    def __init__(self):
        self.logger = get_logger("export")
        self.logger.info("Robust Export Manager initialized")
    
    def export_with_validation(self, data, format_type, output_path):
        """Export data with comprehensive logging"""
        
        self.logger.info(f"Starting export to {format_type} format: {output_path}")
        
        # Log export start
        log_event("export_started", {
            "format": format_type,
            "output_path": output_path,
            "data_items": len(data) if isinstance(data, list) else 1
        }, "export")
        
        try:
            start_time = time.time()
            
            # Validate data
            if not data:
                raise ValueError("No data to export")
            
            # Perform export
            if format_type == "json":
                success = self.export_json(data, output_path)
            elif format_type == "jsonl":
                success = self.export_jsonl(data, output_path)
            elif format_type == "csv":
                success = self.export_csv(data, output_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            duration = time.time() - start_time
            
            if success:
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                
                self.logger.info(f"Export completed successfully in {duration:.2f}s")
                self.logger.info(f"Output file size: {file_size:,} bytes")
                
                # Log successful export
                log_event("export_completed", {
                    "format": format_type,
                    "output_path": output_path,
                    "file_size": file_size,
                    "success": True
                }, "export")
                
                log_performance("export_operation", duration, {
                    "format": format_type,
                    "data_items": len(data) if isinstance(data, list) else 1,
                    "file_size": file_size
                }, "export")
            
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.logger.error(f"Export failed after {duration:.2f}s: {str(e)}")
            
            log_error(e, {
                "operation": "export",
                "format": format_type,
                "output_path": output_path,
                "data_items": len(data) if isinstance(data, list) else 1
            }, "export")
            
            log_event("export_failed", {
                "format": format_type,
                "output_path": output_path,
                "error": str(e),
                "success": False
            }, "export")
            
            return False
```

### **UI/Streamlit Integration**

```python
# enhanced_app.py
from modules.logger import get_logger, log_user_action, log_event
from modules.log_viewer import render_compact_log_viewer, render_log_viewer

class EnhancedUniversalAITrainer:
    def __init__(self):
        self.logger = get_logger("ui")
        self.logger.info("Enhanced Universal AI Trainer UI initialized")
    
    def render_main_interface(self):
        """Render main interface with logging"""
        
        st.title("🧠 Enhanced Universal AI Training Data Creator")
        
        # Log page view
        log_user_action("page_view", {"page": "main"})
        
        # Sidebar with compact log viewer
        with st.sidebar:
            render_compact_log_viewer()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Upload", "✨ Enhance", "📋 Review", "📦 Export", "📊 Logs"
        ])
        
        with tab1:
            self.render_upload_tab()
        
        with tab2:
            self.render_enhance_tab()
        
        with tab3:
            self.render_review_tab()
        
        with tab4:
            self.render_export_tab()
        
        with tab5:
            # Full log viewer
            render_log_viewer()
    
    def render_upload_tab(self):
        """Render upload tab with logging"""
        
        st.subheader("📁 File Upload")
        
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])
        
        if uploaded_file:
            # Log file upload
            log_user_action("file_uploaded", {
                "filename": uploaded_file.name,
                "file_type": uploaded_file.type,
                "file_size": uploaded_file.size
            })
            
            self.logger.info(f"File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            st.success("✅ File uploaded successfully!")
    
    def handle_button_click(self, button_name, action_data=None):
        """Handle button clicks with logging"""
        
        # Log user action
        log_user_action("button_clicked", {
            "button": button_name,
            **(action_data or {})
        })
        
        self.logger.info(f"User clicked button: {button_name}")
```

---

## 📊 **Log Viewer Integration**

### **In Main App**

```python
# Add to your main app
from modules.log_viewer import render_log_viewer, render_compact_log_viewer

# In sidebar for quick monitoring
with st.sidebar:
    st.markdown("---")
    render_compact_log_viewer()

# In dedicated tab for full log analysis
with st.tabs(["Main", "Logs"])[1]:
    render_log_viewer()
```

### **Standalone Log Viewer**

```python
# Create standalone log viewer app
import streamlit as st
from modules.log_viewer import render_log_viewer

st.set_page_config(
    page_title="System Logs",
    page_icon="📊",
    layout="wide"
)

st.title("📊 System Logs & Monitoring")
render_log_viewer()
```

---

## 🎯 **Best Practices**

### **1. Consistent Module Naming**
```python
# Use descriptive module names
logger = get_logger("extractor")      # For content extraction
logger = get_logger("enhancer")       # For AI enhancement
logger = get_logger("validator")      # For quality validation
logger = get_logger("export")         # For data export
logger = get_logger("ui")             # For UI interactions
```

### **2. Structured Event Logging**
```python
# Always include relevant context
log_event("operation_name", {
    "input_size": len(data),
    "parameters": {"param1": value1},
    "success": True,
    "duration": 2.5
}, "module_name")
```

### **3. Performance Tracking**
```python
# Track all significant operations
@log_timing("operation_name", "module_name")
def expensive_operation():
    # implementation
    pass

# Or manual timing
start_time = time.time()
# ... operation ...
duration = time.time() - start_time
log_performance("operation_name", duration, details, "module_name")
```

### **4. Error Context**
```python
# Always provide context for errors
try:
    risky_operation(param1, param2)
except Exception as e:
    log_error(e, {
        "operation": "risky_operation",
        "param1": param1,
        "param2": param2,
        "additional_context": "any relevant info"
    }, "module_name")
    raise
```

---

## 📈 **Monitoring & Analytics**

### **Real-time Monitoring**
- Use the integrated log viewer for real-time monitoring
- Enable auto-refresh for live log streaming
- Monitor error rates and performance metrics

### **Performance Analysis**
- Track operation durations and identify bottlenecks
- Monitor AI operation success rates
- Analyze user interaction patterns

### **Error Analysis**
- Review error patterns and frequencies
- Identify common failure points
- Track error resolution effectiveness

---

## 🚀 **Production Deployment**

### **Log Rotation**
- Logs automatically rotate when they reach size limits
- Old logs are archived and cleaned up automatically
- Configure retention policies in log viewer settings

### **Monitoring Integration**
- Export logs for external monitoring systems
- Set up alerts based on error rates
- Monitor system health through log analytics

### **Performance Optimization**
- Use appropriate log levels for production
- Monitor log file sizes and performance impact
- Optimize logging frequency for critical operations

---

This comprehensive logging integration provides complete visibility into your application's behavior, making debugging, monitoring, and optimization much more effective! 🎉

