"""
Health Check Endpoint for Render.com Deployment
Validates all system components and dependencies
"""

import os
import sys
import logging
import streamlit as st
from pathlib import Path

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def run_health_check():
    """Run comprehensive health check"""
    
    st.set_page_config(
        page_title="Health Check",
        page_icon="🩺",
        layout="centered"
    )
    
    st.title("🩺 System Health Check")
    st.markdown("*Validating Fine-Tune Data System Components*")
    
    health_status = True
    checks = []
    
    # 1. Check Python environment
    try:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        checks.append(("Python Version", f"✅ {python_version}", True))
    except Exception as e:
        checks.append(("Python Version", f"❌ {str(e)}", False))
        health_status = False
    
    # 2. Check core dependencies
    core_deps = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('requests', 'requests'),
        ('openai', 'openai')
    ]
    
    for name, module in core_deps:
        try:
            __import__(module)
            checks.append((f"{name}", "✅ Available", True))
        except ImportError:
            checks.append((f"{name}", "❌ Missing", False))
            health_status = False
    
    # 3. Check OCR dependencies
    ocr_deps = [
        ('Tesseract OCR', 'pytesseract'),
        ('PDF2Image', 'pdf2image'),
        ('PIL/Pillow', 'PIL')
    ]
    
    for name, module in ocr_deps:
        try:
            __import__(module)
            checks.append((f"{name}", "✅ Available", True))
        except ImportError:
            checks.append((f"{name}", "⚠️ Optional", True))  # OCR is optional
    
    # 4. Check system resources
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        memory_gb = memory.total / (1024**3)
        disk_gb = disk.free / (1024**3)
        
        checks.append(("Memory", f"✅ {memory_gb:.1f} GB available", True))
        checks.append(("Disk Space", f"✅ {disk_gb:.1f} GB free", True))
        
        if memory_gb < 0.4:  # Less than 400MB
            checks.append(("Memory Warning", "⚠️ Low memory detected", True))
        
    except Exception as e:
        checks.append(("System Resources", f"⚠️ Cannot check: {str(e)}", True))
    
    # 5. Check environment variables
    env_vars = [
        'OPENAI_API_KEY',
        'LOW_MEM_MODE',
        'DEBUG_MODE'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            masked_value = value[:8] + "..." if len(value) > 8 else value
            checks.append((f"ENV: {var}", f"✅ Set ({masked_value})", True))
        else:
            status = "⚠️ Not set" if var != 'OPENAI_API_KEY' else "❌ Required"
            is_ok = var != 'OPENAI_API_KEY'
            checks.append((f"ENV: {var}", status, is_ok))
            if not is_ok:
                health_status = False
    
    # 6. Check module imports
    try:
        from large_file_ocr_handler import LargeFileOCRHandler
        checks.append(("Large File OCR Handler", "✅ Available", True))
    except ImportError as e:
        checks.append(("Large File OCR Handler", f"❌ {str(e)}", False))
        health_status = False
    
    try:
        from input_validation import InputValidator
        checks.append(("Input Validation", "✅ Available", True))
    except ImportError as e:
        checks.append(("Input Validation", f"❌ {str(e)}", False))
        health_status = False
    
    # Display results
    st.subheader("🔍 Health Check Results")
    
    for check_name, status, is_ok in checks:
        if is_ok:
            st.success(f"**{check_name}:** {status}")
        else:
            st.error(f"**{check_name}:** {status}")
    
    # Overall status
    st.subheader("📊 Overall Status")
    
    if health_status:
        st.success("🎉 All critical systems operational!")
        st.balloons()
    else:
        st.error("❌ System has critical issues that need attention")
    
    # System info
    with st.expander("📋 System Information"):
        st.write(f"**Python:** {sys.version}")
        st.write(f"**Platform:** {sys.platform}")
        st.write(f"**Working Directory:** {os.getcwd()}")
        st.write(f"**Environment Variables:** {len(os.environ)} total")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Check"):
            st.rerun()
    
    with col2:
        if st.button("📱 Go to Main App"):
            st.switch_page("app.py")
    
    with col3:
        st.write("**Status:** Ready for deployment" if health_status else "**Status:** Needs attention")

if __name__ == "__main__":
    run_health_check()

