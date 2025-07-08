#!/bin/bash

# Health Check Script for Fine-Tune Data System
# =============================================

echo "🔍 Running health check for Fine-Tune Data System..."

# Check Python version
echo "📋 Checking Python version..."
python3 --version || exit 1

# Check if we're in the right directory
if [ ! -f "enhanced_app_production_optimized.py" ]; then
    echo "❌ Main application file not found"
    exit 1
fi

# Test Python imports
echo "📦 Testing Python imports..."
python3 -c "
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

try:
    # Test core imports
    import streamlit
    import pandas
    import numpy
    import pydantic
    print('✅ Core dependencies imported successfully')
    
    # Test OCR dependencies (optional)
    try:
        import pytesseract
        import pdf2image
        import fitz  # PyMuPDF
        print('✅ OCR dependencies available')
    except ImportError:
        print('⚠️  OCR dependencies not available (optional)')
    
    # Test async dependencies
    try:
        import openai
        import asyncio
        print('✅ Async processing dependencies available')
    except ImportError:
        print('⚠️  OpenAI/async dependencies not available (optional)')
    
    # Test system monitoring
    try:
        import psutil
        print('✅ System monitoring available')
    except ImportError:
        print('⚠️  System monitoring not available')
    
    print('✅ All critical imports successful')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" || exit 1

# Test module imports
echo "🧩 Testing application modules..."
python3 -c "
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

try:
    # Test core modules
    from manual_review import ManualReviewInterface
    from dynamic_prompt_engine import DynamicPromptEngine
    from smart_content_detector import SmartContentDetector
    print('✅ Core modules imported successfully')
    
    # Test production optimization modules
    from ocr_pdf_processor import OCRPDFProcessor
    from async_enhancement_processor import AsyncEnhancementProcessor
    from lean_session_manager import LeanSessionManager
    print('✅ Production optimization modules imported successfully')
    
    print('✅ All application modules imported successfully')
    
except ImportError as e:
    print(f'❌ Module import error: {e}')
    sys.exit(1)
" || exit 1

# Check required directories
echo "📁 Checking directory structure..."
for dir in "modules" "prompts" ".streamlit"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Required directory missing: $dir"
        exit 1
    fi
done
echo "✅ Directory structure valid"

# Check required files
echo "📄 Checking required files..."
required_files=(
    "enhanced_app_production_optimized.py"
    "requirements_production_optimized.txt"
    "render_optimized.yaml"
    ".streamlit/config.toml"
    ".streamlit/secrets.toml"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done
echo "✅ Required files present"

# Check prompt templates
echo "🎭 Checking prompt templates..."
prompt_count=$(ls prompts/*.txt 2>/dev/null | wc -l)
if [ "$prompt_count" -lt 6 ]; then
    echo "⚠️  Expected 6 prompt templates, found $prompt_count"
else
    echo "✅ Prompt templates available ($prompt_count)"
fi

# Test Tesseract (if available)
echo "👁️  Testing OCR capabilities..."
if command -v tesseract &> /dev/null; then
    tesseract_version=$(tesseract --version 2>&1 | head -n1)
    echo "✅ Tesseract available: $tesseract_version"
else
    echo "⚠️  Tesseract not available (OCR features will be limited)"
fi

# Check memory and system resources
echo "💾 Checking system resources..."
python3 -c "
import psutil
import os

# Memory check
memory = psutil.virtual_memory()
memory_gb = memory.total / (1024**3)
print(f'💾 Total memory: {memory_gb:.1f} GB')

if memory_gb < 0.4:  # Less than 400MB
    print('⚠️  Low memory detected - enabling low memory mode')
    os.environ['LOW_MEM_MODE'] = '1'
else:
    print('✅ Sufficient memory available')

# Disk space check
disk = psutil.disk_usage('/')
disk_gb = disk.free / (1024**3)
print(f'💽 Free disk space: {disk_gb:.1f} GB')

if disk_gb < 1:
    print('⚠️  Low disk space detected')
else:
    print('✅ Sufficient disk space available')
"

# Final status
echo ""
echo "🎉 Health check completed successfully!"
echo "✅ Fine-Tune Data System is ready to run"
echo ""
echo "To start the application:"
echo "streamlit run enhanced_app_production_optimized.py"
echo ""

exit 0

