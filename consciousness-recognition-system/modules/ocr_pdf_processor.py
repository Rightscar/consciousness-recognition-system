"""
OCR PDF Processor Module
========================

Handles large PDFs and image-only PDFs with OCR support and memory monitoring.
Implements streaming processing for memory efficiency on low-resource servers.

Features:
- Image-only page detection
- Tesseract OCR integration
- Memory monitoring and management
- Streaming processing for large files
- Progress tracking and error handling
"""

import os
import gc
import psutil
import logging
from typing import Dict, Any, List, Optional, Generator, Tuple
from pathlib import Path
import streamlit as st

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available - PDF processing will be limited")

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR dependencies not available - image-only PDFs cannot be processed")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


class MemoryMonitor:
    """Memory monitoring and management for large file processing"""
    
    def __init__(self, max_memory_mb: int = 800):
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1e6
        except Exception as e:
            self.logger.warning(f"Could not get memory usage: {e}")
            return 0.0
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit"""
        current_mb = self.get_memory_usage_mb()
        return current_mb > self.max_memory_mb
    
    def free_memory(self):
        """Attempt to free memory"""
        gc.collect()
        
        # Clear Streamlit cache if available
        if hasattr(st, 'cache_data'):
            try:
                st.cache_data.clear()
            except:
                pass
        
        if hasattr(st, 'cache_resource'):
            try:
                st.cache_resource.clear()
            except:
                pass
    
    def memory_warning(self) -> bool:
        """Check memory and show warning if needed"""
        if self.check_memory_limit():
            current_mb = self.get_memory_usage_mb()
            st.warning(f"⚠️ High memory usage detected: {current_mb:.1f}MB / {self.max_memory_mb}MB. Freeing cache...")
            self.free_memory()
            return True
        return False


class OCRPDFProcessor:
    """Advanced PDF processor with OCR support and memory management"""
    
    def __init__(self, max_memory_mb: int = 800, chunk_size: int = 5):
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        self.chunk_size = chunk_size  # Pages to process at once
        self.logger = logging.getLogger(__name__)
        
        # Check OCR availability
        self.ocr_available = OCR_AVAILABLE
        if not self.ocr_available:
            self.logger.warning("OCR not available - install pytesseract and pdf2image for image-only PDF support")
    
    def detect_image_only_pages(self, pdf_path: str) -> List[int]:
        """Detect which pages are image-only and need OCR"""
        image_pages = []
        
        if not PYMUPDF_AVAILABLE:
            self.logger.warning("PyMuPDF not available - cannot detect image-only pages")
            return image_pages
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get text content
                text = page.get_text().strip()
                
                # Check if page has minimal text but contains images
                if len(text) < 50:  # Very little text
                    page_dict = page.get_text("dict")
                    
                    # Look for image objects
                    has_images = False
                    if "blocks" in page_dict:
                        for block in page_dict["blocks"]:
                            if "type" in block and block["type"] == 1:  # Image block
                                has_images = True
                                break
                    
                    if has_images:
                        image_pages.append(page_num)
                        self.logger.info(f"Detected image-only page: {page_num + 1}")
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error detecting image-only pages: {e}")
        
        return image_pages
    
    def extract_text_with_ocr(self, pdf_path: str, page_num: int) -> str:
        """Extract text from a single page using OCR"""
        if not self.ocr_available:
            return f"[OCR not available for page {page_num + 1}]"
        
        try:
            # Convert PDF page to image
            if PYMUPDF_AVAILABLE:
                # Use PyMuPDF for better performance
                doc = fitz.open(pdf_path)
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                from io import BytesIO
                image = Image.open(BytesIO(img_data))
                doc.close()
            else:
                # Fallback to pdf2image
                images = pdf2image.convert_from_path(
                    pdf_path, 
                    first_page=page_num + 1, 
                    last_page=page_num + 1,
                    dpi=200
                )
                image = images[0] if images else None
            
            if image is None:
                return f"[Could not convert page {page_num + 1} to image]"
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang='eng')
            
            # Clean up
            image.close()
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"OCR failed for page {page_num + 1}: {e}")
            return f"[OCR failed for page {page_num + 1}: {str(e)}]"
    
    def process_pdf_streaming(self, pdf_path: str, file_size_mb: float) -> Generator[Dict[str, Any], None, None]:
        """Process PDF in streaming mode for memory efficiency"""
        
        # Determine if we need streaming based on file size
        use_streaming = file_size_mb > 50  # Stream files larger than 50MB
        
        if use_streaming:
            self.logger.info(f"Using streaming mode for {file_size_mb:.1f}MB PDF")
            yield from self._process_pdf_streaming_mode(pdf_path)
        else:
            self.logger.info(f"Using standard mode for {file_size_mb:.1f}MB PDF")
            yield from self._process_pdf_standard_mode(pdf_path)
    
    def _process_pdf_streaming_mode(self, pdf_path: str) -> Generator[Dict[str, Any], None, None]:
        """Process PDF in streaming mode - page by page"""
        
        if not PYMUPDF_AVAILABLE:
            yield {"error": "PyMuPDF required for streaming mode"}
            return
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Detect image-only pages first
            image_pages = self.detect_image_only_pages(pdf_path)
            
            # Process pages in chunks
            for chunk_start in range(0, total_pages, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_pages)
                
                chunk_data = {
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "total_pages": total_pages,
                    "pages": []
                }
                
                for page_num in range(chunk_start, chunk_end):
                    # Check memory before processing each page
                    if self.memory_monitor.memory_warning():
                        self.logger.warning(f"Memory warning during page {page_num + 1}")
                    
                    page = doc.load_page(page_num)
                    
                    if page_num in image_pages:
                        # Use OCR for image-only pages
                        text = self.extract_text_with_ocr(pdf_path, page_num)
                        method = "OCR"
                    else:
                        # Extract text normally
                        text = page.get_text()
                        method = "text_extraction"
                    
                    page_data = {
                        "page_number": page_num + 1,
                        "text": text,
                        "method": method,
                        "char_count": len(text)
                    }
                    
                    chunk_data["pages"].append(page_data)
                
                yield chunk_data
                
                # Force garbage collection after each chunk
                gc.collect()
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error in streaming PDF processing: {e}")
            yield {"error": str(e)}
    
    def _process_pdf_standard_mode(self, pdf_path: str) -> Generator[Dict[str, Any], None, None]:
        """Process PDF in standard mode - all at once"""
        
        try:
            # Try PyMuPDF first
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(pdf_path)
                
                # Detect image-only pages
                image_pages = self.detect_image_only_pages(pdf_path)
                
                all_pages = []
                total_pages = len(doc)
                
                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    
                    if page_num in image_pages:
                        text = self.extract_text_with_ocr(pdf_path, page_num)
                        method = "OCR"
                    else:
                        text = page.get_text()
                        method = "text_extraction"
                    
                    page_data = {
                        "page_number": page_num + 1,
                        "text": text,
                        "method": method,
                        "char_count": len(text)
                    }
                    
                    all_pages.append(page_data)
                
                doc.close()
                
                yield {
                    "chunk_start": 0,
                    "chunk_end": total_pages,
                    "total_pages": total_pages,
                    "pages": all_pages
                }
            
            # Fallback to PyPDF2
            elif PYPDF2_AVAILABLE:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    all_pages = []
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        
                        page_data = {
                            "page_number": page_num + 1,
                            "text": text,
                            "method": "PyPDF2",
                            "char_count": len(text)
                        }
                        
                        all_pages.append(page_data)
                    
                    yield {
                        "chunk_start": 0,
                        "chunk_end": len(all_pages),
                        "total_pages": len(all_pages),
                        "pages": all_pages
                    }
            
            else:
                yield {"error": "No PDF processing library available"}
                
        except Exception as e:
            self.logger.error(f"Error in standard PDF processing: {e}")
            yield {"error": str(e)}
    
    def get_processing_stats(self, pdf_path: str) -> Dict[str, Any]:
        """Get PDF processing statistics and recommendations"""
        
        try:
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            
            stats = {
                "file_size_mb": file_size_mb,
                "recommended_mode": "streaming" if file_size_mb > 50 else "standard",
                "memory_limit_mb": self.memory_monitor.max_memory_mb,
                "current_memory_mb": self.memory_monitor.get_memory_usage_mb(),
                "ocr_available": self.ocr_available,
                "pymupdf_available": PYMUPDF_AVAILABLE,
                "estimated_processing_time": self._estimate_processing_time(file_size_mb)
            }
            
            # Add warnings
            warnings = []
            if file_size_mb > 100:
                warnings.append("Very large file - consider using streaming mode")
            if not self.ocr_available:
                warnings.append("OCR not available - image-only pages will not be processed")
            if self.memory_monitor.check_memory_limit():
                warnings.append("High memory usage detected")
            
            stats["warnings"] = warnings
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def _estimate_processing_time(self, file_size_mb: float) -> str:
        """Estimate processing time based on file size"""
        
        # Rough estimates based on typical performance
        if file_size_mb < 1:
            return "< 30 seconds"
        elif file_size_mb < 10:
            return "1-3 minutes"
        elif file_size_mb < 50:
            return "3-10 minutes"
        elif file_size_mb < 100:
            return "10-20 minutes"
        else:
            return "20+ minutes"


# Global OCR processor instance
ocr_processor = OCRPDFProcessor()


def check_ocr_dependencies() -> Dict[str, bool]:
    """Check if OCR dependencies are available"""
    return {
        "pytesseract": OCR_AVAILABLE,
        "pymupdf": PYMUPDF_AVAILABLE,
        "pypdf2": PYPDF2_AVAILABLE,
        "tesseract_binary": _check_tesseract_binary()
    }


def _check_tesseract_binary() -> bool:
    """Check if Tesseract binary is available"""
    try:
        if OCR_AVAILABLE:
            pytesseract.get_tesseract_version()
            return True
    except:
        pass
    return False


def get_ocr_installation_guide() -> str:
    """Get installation guide for OCR dependencies"""
    
    guide = """
# OCR Installation Guide

## For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
pip install pytesseract pdf2image Pillow
```

## For macOS:
```bash
brew install tesseract
pip install pytesseract pdf2image Pillow
```

## For Windows:
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH
3. pip install pytesseract pdf2image Pillow

## For Render.com deployment:
Add to your build script:
```bash
apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng
```
"""
    
    return guide

