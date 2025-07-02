"""
Enhanced PDF Extractor with Chunking for Large Files

Handles large PDFs efficiently with memory management and progress tracking.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Generator
import tempfile
import os
from pathlib import Path
import time

try:
    from .extractor import PDFExtractor
except ImportError:
    # Fallback for standalone usage
    import sys
    sys.path.append('..')
    from extractor import PDFExtractor


class EnhancedPDFExtractor(PDFExtractor):
    """Enhanced PDF extractor with chunking and progress tracking."""
    
    def __init__(self, chunk_size: int = 20):
        """
        Initialize enhanced extractor.
        
        Args:
            chunk_size: Number of pages to process per chunk
        """
        super().__init__()
        self.chunk_size = chunk_size
    
    def extract_text_chunked(
        self, 
        pdf_path: str, 
        progress_callback: Optional[callable] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Extract text from PDF in chunks with progress tracking.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Function to call with progress updates
            
        Yields:
            Dict containing chunk info and extracted text
        """
        try:
            import fitz  # PyMuPDF
            
            # Open PDF and get total pages
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            if progress_callback:
                progress_callback(0, f"Starting extraction of {total_pages} pages...")
            
            # Process in chunks
            for start_page in range(0, total_pages, self.chunk_size):
                end_page = min(start_page + self.chunk_size, total_pages)
                
                # Extract text from current chunk
                chunk_text = ""
                for page_num in range(start_page, end_page):
                    page = doc[page_num]
                    chunk_text += page.get_text() + "\n"
                
                # Calculate progress
                progress = (end_page / total_pages) * 100
                
                if progress_callback:
                    progress_callback(
                        progress, 
                        f"Processed pages {start_page + 1}-{end_page} of {total_pages}"
                    )
                
                yield {
                    'chunk_id': start_page // self.chunk_size,
                    'start_page': start_page + 1,
                    'end_page': end_page,
                    'total_pages': total_pages,
                    'text': chunk_text,
                    'progress': progress
                }
                
                # Small delay to prevent UI freezing
                time.sleep(0.01)
            
            doc.close()
            
            if progress_callback:
                progress_callback(100, "Extraction completed!")
                
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error: {str(e)}")
            raise
    
    def extract_text_with_progress(
        self, 
        pdf_path: str,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text with Streamlit progress tracking.
        
        Args:
            pdf_path: Path to PDF file
            show_progress: Whether to show progress in Streamlit
            
        Returns:
            Dict with extraction results
        """
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        all_text = ""
        chunk_info = []
        
        def update_progress(progress: float, message: str):
            if show_progress:
                progress_bar.progress(progress / 100)
                status_text.text(message)
        
        try:
            for chunk in self.extract_text_chunked(pdf_path, update_progress):
                all_text += chunk['text']
                chunk_info.append({
                    'chunk_id': chunk['chunk_id'],
                    'pages': f"{chunk['start_page']}-{chunk['end_page']}",
                    'text_length': len(chunk['text'])
                })
            
            if show_progress:
                progress_bar.progress(100)
                status_text.text("✅ Extraction completed successfully!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
            
            return {
                'success': True,
                'text': all_text,
                'total_chunks': len(chunk_info),
                'chunk_info': chunk_info,
                'total_length': len(all_text)
            }
            
        except Exception as e:
            if show_progress:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Extraction failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'text': all_text,  # Return partial text if any
                'chunk_info': chunk_info
            }
    
    def validate_file_size(self, file_path: str, max_size_mb: int = 50) -> Dict[str, Any]:
        """
        Validate PDF file size and provide warnings.
        
        Args:
            file_path: Path to PDF file
            max_size_mb: Maximum recommended size in MB
            
        Returns:
            Dict with validation results
        """
        try:
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            
            result = {
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
                    result['recommendation'] = f"Processing will use chunking (pages processed in batches of {self.chunk_size})."
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'size_bytes': 0,
                'size_mb': 0,
                'is_large': False
            }
    
    def estimate_processing_time(self, file_size_mb: float) -> str:
        """
        Estimate processing time based on file size.
        
        Args:
            file_size_mb: File size in megabytes
            
        Returns:
            Estimated time string
        """
        # Rough estimates based on typical performance
        if file_size_mb < 5:
            return "< 30 seconds"
        elif file_size_mb < 20:
            return "1-2 minutes"
        elif file_size_mb < 50:
            return "2-5 minutes"
        elif file_size_mb < 100:
            return "5-10 minutes"
        else:
            return "10+ minutes"


def create_chunked_extractor(chunk_size: int = 20) -> EnhancedPDFExtractor:
    """
    Factory function to create enhanced extractor.
    
    Args:
        chunk_size: Pages per chunk
        
    Returns:
        EnhancedPDFExtractor instance
    """
    return EnhancedPDFExtractor(chunk_size=chunk_size)

