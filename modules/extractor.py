"""
PDF Text Extractor for Consciousness Recognition System

Handles PDF text extraction with support for both text-based and scanned PDFs.
"""

import os
from typing import Dict, Any, Optional


class PDFExtractor:
    """Basic PDF text extractor."""
    
    def __init__(self):
        """Initialize the PDF extractor."""
        pass
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            # Try PyMuPDF first (faster and more reliable)
            import fitz
            
            doc = fitz.open(pdf_path)
            text = ""
            
            for page in doc:
                text += page.get_text() + "\n"
            
            doc.close()
            return text
            
        except ImportError:
            # Fallback to pdfplumber
            try:
                import pdfplumber
                
                text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                return text
                
            except ImportError:
                raise ImportError("Neither PyMuPDF nor pdfplumber is installed. Install with: pip install PyMuPDF pdfplumber")
        
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def validate_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Validate PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Validation results
        """
        if not os.path.exists(pdf_path):
            return {'valid': False, 'error': 'File does not exist'}
        
        if not pdf_path.lower().endswith('.pdf'):
            return {'valid': False, 'error': 'File is not a PDF'}
        
        try:
            # Try to open with PyMuPDF
            import fitz
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            
            return {
                'valid': True,
                'page_count': page_count,
                'file_size': os.path.getsize(pdf_path)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}

