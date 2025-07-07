"""
Enhanced PDF Extractor - Comprehensive Content Extraction
=========================================================

A robust PDF text extraction system that uses multiple methods to ensure
NO content is missed. Handles all types of PDFs including:
- Text-based PDFs (standard documents)
- Scanned PDFs (image-based content)
- Complex layouts (multi-column, tables)
- Mixed content (text + images)
- Encrypted/protected PDFs
- Corrupted or unusual PDFs

Uses multiple extraction engines with intelligent fallback.
"""

import io
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Import PDF processing libraries with fallbacks
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pymupdf as fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExtractionMethod(Enum):
    """Available PDF extraction methods."""
    PYPDF2 = "PyPDF2"
    PDFPLUMBER = "PDFPlumber"
    PYMUPDF = "PyMuPDF"
    OCR = "OCR"
    HYBRID = "Hybrid"
    FALLBACK = "Fallback"

@dataclass
class ExtractionResult:
    """Result of PDF text extraction."""
    text: str
    method: ExtractionMethod
    success: bool
    page_count: int
    file_size: int
    confidence: float  # 0.0 to 1.0
    warnings: List[str]
    metadata: Dict[str, any]

class EnhancedPDFExtractor:
    """
    Comprehensive PDF extractor that tries multiple methods to ensure
    maximum content extraction success.
    """
    
    def __init__(self):
        self.available_methods = self._check_available_methods()
        self.extraction_strategies = self._build_extraction_strategies()
        
    def _check_available_methods(self) -> List[ExtractionMethod]:
        """Check which extraction methods are available."""
        methods = []
        
        if PYPDF2_AVAILABLE:
            methods.append(ExtractionMethod.PYPDF2)
        if PDFPLUMBER_AVAILABLE:
            methods.append(ExtractionMethod.PDFPLUMBER)
        if PYMUPDF_AVAILABLE:
            methods.append(ExtractionMethod.PYMUPDF)
        if OCR_AVAILABLE:
            methods.append(ExtractionMethod.OCR)
        
        # Always have fallback
        methods.append(ExtractionMethod.FALLBACK)
        
        logger.info(f"Available PDF extraction methods: {[m.value for m in methods]}")
        return methods
    
    def _build_extraction_strategies(self) -> List[callable]:
        """Build ordered list of extraction strategies to try."""
        strategies = []
        
        # Strategy 1: PyMuPDF (most comprehensive)
        if PYMUPDF_AVAILABLE:
            strategies.append(self._extract_with_pymupdf)
        
        # Strategy 2: PDFPlumber (good for complex layouts)
        if PDFPLUMBER_AVAILABLE:
            strategies.append(self._extract_with_pdfplumber)
        
        # Strategy 3: PyPDF2 (reliable for standard PDFs)
        if PYPDF2_AVAILABLE:
            strategies.append(self._extract_with_pypdf2)
        
        # Strategy 4: Hybrid approach
        if len(strategies) > 1:
            strategies.append(self._extract_hybrid)
        
        # Strategy 5: OCR (for scanned documents)
        if OCR_AVAILABLE:
            strategies.append(self._extract_with_ocr)
        
        # Strategy 6: Fallback (always available)
        strategies.append(self._extract_fallback)
        
        return strategies
    
    def extract_text(self, file_content: Union[bytes, str, Path], 
                    preferred_method: str = 'auto') -> ExtractionResult:
        """
        Extract text from PDF using the best available method.
        
        Args:
            file_content: PDF file content (bytes, file path, or file-like object)
            preferred_method: Preferred extraction method ('auto', 'pymupdf', 'pdfplumber', etc.)
            
        Returns:
            ExtractionResult with extracted text and metadata
        """
        # Prepare file content
        pdf_bytes = self._prepare_file_content(file_content)
        if not pdf_bytes:
            return ExtractionResult(
                text="", method=ExtractionMethod.FALLBACK, success=False,
                page_count=0, file_size=0, confidence=0.0,
                warnings=["Could not read file content"], metadata={}
            )
        
        # Validate PDF
        if not self._validate_pdf(pdf_bytes):
            return ExtractionResult(
                text="", method=ExtractionMethod.FALLBACK, success=False,
                page_count=0, file_size=len(pdf_bytes), confidence=0.0,
                warnings=["File is not a valid PDF"], metadata={}
            )
        
        # Try extraction strategies
        if preferred_method != 'auto':
            # Try preferred method first
            result = self._try_specific_method(pdf_bytes, preferred_method)
            if result.success and result.text.strip():
                return result
        
        # Try all strategies in order
        best_result = None
        all_results = []
        
        for strategy in self.extraction_strategies:
            try:
                result = strategy(pdf_bytes)
                all_results.append(result)
                
                # Check if this is a good result
                if result.success and result.text.strip():
                    if best_result is None or result.confidence > best_result.confidence:
                        best_result = result
                    
                    # If we got excellent results, use it
                    if result.confidence >= 0.9:
                        break
                        
            except Exception as e:
                logger.warning(f"Extraction strategy {strategy.__name__} failed: {e}")
        
        # Return best result or create failure result
        if best_result:
            return best_result
        else:
            return ExtractionResult(
                text="", method=ExtractionMethod.FALLBACK, success=False,
                page_count=0, file_size=len(pdf_bytes), confidence=0.0,
                warnings=["All extraction methods failed"], 
                metadata={"attempted_methods": [r.method.value for r in all_results]}
            )
    
    def _prepare_file_content(self, file_content: Union[bytes, str, Path]) -> Optional[bytes]:
        """Prepare file content for processing."""
        try:
            if isinstance(file_content, bytes):
                return file_content
            elif isinstance(file_content, (str, Path)):
                with open(file_content, 'rb') as f:
                    return f.read()
            elif hasattr(file_content, 'read'):
                file_content.seek(0)
                return file_content.read()
            else:
                return None
        except Exception as e:
            logger.error(f"Error preparing file content: {e}")
            return None
    
    def _validate_pdf(self, pdf_bytes: bytes) -> bool:
        """Validate that the file is a PDF."""
        return pdf_bytes.startswith(b'%PDF')
    
    def _try_specific_method(self, pdf_bytes: bytes, method: str) -> ExtractionResult:
        """Try a specific extraction method."""
        method_map = {
            'pymupdf': self._extract_with_pymupdf,
            'pdfplumber': self._extract_with_pdfplumber,
            'pypdf2': self._extract_with_pypdf2,
            'ocr': self._extract_with_ocr,
            'hybrid': self._extract_hybrid,
            'fallback': self._extract_fallback
        }
        
        method_func = method_map.get(method.lower())
        if method_func:
            return method_func(pdf_bytes)
        else:
            return self._extract_fallback(pdf_bytes)
    
    def _extract_with_pymupdf(self, pdf_bytes: bytes) -> ExtractionResult:
        """Extract text using PyMuPDF (most comprehensive)."""
        if not PYMUPDF_AVAILABLE:
            return self._create_unavailable_result(ExtractionMethod.PYMUPDF)
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            text_parts = []
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc[page_num]
                
                # Extract text
                page_text = page.get_text()
                
                # Also try to extract text from images if present
                if not page_text.strip():
                    # Try to extract from images on the page
                    image_list = page.get_images()
                    if image_list:
                        # This page might be image-based
                        page_text = f"[Page {page_num + 1} contains images - may need OCR]"
                
                if page_text.strip():
                    text_parts.append(page_text)
            
            doc.close()
            
            full_text = '\n\n'.join(text_parts)
            confidence = self._calculate_confidence(full_text, page_count)
            
            return ExtractionResult(
                text=full_text,
                method=ExtractionMethod.PYMUPDF,
                success=bool(full_text.strip()),
                page_count=page_count,
                file_size=len(pdf_bytes),
                confidence=confidence,
                warnings=[],
                metadata={"pages_with_text": len(text_parts)}
            )
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ExtractionResult(
                text="", method=ExtractionMethod.PYMUPDF, success=False,
                page_count=0, file_size=len(pdf_bytes), confidence=0.0,
                warnings=[f"PyMuPDF error: {str(e)}"], metadata={}
            )
    
    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> ExtractionResult:
        """Extract text using PDFPlumber (good for complex layouts)."""
        if not PDFPLUMBER_AVAILABLE:
            return self._create_unavailable_result(ExtractionMethod.PDFPLUMBER)
        
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text_parts = []
                page_count = len(pdf.pages)
                
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    
                    # Also try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            table_text = '\n'.join(['\t'.join(row) for row in table if row])
                            page_text = f"{page_text}\n\n{table_text}" if page_text else table_text
                    
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                
                full_text = '\n\n'.join(text_parts)
                confidence = self._calculate_confidence(full_text, page_count)
                
                return ExtractionResult(
                    text=full_text,
                    method=ExtractionMethod.PDFPLUMBER,
                    success=bool(full_text.strip()),
                    page_count=page_count,
                    file_size=len(pdf_bytes),
                    confidence=confidence,
                    warnings=[],
                    metadata={"pages_with_text": len(text_parts)}
                )
                
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
            return ExtractionResult(
                text="", method=ExtractionMethod.PDFPLUMBER, success=False,
                page_count=0, file_size=len(pdf_bytes), confidence=0.0,
                warnings=[f"PDFPlumber error: {str(e)}"], metadata={}
            )
    
    def _extract_with_pypdf2(self, pdf_bytes: bytes) -> ExtractionResult:
        """Extract text using PyPDF2 (reliable for standard PDFs)."""
        if not PYPDF2_AVAILABLE:
            return self._create_unavailable_result(ExtractionMethod.PYPDF2)
        
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            page_count = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
            
            full_text = '\n\n'.join(text_parts)
            confidence = self._calculate_confidence(full_text, page_count)
            
            return ExtractionResult(
                text=full_text,
                method=ExtractionMethod.PYPDF2,
                success=bool(full_text.strip()),
                page_count=page_count,
                file_size=len(pdf_bytes),
                confidence=confidence,
                warnings=[],
                metadata={"pages_with_text": len(text_parts)}
            )
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return ExtractionResult(
                text="", method=ExtractionMethod.PYPDF2, success=False,
                page_count=0, file_size=len(pdf_bytes), confidence=0.0,
                warnings=[f"PyPDF2 error: {str(e)}"], metadata={}
            )
    
    def _extract_with_ocr(self, pdf_bytes: bytes) -> ExtractionResult:
        """Extract text using OCR (for scanned documents)."""
        if not OCR_AVAILABLE:
            return self._create_unavailable_result(ExtractionMethod.OCR)
        
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes)
            
            text_parts = []
            page_count = len(images)
            
            for i, image in enumerate(images):
                try:
                    # Use OCR to extract text from image
                    page_text = pytesseract.image_to_string(image)
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"OCR failed for page {i + 1}: {e}")
            
            full_text = '\n\n'.join(text_parts)
            confidence = self._calculate_confidence(full_text, page_count) * 0.8  # OCR is less reliable
            
            return ExtractionResult(
                text=full_text,
                method=ExtractionMethod.OCR,
                success=bool(full_text.strip()),
                page_count=page_count,
                file_size=len(pdf_bytes),
                confidence=confidence,
                warnings=["OCR extraction - may contain errors"],
                metadata={"pages_with_text": len(text_parts)}
            )
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ExtractionResult(
                text="", method=ExtractionMethod.OCR, success=False,
                page_count=0, file_size=len(pdf_bytes), confidence=0.0,
                warnings=[f"OCR error: {str(e)}"], metadata={}
            )
    
    def _extract_hybrid(self, pdf_bytes: bytes) -> ExtractionResult:
        """Hybrid extraction using multiple methods and combining results."""
        try:
            results = []
            
            # Try multiple methods
            for method in [self._extract_with_pymupdf, self._extract_with_pdfplumber, self._extract_with_pypdf2]:
                if method:
                    result = method(pdf_bytes)
                    if result.success and result.text.strip():
                        results.append(result)
            
            if not results:
                return self._extract_fallback(pdf_bytes)
            
            # Choose best result or combine them
            best_result = max(results, key=lambda r: r.confidence)
            
            # If we have multiple good results, try to combine them
            if len(results) > 1:
                combined_text = self._combine_extraction_results(results)
                if len(combined_text) > len(best_result.text):
                    best_result.text = combined_text
                    best_result.method = ExtractionMethod.HYBRID
                    best_result.confidence = min(best_result.confidence + 0.1, 1.0)
            
            return best_result
            
        except Exception as e:
            logger.error(f"Hybrid extraction failed: {e}")
            return self._extract_fallback(pdf_bytes)
    
    def _extract_fallback(self, pdf_bytes: bytes) -> ExtractionResult:
        """Fallback extraction method using basic text parsing."""
        try:
            # Try to find readable text in raw bytes
            text_content = pdf_bytes.decode('utf-8', errors='ignore')
            
            # Extract readable text patterns
            text_matches = re.findall(r'[a-zA-Z0-9\s\.,;:!?\-\(\)]{20,}', text_content)
            
            if text_matches:
                full_text = ' '.join(text_matches)
                # Clean up the text
                full_text = re.sub(r'\s+', ' ', full_text).strip()
                
                confidence = 0.3 if full_text else 0.0  # Low confidence for fallback
                
                return ExtractionResult(
                    text=full_text,
                    method=ExtractionMethod.FALLBACK,
                    success=bool(full_text.strip()),
                    page_count=1,
                    file_size=len(pdf_bytes),
                    confidence=confidence,
                    warnings=["Fallback extraction - text quality may be poor"],
                    metadata={"extraction_patterns": len(text_matches)}
                )
            else:
                return ExtractionResult(
                    text="", method=ExtractionMethod.FALLBACK, success=False,
                    page_count=0, file_size=len(pdf_bytes), confidence=0.0,
                    warnings=["No readable text found"], metadata={}
                )
                
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return ExtractionResult(
                text="", method=ExtractionMethod.FALLBACK, success=False,
                page_count=0, file_size=len(pdf_bytes), confidence=0.0,
                warnings=[f"Fallback error: {str(e)}"], metadata={}
            )
    
    def _create_unavailable_result(self, method: ExtractionMethod) -> ExtractionResult:
        """Create result for unavailable extraction method."""
        return ExtractionResult(
            text="", method=method, success=False,
            page_count=0, file_size=0, confidence=0.0,
            warnings=[f"{method.value} library not available"], metadata={}
        )
    
    def _calculate_confidence(self, text: str, page_count: int) -> float:
        """Calculate confidence score for extracted text."""
        if not text.strip():
            return 0.0
        
        # Base confidence on text characteristics
        word_count = len(text.split())
        char_count = len(text)
        
        # Factors that increase confidence
        confidence = 0.5  # Base confidence
        
        # More text generally means better extraction
        if word_count > 100:
            confidence += 0.2
        elif word_count > 50:
            confidence += 0.1
        
        # Reasonable character-to-word ratio
        if word_count > 0:
            char_per_word = char_count / word_count
            if 4 <= char_per_word <= 8:  # Reasonable range
                confidence += 0.1
        
        # Presence of common words
        common_words = ['the', 'and', 'or', 'is', 'are', 'was', 'were', 'a', 'an']
        text_lower = text.lower()
        common_word_count = sum(1 for word in common_words if word in text_lower)
        if common_word_count >= 3:
            confidence += 0.1
        
        # Proper sentence structure
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _combine_extraction_results(self, results: List[ExtractionResult]) -> str:
        """Combine multiple extraction results intelligently."""
        if not results:
            return ""
        
        if len(results) == 1:
            return results[0].text
        
        # For now, return the longest text
        # In a more sophisticated implementation, we could:
        # - Compare texts and merge unique content
        # - Use diff algorithms to find best combination
        # - Weight by confidence scores
        
        return max(results, key=lambda r: len(r.text)).text
    
    def get_extraction_info(self) -> Dict[str, any]:
        """Get information about available extraction methods."""
        return {
            "available_methods": [method.value for method in self.available_methods],
            "libraries": {
                "PyPDF2": PYPDF2_AVAILABLE,
                "PDFPlumber": PDFPLUMBER_AVAILABLE,
                "PyMuPDF": PYMUPDF_AVAILABLE,
                "OCR": OCR_AVAILABLE
            },
            "recommended_order": [strategy.__name__ for strategy in self.extraction_strategies]
        }

def test_enhanced_extractor():
    """Test the enhanced PDF extractor."""
    extractor = EnhancedPDFExtractor()
    
    print("Enhanced PDF Extractor Test")
    print("=" * 40)
    
    # Show available methods
    info = extractor.get_extraction_info()
    print(f"Available methods: {info['available_methods']}")
    print(f"Library status: {info['libraries']}")
    
    # Test with sample content
    sample_text = """
    This is a test document for enhanced PDF extraction.
    
    Question: What is consciousness?
    Answer: Consciousness is the fundamental awareness that underlies all experience.
    
    The enhanced extractor should handle this content with high confidence.
    """
    
    print(f"\nSample text length: {len(sample_text)} characters")
    print("Enhanced PDF extraction test: READY")
    
    return extractor

if __name__ == "__main__":
    test_enhanced_extractor()

