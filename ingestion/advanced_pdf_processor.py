"""
Advanced PDF Processor with OCR Fallback
========================================

Handles both standard text extraction and OCR fallback for scanned PDFs.
Integrates Tesseract OCR with pdfminer.six for comprehensive document processing.

Features:
- Automatic detection of scanned vs text-based PDFs
- Tesseract OCR integration with preprocessing
- Intelligent text post-processing and cleanup
- Multi-language OCR support
- Performance optimization for large documents
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from PIL import Image
import pdf2image
import pytesseract
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import cv2
import numpy as np
from modules.logger import get_logger
from shared.text_utils import clean_content, normalize_text

class AdvancedPDFProcessor:
    """
    Advanced PDF processor with OCR fallback capabilities.
    
    Automatically detects whether a PDF contains extractable text or requires OCR,
    then applies the appropriate extraction method for optimal results.
    """
    
    def __init__(self, 
                 tesseract_cmd: str = None,
                 temp_dir: str = None,
                 ocr_languages: List[str] = None):
        """
        Initialize the advanced PDF processor.
        
        Args:
            tesseract_cmd: Path to Tesseract executable
            temp_dir: Temporary directory for image processing
            ocr_languages: List of OCR languages (e.g., ['eng', 'fra', 'deu'])
        """
        self.logger = get_logger("advanced_pdf_processor")
        
        # Configure Tesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Setup temporary directory
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "pdf_ocr"
        self.temp_dir.mkdir(exist_ok=True)
        
        # OCR configuration
        self.ocr_languages = ocr_languages or ['eng']
        self.ocr_config = '--oem 3 --psm 6'  # Default Tesseract config
        
        # Detection thresholds
        self.min_text_ratio = 0.1  # Minimum text-to-page ratio for text-based PDF
        self.min_words_per_page = 10  # Minimum words per page for text-based PDF
        
        self.logger.info("Advanced PDF processor initialized")
    
    def extract_with_ocr_fallback(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF with automatic OCR fallback.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            self.logger.info(f"Processing PDF: {pdf_path.name}")
            
            # Step 1: Try standard text extraction
            standard_result = self._extract_text_standard(pdf_path)
            
            # Step 2: Analyze if OCR is needed
            needs_ocr = self._needs_ocr_processing(standard_result)
            
            if needs_ocr:
                self.logger.info("Standard extraction insufficient, using OCR fallback")
                ocr_result = self._extract_with_tesseract(pdf_path)
                
                # Combine results if both have content
                if standard_result['text'].strip() and ocr_result['text'].strip():
                    combined_text = self._combine_extraction_results(
                        standard_result['text'], 
                        ocr_result['text']
                    )
                    result = {
                        'text': combined_text,
                        'method': 'hybrid',
                        'pages_processed': max(standard_result['pages'], ocr_result['pages']),
                        'confidence': (standard_result['confidence'] + ocr_result['confidence']) / 2,
                        'metadata': {
                            'standard_extraction': standard_result['metadata'],
                            'ocr_extraction': ocr_result['metadata']
                        }
                    }
                else:
                    # Use OCR result if standard extraction failed
                    result = ocr_result
                    result['method'] = 'ocr_only'
            else:
                self.logger.info("Standard extraction successful")
                result = standard_result
                result['method'] = 'standard'
            
            # Step 3: Post-process the extracted text
            result['text'] = self._post_process_extracted_text(result['text'])
            
            self.logger.info(f"Extraction complete: {len(result['text'])} characters, method: {result['method']}")
            return result
            
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {str(e)}")
            raise
    
    def _extract_text_standard(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using standard PDF text extraction."""
        try:
            # Configure layout analysis parameters
            laparams = LAParams(
                boxes_flow=0.5,
                word_margin=0.1,
                char_margin=2.0,
                line_margin=0.5
            )
            
            # Extract text
            text = extract_text(str(pdf_path), laparams=laparams)
            
            # Calculate basic metrics
            pages = self._estimate_page_count(pdf_path)
            word_count = len(text.split())
            char_count = len(text)
            
            # Calculate confidence based on text density
            confidence = min(1.0, (word_count / max(pages, 1)) / 50)  # 50 words per page baseline
            
            return {
                'text': text,
                'pages': pages,
                'confidence': confidence,
                'metadata': {
                    'word_count': word_count,
                    'char_count': char_count,
                    'words_per_page': word_count / max(pages, 1),
                    'extraction_method': 'pdfminer'
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Standard extraction failed: {str(e)}")
            return {
                'text': '',
                'pages': 0,
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def _extract_with_tesseract(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using Tesseract OCR."""
        try:
            self.logger.info("Starting OCR extraction")
            
            # Convert PDF to images
            images = self._pdf_to_images(pdf_path)
            
            if not images:
                raise Exception("Failed to convert PDF to images")
            
            # Process each page with OCR
            all_text = []
            total_confidence = 0
            processed_pages = 0
            
            for i, image in enumerate(images):
                try:
                    # Preprocess image for better OCR
                    processed_image = self._preprocess_image_for_ocr(image)
                    
                    # Perform OCR with confidence data
                    ocr_data = pytesseract.image_to_data(
                        processed_image,
                        lang='+'.join(self.ocr_languages),
                        config=self.ocr_config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Extract text and calculate page confidence
                    page_text, page_confidence = self._extract_text_from_ocr_data(ocr_data)
                    
                    if page_text.strip():
                        all_text.append(page_text)
                        total_confidence += page_confidence
                        processed_pages += 1
                    
                    self.logger.debug(f"Processed page {i+1}: {len(page_text)} chars, confidence: {page_confidence:.2f}")
                    
                except Exception as e:
                    self.logger.warning(f"OCR failed for page {i+1}: {str(e)}")
                    continue
            
            # Combine all text
            combined_text = '\n\n'.join(all_text)
            avg_confidence = total_confidence / max(processed_pages, 1)
            
            return {
                'text': combined_text,
                'pages': processed_pages,
                'confidence': avg_confidence / 100,  # Normalize to 0-1
                'metadata': {
                    'word_count': len(combined_text.split()),
                    'char_count': len(combined_text),
                    'ocr_languages': self.ocr_languages,
                    'pages_with_text': processed_pages,
                    'total_pages': len(images),
                    'extraction_method': 'tesseract_ocr'
                }
            }
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            return {
                'text': '',
                'pages': 0,
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
        finally:
            # Cleanup temporary images
            self._cleanup_temp_files()
    
    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF pages to images for OCR processing."""
        try:
            # Convert PDF to images with optimized settings
            images = pdf2image.convert_from_path(
                str(pdf_path),
                dpi=300,  # High DPI for better OCR accuracy
                fmt='RGB',
                thread_count=2,
                poppler_path=None  # Use system poppler
            )
            
            self.logger.info(f"Converted PDF to {len(images)} images")
            return images
            
        except Exception as e:
            self.logger.error(f"PDF to image conversion failed: {str(e)}")
            return []
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy."""
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(cleaned)
            
            return processed_image
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {str(e)}")
            return image  # Return original if preprocessing fails
    
    def _extract_text_from_ocr_data(self, ocr_data: Dict) -> Tuple[str, float]:
        """Extract text and confidence from Tesseract OCR data."""
        words = []
        confidences = []
        
        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i].strip()
            confidence = ocr_data['conf'][i]
            
            if word and confidence > 0:  # Filter out empty words and invalid confidence
                words.append(word)
                confidences.append(confidence)
        
        # Combine words into text
        text = ' '.join(words)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return text, avg_confidence
    
    def _needs_ocr_processing(self, standard_result: Dict[str, Any]) -> bool:
        """Determine if OCR processing is needed based on standard extraction results."""
        text = standard_result['text']
        pages = standard_result['pages']
        
        if not text.strip():
            return True  # No text extracted, definitely need OCR
        
        # Calculate text density metrics
        word_count = len(text.split())
        words_per_page = word_count / max(pages, 1)
        
        # Check if text density is too low
        if words_per_page < self.min_words_per_page:
            self.logger.info(f"Low word density: {words_per_page:.1f} words/page (threshold: {self.min_words_per_page})")
            return True
        
        # Check for signs of scanned content (lots of garbled text)
        garbled_ratio = self._calculate_garbled_ratio(text)
        if garbled_ratio > 0.3:  # More than 30% garbled text
            self.logger.info(f"High garbled text ratio: {garbled_ratio:.2f}")
            return True
        
        return False
    
    def _calculate_garbled_ratio(self, text: str) -> float:
        """Calculate the ratio of potentially garbled text."""
        words = text.split()
        if not words:
            return 0.0
        
        garbled_count = 0
        for word in words:
            # Check for signs of garbled text
            if (len(word) > 20 or  # Extremely long words
                sum(1 for c in word if not c.isalnum()) / len(word) > 0.5 or  # Too many special chars
                word.isupper() and len(word) > 10):  # Long all-caps words
                garbled_count += 1
        
        return garbled_count / len(words)
    
    def _combine_extraction_results(self, standard_text: str, ocr_text: str) -> str:
        """Intelligently combine standard and OCR extraction results."""
        # Simple combination strategy - use the longer text as base
        if len(ocr_text) > len(standard_text) * 1.5:
            return ocr_text
        elif len(standard_text) > len(ocr_text) * 1.5:
            return standard_text
        else:
            # Similar lengths - combine both
            return f"{standard_text}\n\n--- OCR SUPPLEMENT ---\n\n{ocr_text}"
    
    def _post_process_extracted_text(self, text: str) -> str:
        """Post-process extracted text for better quality."""
        if not text:
            return text
        
        # Apply text cleaning and normalization
        cleaned = clean_content(text, aggressive=True)
        normalized = normalize_text(cleaned, remove_extra_whitespace=True)
        
        # Additional OCR-specific cleaning
        normalized = self._clean_ocr_artifacts(normalized)
        
        return normalized
    
    def _clean_ocr_artifacts(self, text: str) -> str:
        """Clean common OCR artifacts and errors."""
        import re
        
        # Fix common OCR character substitutions
        ocr_fixes = {
            r'\b0\b': 'O',  # Zero to O
            r'\b1\b': 'I',  # One to I (in context)
            r'rn': 'm',     # rn to m
            r'vv': 'w',     # vv to w
            r'\|': 'l',     # | to l
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove excessive whitespace and line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {3,}', ' ', text)
        
        return text
    
    def _estimate_page_count(self, pdf_path: Path) -> int:
        """Estimate the number of pages in a PDF."""
        try:
            # Use pdf2image to get accurate page count
            images = pdf2image.convert_from_path(str(pdf_path), first_page=1, last_page=1)
            # This is a simple way to get page count - there are more efficient methods
            # but this works reliably across different PDF types
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return len(reader.pages)
        except:
            return 1  # Fallback
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        try:
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
        except Exception as e:
            self.logger.warning(f"Temp file cleanup failed: {str(e)}")
    
    def is_scanned_pdf(self, text: str, pages: int = 1) -> bool:
        """Public method to check if a PDF appears to be scanned."""
        if not text.strip():
            return True
        
        words_per_page = len(text.split()) / max(pages, 1)
        return words_per_page < self.min_words_per_page
    
    def get_extraction_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed statistics about the extraction process."""
        return {
            'method_used': result.get('method', 'unknown'),
            'pages_processed': result.get('pages', 0),
            'confidence_score': result.get('confidence', 0.0),
            'text_length': len(result.get('text', '')),
            'word_count': len(result.get('text', '').split()),
            'metadata': result.get('metadata', {})
        }

# Convenience functions for easy integration
def extract_pdf_with_ocr_fallback(pdf_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to extract text from PDF with OCR fallback.
    
    Args:
        pdf_path: Path to PDF file
        **kwargs: Additional arguments for AdvancedPDFProcessor
    
    Returns:
        Extraction result dictionary
    """
    processor = AdvancedPDFProcessor(**kwargs)
    return processor.extract_with_ocr_fallback(pdf_path)

def is_pdf_scanned(pdf_path: str) -> bool:
    """
    Quick check if a PDF appears to be scanned.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        True if PDF appears to be scanned
    """
    processor = AdvancedPDFProcessor()
    standard_result = processor._extract_text_standard(Path(pdf_path))
    return processor.is_scanned_pdf(standard_result['text'], standard_result['pages'])

# Example usage
if __name__ == "__main__":
    # Example usage
    processor = AdvancedPDFProcessor()
    
    # Process a PDF file
    result = processor.extract_with_ocr_fallback("example.pdf")
    
    print(f"Extraction method: {result['method']}")
    print(f"Text length: {len(result['text'])}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Pages processed: {result['pages']}")
    
    # Get detailed stats
    stats = processor.get_extraction_stats(result)
    print(f"Extraction stats: {stats}")

