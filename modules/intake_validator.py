"""
Comprehensive Intake Validation for Consciousness Recognition System

Provides secure file validation, content verification, and safe processing
for uploaded PDF files.
"""

import os
import tempfile
import hashlib
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import psutil
from dataclasses import dataclass


@dataclass
class FileValidationResult:
    """Result of file validation."""
    is_valid: bool
    file_hash: str
    file_size: int
    page_count: int
    has_text: bool
    is_encrypted: bool
    encoding_issues: List[str]
    security_warnings: List[str]
    processing_recommendations: List[str]
    error_message: Optional[str] = None


class SecureIntakeValidator:
    """Secure file intake validation and processing."""
    
    def __init__(self):
        """Initialize the secure intake validator."""
        self.max_file_size = 100 * 1024 * 1024  # 100MB hard limit
        self.max_pages = 1000  # Maximum pages per PDF
        self.allowed_mime_types = ['application/pdf']
        self.processed_hashes = set()  # Track processed files
        
        # Security patterns to detect in PDFs
        self.security_patterns = [
            b'/JavaScript',
            b'/JS',
            b'/Launch',
            b'/EmbeddedFile',
            b'/FileAttachment'
        ]
    
    def validate_upload(self, uploaded_file) -> FileValidationResult:
        """
        Comprehensive validation of uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            FileValidationResult with validation details
        """
        try:
            # Step 1: Basic file checks
            if uploaded_file is None:
                return FileValidationResult(
                    is_valid=False,
                    file_hash="",
                    file_size=0,
                    page_count=0,
                    has_text=False,
                    is_encrypted=False,
                    encoding_issues=[],
                    security_warnings=[],
                    processing_recommendations=[],
                    error_message="No file provided"
                )
            
            # Step 2: Size validation
            file_size = uploaded_file.size
            if file_size > self.max_file_size:
                return FileValidationResult(
                    is_valid=False,
                    file_hash="",
                    file_size=file_size,
                    page_count=0,
                    has_text=False,
                    is_encrypted=False,
                    encoding_issues=[],
                    security_warnings=[f"File too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size / 1024 / 1024}MB"],
                    processing_recommendations=[],
                    error_message=f"File exceeds maximum size limit of {self.max_file_size / 1024 / 1024}MB"
                )
            
            # Step 3: Create secure temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                # Read file content in chunks to avoid memory issues
                file_content = self._read_file_safely(uploaded_file)
                if file_content is None:
                    return FileValidationResult(
                        is_valid=False,
                        file_hash="",
                        file_size=file_size,
                        page_count=0,
                        has_text=False,
                        is_encrypted=False,
                        encoding_issues=[],
                        security_warnings=["Failed to read file content"],
                        processing_recommendations=[],
                        error_message="Unable to read file content"
                    )
                
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                # Step 4: Generate file hash for duplicate detection
                file_hash = hashlib.sha256(file_content).hexdigest()
                
                # Step 5: MIME type validation
                mime_validation = self._validate_mime_type(tmp_path)
                if not mime_validation['is_valid']:
                    return FileValidationResult(
                        is_valid=False,
                        file_hash=file_hash,
                        file_size=file_size,
                        page_count=0,
                        has_text=False,
                        is_encrypted=False,
                        encoding_issues=[],
                        security_warnings=mime_validation['warnings'],
                        processing_recommendations=[],
                        error_message=mime_validation['error']
                    )
                
                # Step 6: PDF structure validation
                pdf_validation = self._validate_pdf_structure(tmp_path)
                
                # Step 7: Security scanning
                security_scan = self._scan_for_security_issues(file_content)
                
                # Step 8: Text extraction test
                text_validation = self._validate_text_extraction(tmp_path)
                
                # Step 9: Duplicate detection
                is_duplicate = file_hash in self.processed_hashes
                if not is_duplicate:
                    self.processed_hashes.add(file_hash)
                
                # Compile results
                is_valid = (
                    pdf_validation['is_valid'] and
                    len(security_scan['critical_issues']) == 0 and
                    not is_duplicate
                )
                
                recommendations = []
                if is_duplicate:
                    recommendations.append("File already processed - skipping to avoid duplicates")
                if pdf_validation['page_count'] > 200:
                    recommendations.append("Large PDF detected - will use chunked processing")
                if not text_validation['has_extractable_text']:
                    recommendations.append("No extractable text found - may need OCR processing")
                if text_validation['encoding_issues']:
                    recommendations.append("Text encoding issues detected - may affect quality")
                
                return FileValidationResult(
                    is_valid=is_valid,
                    file_hash=file_hash,
                    file_size=file_size,
                    page_count=pdf_validation['page_count'],
                    has_text=text_validation['has_extractable_text'],
                    is_encrypted=pdf_validation['is_encrypted'],
                    encoding_issues=text_validation['encoding_issues'],
                    security_warnings=security_scan['warnings'] + security_scan['critical_issues'],
                    processing_recommendations=recommendations,
                    error_message="Duplicate file detected" if is_duplicate else None
                )
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_hash="",
                file_size=0,
                page_count=0,
                has_text=False,
                is_encrypted=False,
                encoding_issues=[],
                security_warnings=[],
                processing_recommendations=[],
                error_message=f"Validation failed: {str(e)}"
            )
    
    def _read_file_safely(self, uploaded_file, chunk_size: int = 8192) -> Optional[bytes]:
        """Safely read uploaded file in chunks."""
        try:
            uploaded_file.seek(0)  # Reset file pointer
            content = b""
            
            while True:
                chunk = uploaded_file.read(chunk_size)
                if not chunk:
                    break
                content += chunk
                
                # Check memory usage
                if len(content) > self.max_file_size:
                    return None
            
            return content
            
        except Exception:
            return None
    
    def _validate_mime_type(self, file_path: str) -> Dict[str, Any]:
        """Validate file MIME type."""
        try:
            # Check magic number
            with open(file_path, 'rb') as f:
                header = f.read(8)
            
            if not header.startswith(b'%PDF-'):
                return {
                    'is_valid': False,
                    'error': 'File is not a valid PDF (missing PDF header)',
                    'warnings': ['Invalid PDF magic number']
                }
            
            # Use python-magic if available
            try:
                if MAGIC_AVAILABLE:
                    mime_type = magic.from_file(file_path, mime=True)
                    if mime_type not in self.allowed_mime_types:
                        return {
                            'is_valid': False,
                            'error': f'Invalid MIME type: {mime_type}',
                            'warnings': [f'Unexpected MIME type: {mime_type}']
                        }
                else:
                    # Fallback: just use header check
                    pass
            except Exception:
                # Fallback to basic header check
                pass
            
            return {
                'is_valid': True,
                'error': None,
                'warnings': []
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'MIME validation failed: {str(e)}',
                'warnings': ['Could not validate file type']
            }
    
    def _validate_pdf_structure(self, file_path: str) -> Dict[str, Any]:
        """Validate PDF structure and extract metadata."""
        try:
            doc = fitz.open(file_path)
            
            page_count = len(doc)
            is_encrypted = doc.needs_pass
            
            # Check for reasonable page count
            if page_count > self.max_pages:
                doc.close()
                return {
                    'is_valid': False,
                    'page_count': page_count,
                    'is_encrypted': is_encrypted,
                    'error': f'PDF has too many pages: {page_count} > {self.max_pages}'
                }
            
            if page_count == 0:
                doc.close()
                return {
                    'is_valid': False,
                    'page_count': 0,
                    'is_encrypted': is_encrypted,
                    'error': 'PDF has no pages'
                }
            
            doc.close()
            
            return {
                'is_valid': True,
                'page_count': page_count,
                'is_encrypted': is_encrypted,
                'error': None
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'page_count': 0,
                'is_encrypted': False,
                'error': f'PDF structure validation failed: {str(e)}'
            }
    
    def _scan_for_security_issues(self, file_content: bytes) -> Dict[str, Any]:
        """Scan for potential security issues in PDF content."""
        warnings = []
        critical_issues = []
        
        # Check for suspicious patterns
        for pattern in self.security_patterns:
            if pattern in file_content:
                if pattern in [b'/JavaScript', b'/JS']:
                    critical_issues.append(f"JavaScript detected in PDF - potential security risk")
                elif pattern == b'/Launch':
                    critical_issues.append(f"Launch action detected - potential security risk")
                else:
                    warnings.append(f"Embedded content detected: {pattern.decode('utf-8', errors='ignore')}")
        
        # Check file size vs content ratio (potential zip bomb)
        if len(file_content) < 1000 and b'/FlateDecode' in file_content:
            warnings.append("Highly compressed content detected - monitor processing time")
        
        return {
            'warnings': warnings,
            'critical_issues': critical_issues
        }
    
    def _validate_text_extraction(self, file_path: str) -> Dict[str, Any]:
        """Test text extraction and validate encoding."""
        try:
            doc = fitz.open(file_path)
            
            total_text = ""
            encoding_issues = []
            
            # Sample first few pages for text extraction test
            sample_pages = min(3, len(doc))
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Check for encoding issues
                try:
                    page_text.encode('utf-8')
                except UnicodeEncodeError as e:
                    encoding_issues.append(f"Page {page_num + 1}: Unicode encoding error")
                
                # Check for garbled text (too many replacement characters)
                replacement_ratio = page_text.count('�') / max(len(page_text), 1)
                if replacement_ratio > 0.1:
                    encoding_issues.append(f"Page {page_num + 1}: High replacement character ratio ({replacement_ratio:.2f})")
                
                total_text += page_text
            
            doc.close()
            
            # Determine if meaningful text was extracted
            meaningful_chars = sum(1 for c in total_text if c.isalnum() or c.isspace())
            has_extractable_text = meaningful_chars > 50  # At least 50 meaningful characters
            
            return {
                'has_extractable_text': has_extractable_text,
                'total_characters': len(total_text),
                'meaningful_characters': meaningful_chars,
                'encoding_issues': encoding_issues
            }
            
        except Exception as e:
            return {
                'has_extractable_text': False,
                'total_characters': 0,
                'meaningful_characters': 0,
                'encoding_issues': [f"Text extraction test failed: {str(e)}"]
            }
    
    def get_processing_strategy(self, validation_result: FileValidationResult) -> Dict[str, Any]:
        """Get recommended processing strategy based on validation results."""
        strategy = {
            'use_chunking': validation_result.page_count > 50,
            'chunk_size': min(20, max(5, validation_result.page_count // 10)),
            'needs_ocr': not validation_result.has_text,
            'encoding_safe': len(validation_result.encoding_issues) == 0,
            'security_safe': len([w for w in validation_result.security_warnings if 'critical' in w.lower()]) == 0
        }
        
        return strategy

