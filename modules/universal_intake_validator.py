"""
Universal Intake Validator for Consciousness Recognition System

Extends secure validation to support multiple e-book formats beyond PDF.
Provides format-specific validation and security checks.
"""

import os
import tempfile
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

from .intake_validator import SecureIntakeValidator, FileValidationResult
from .universal_extractor import UniversalTextExtractor


class UniversalIntakeValidator(SecureIntakeValidator):
    """Universal intake validator supporting multiple e-book formats."""
    
    def __init__(self):
        """Initialize the universal intake validator."""
        super().__init__()
        
        # Extend supported formats
        self.allowed_mime_types.extend([
            'application/epub+zip',
            'application/x-mobipocket-ebook',
            'application/vnd.amazon.ebook',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'application/rtf',
            'text/html',
            'text/xml'
        ])
        
        # Format-specific limits
        self.format_limits = {
            '.pdf': {'max_size': 100 * 1024 * 1024, 'max_pages': 1000},
            '.epub': {'max_size': 50 * 1024 * 1024, 'max_chapters': 500},
            '.azw3': {'max_size': 50 * 1024 * 1024, 'max_chapters': 500},
            '.mobi': {'max_size': 50 * 1024 * 1024, 'max_chapters': 500},
            '.txt': {'max_size': 10 * 1024 * 1024, 'max_lines': 100000},
            '.docx': {'max_size': 25 * 1024 * 1024, 'max_pages': 500},
            '.doc': {'max_size': 25 * 1024 * 1024, 'max_pages': 500},
            '.rtf': {'max_size': 25 * 1024 * 1024, 'max_pages': 500},
            '.html': {'max_size': 10 * 1024 * 1024, 'max_elements': 10000},
            '.htm': {'max_size': 10 * 1024 * 1024, 'max_elements': 10000}
        }
        
        # Initialize universal extractor for format detection
        self.extractor = UniversalTextExtractor()
        
        # Format-specific security patterns
        self.format_security_patterns = {
            '.epub': [b'<script', b'javascript:', b'vbscript:', b'<object', b'<embed'],
            '.html': [b'<script', b'javascript:', b'vbscript:', b'<object', b'<embed', b'<iframe'],
            '.htm': [b'<script', b'javascript:', b'vbscript:', b'<object', b'<embed', b'<iframe']
        }
    
    def validate_upload(self, uploaded_file) -> FileValidationResult:
        """
        Comprehensive validation of uploaded file for any supported format.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            FileValidationResult with validation details
        """
        try:
            # Basic checks
            if uploaded_file is None:
                return self._create_invalid_result("No file provided")
            
            # Get file extension
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            # Check if format is supported
            if file_ext not in self.format_limits:
                return self._create_invalid_result(
                    f"Unsupported format: {file_ext}",
                    security_warnings=[f"Format {file_ext} not supported"]
                )
            
            # Check if extractor is available for this format
            supported_formats = self.extractor.get_supported_formats()
            if file_ext not in supported_formats:
                return self._create_invalid_result(
                    f"Extractor not available for {file_ext}",
                    security_warnings=[f"Missing dependencies for {file_ext} format"]
                )
            
            # Format-specific size validation
            format_limits = self.format_limits[file_ext]
            file_size = uploaded_file.size
            
            if file_size > format_limits['max_size']:
                return self._create_invalid_result(
                    f"File too large: {file_size / 1024 / 1024:.1f}MB > {format_limits['max_size'] / 1024 / 1024}MB",
                    security_warnings=[f"File exceeds {file_ext} size limit"]
                )
            
            # Create secure temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                file_content = self._read_file_safely(uploaded_file)
                if file_content is None:
                    return self._create_invalid_result("Unable to read file content")
                
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                # Generate file hash
                file_hash = hashlib.sha256(file_content).hexdigest()
                
                # Format-specific validation
                format_validation = self._validate_format_specific(tmp_path, file_ext, file_content)
                
                # Security scanning
                security_scan = self._scan_format_security(file_content, file_ext)
                
                # Text extraction test
                text_validation = self._validate_text_extraction_universal(tmp_path, file_ext)
                
                # Duplicate detection
                is_duplicate = file_hash in self.processed_hashes
                if not is_duplicate:
                    self.processed_hashes.add(file_hash)
                
                # Compile results
                is_valid = (
                    format_validation['is_valid'] and
                    len(security_scan['critical_issues']) == 0 and
                    not is_duplicate
                )
                
                recommendations = self._generate_format_recommendations(
                    file_ext, format_validation, text_validation, is_duplicate
                )
                
                return FileValidationResult(
                    is_valid=is_valid,
                    file_hash=file_hash,
                    file_size=file_size,
                    page_count=format_validation.get('content_count', 0),
                    has_text=text_validation['has_extractable_text'],
                    is_encrypted=format_validation.get('is_encrypted', False),
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
            return self._create_invalid_result(f"Validation failed: {str(e)}")
    
    def _validate_format_specific(self, file_path: str, file_ext: str, file_content: bytes) -> Dict[str, Any]:
        """Perform format-specific validation."""
        try:
            if file_ext == '.pdf':
                return self._validate_pdf_structure(file_path)
            elif file_ext == '.epub':
                return self._validate_epub_structure(file_path)
            elif file_ext in ['.azw3', '.mobi']:
                return self._validate_kindle_structure(file_path, file_ext)
            elif file_ext == '.txt':
                return self._validate_text_structure(file_path)
            elif file_ext == '.docx':
                return self._validate_docx_structure(file_path)
            elif file_ext in ['.html', '.htm']:
                return self._validate_html_structure(file_path)
            else:
                return {'is_valid': True, 'content_count': 0}
                
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'Format validation failed: {str(e)}',
                'content_count': 0
            }
    
    def _validate_epub_structure(self, file_path: str) -> Dict[str, Any]:
        """Validate EPUB file structure."""
        try:
            # Check if it's a valid ZIP file
            with zipfile.ZipFile(file_path, 'r') as epub_zip:
                file_list = epub_zip.namelist()
                
                # Check for required EPUB files
                required_files = ['META-INF/container.xml', 'mimetype']
                missing_files = [f for f in required_files if f not in file_list]
                
                if missing_files:
                    return {
                        'is_valid': False,
                        'error': f'Invalid EPUB: missing {missing_files}',
                        'content_count': 0
                    }
                
                # Check mimetype
                mimetype = epub_zip.read('mimetype').decode('utf-8').strip()
                if mimetype != 'application/epub+zip':
                    return {
                        'is_valid': False,
                        'error': f'Invalid EPUB mimetype: {mimetype}',
                        'content_count': 0
                    }
                
                # Count content files
                content_files = [f for f in file_list if f.endswith(('.html', '.xhtml', '.xml'))]
                chapter_count = len(content_files)
                
                # Check chapter limit
                max_chapters = self.format_limits['.epub']['max_chapters']
                if chapter_count > max_chapters:
                    return {
                        'is_valid': False,
                        'error': f'Too many chapters: {chapter_count} > {max_chapters}',
                        'content_count': chapter_count
                    }
                
                return {
                    'is_valid': True,
                    'content_count': chapter_count,
                    'is_encrypted': False
                }
                
        except zipfile.BadZipFile:
            return {
                'is_valid': False,
                'error': 'File is not a valid ZIP/EPUB archive',
                'content_count': 0
            }
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'EPUB validation failed: {str(e)}',
                'content_count': 0
            }
    
    def _validate_kindle_structure(self, file_path: str, file_ext: str) -> Dict[str, Any]:
        """Validate Kindle format structure."""
        try:
            # Basic file header check
            with open(file_path, 'rb') as f:
                header = f.read(100)
            
            # Check for Kindle format signatures
            if file_ext == '.azw3':
                # AZW3 files should have specific headers
                if not (b'BOOKMOBI' in header or b'TPZ' in header):
                    return {
                        'is_valid': False,
                        'error': 'Invalid AZW3 file header',
                        'content_count': 0
                    }
            elif file_ext == '.mobi':
                # MOBI files should have BOOKMOBI header
                if b'BOOKMOBI' not in header:
                    return {
                        'is_valid': False,
                        'error': 'Invalid MOBI file header',
                        'content_count': 0
                    }
            
            # For detailed validation, we'd need calibre or similar
            # For now, just check basic structure
            file_size = os.path.getsize(file_path)
            estimated_chapters = max(1, file_size // (50 * 1024))  # Rough estimate
            
            return {
                'is_valid': True,
                'content_count': estimated_chapters,
                'is_encrypted': False  # Would need deeper analysis to detect DRM
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'{file_ext.upper()} validation failed: {str(e)}',
                'content_count': 0
            }
    
    def _validate_text_structure(self, file_path: str) -> Dict[str, Any]:
        """Validate plain text file structure."""
        try:
            # Check file size and line count
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for _ in f)
            
            max_lines = self.format_limits['.txt']['max_lines']
            if line_count > max_lines:
                return {
                    'is_valid': False,
                    'error': f'Too many lines: {line_count} > {max_lines}',
                    'content_count': line_count
                }
            
            return {
                'is_valid': True,
                'content_count': line_count,
                'is_encrypted': False
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'Text validation failed: {str(e)}',
                'content_count': 0
            }
    
    def _validate_docx_structure(self, file_path: str) -> Dict[str, Any]:
        """Validate DOCX file structure."""
        try:
            # Check if it's a valid ZIP file (DOCX is ZIP-based)
            with zipfile.ZipFile(file_path, 'r') as docx_zip:
                file_list = docx_zip.namelist()
                
                # Check for required DOCX files
                required_files = ['[Content_Types].xml', '_rels/.rels']
                missing_files = [f for f in required_files if f not in file_list]
                
                if missing_files:
                    return {
                        'is_valid': False,
                        'error': f'Invalid DOCX: missing {missing_files}',
                        'content_count': 0
                    }
                
                # Estimate page count (rough approximation)
                file_size = os.path.getsize(file_path)
                estimated_pages = max(1, file_size // (100 * 1024))  # Very rough estimate
                
                return {
                    'is_valid': True,
                    'content_count': estimated_pages,
                    'is_encrypted': False  # Would need deeper analysis
                }
                
        except zipfile.BadZipFile:
            return {
                'is_valid': False,
                'error': 'File is not a valid ZIP/DOCX archive',
                'content_count': 0
            }
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'DOCX validation failed: {str(e)}',
                'content_count': 0
            }
    
    def _validate_html_structure(self, file_path: str) -> Dict[str, Any]:
        """Validate HTML file structure."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic HTML structure check
            if not ('<html' in content.lower() or '<body' in content.lower()):
                return {
                    'is_valid': False,
                    'error': 'File does not appear to be valid HTML',
                    'content_count': 0
                }
            
            # Count elements (rough approximation)
            element_count = content.count('<') - content.count('</')
            max_elements = self.format_limits['.html']['max_elements']
            
            if element_count > max_elements:
                return {
                    'is_valid': False,
                    'error': f'Too many HTML elements: {element_count} > {max_elements}',
                    'content_count': element_count
                }
            
            return {
                'is_valid': True,
                'content_count': element_count,
                'is_encrypted': False
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'HTML validation failed: {str(e)}',
                'content_count': 0
            }
    
    def _scan_format_security(self, file_content: bytes, file_ext: str) -> Dict[str, Any]:
        """Scan for format-specific security issues."""
        warnings = []
        critical_issues = []
        
        # Use base security scanning
        base_scan = self._scan_for_security_issues(file_content)
        warnings.extend(base_scan['warnings'])
        critical_issues.extend(base_scan['critical_issues'])
        
        # Format-specific security patterns
        if file_ext in self.format_security_patterns:
            patterns = self.format_security_patterns[file_ext]
            
            for pattern in patterns:
                if pattern in file_content:
                    if pattern in [b'<script', b'javascript:', b'vbscript:']:
                        critical_issues.append(f"Potentially malicious script content detected in {file_ext}")
                    else:
                        warnings.append(f"Embedded content detected in {file_ext}: {pattern.decode('utf-8', errors='ignore')}")
        
        return {
            'warnings': warnings,
            'critical_issues': critical_issues
        }
    
    def _validate_text_extraction_universal(self, file_path: str, file_ext: str) -> Dict[str, Any]:
        """Test text extraction for any supported format."""
        try:
            # Use universal extractor to test extraction
            extraction_result = self.extractor.extract_text(file_path)
            
            if not extraction_result['success']:
                return {
                    'has_extractable_text': False,
                    'total_characters': 0,
                    'meaningful_characters': 0,
                    'encoding_issues': [f"Text extraction failed: {extraction_result['error']}"]
                }
            
            text = extraction_result['text']
            encoding_issues = []
            
            # Check for encoding issues
            try:
                text.encode('utf-8')
            except UnicodeEncodeError as e:
                encoding_issues.append(f"Unicode encoding error: {str(e)}")
            
            # Check for garbled text
            replacement_ratio = text.count('�') / max(len(text), 1)
            if replacement_ratio > 0.1:
                encoding_issues.append(f"High replacement character ratio: {replacement_ratio:.2f}")
            
            # Determine if meaningful text was extracted
            meaningful_chars = sum(1 for c in text if c.isalnum() or c.isspace())
            has_extractable_text = meaningful_chars > 50
            
            return {
                'has_extractable_text': has_extractable_text,
                'total_characters': len(text),
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
    
    def _generate_format_recommendations(
        self, 
        file_ext: str, 
        format_validation: Dict[str, Any], 
        text_validation: Dict[str, Any], 
        is_duplicate: bool
    ) -> List[str]:
        """Generate format-specific processing recommendations."""
        recommendations = []
        
        if is_duplicate:
            recommendations.append("File already processed - skipping to avoid duplicates")
            return recommendations
        
        # Format-specific recommendations
        if file_ext == '.pdf':
            if format_validation.get('content_count', 0) > 200:
                recommendations.append("Large PDF detected - will use chunked processing")
        
        elif file_ext == '.epub':
            recommendations.append("EPUB format detected - excellent text extraction quality expected")
            if format_validation.get('content_count', 0) > 100:
                recommendations.append("Large EPUB with many chapters - processing may take time")
        
        elif file_ext in ['.azw3', '.mobi']:
            recommendations.append(f"{file_ext.upper()} format detected - requires calibre for optimal extraction")
        
        elif file_ext == '.txt':
            recommendations.append("Plain text format - perfect extraction quality")
        
        elif file_ext == '.docx':
            recommendations.append("DOCX format detected - good text extraction expected")
        
        elif file_ext in ['.html', '.htm']:
            recommendations.append("HTML format detected - will extract text content only")
        
        # Text extraction recommendations
        if not text_validation['has_extractable_text']:
            recommendations.append("No extractable text found - check file content")
        
        if text_validation['encoding_issues']:
            recommendations.append("Text encoding issues detected - may affect quality")
        
        return recommendations
    
    def _create_invalid_result(self, error_message: str, security_warnings: List[str] = None) -> FileValidationResult:
        """Create an invalid validation result."""
        return FileValidationResult(
            is_valid=False,
            file_hash="",
            file_size=0,
            page_count=0,
            has_text=False,
            is_encrypted=False,
            encoding_issues=[],
            security_warnings=security_warnings or [],
            processing_recommendations=[],
            error_message=error_message
        )
    
    def get_supported_formats_info(self) -> Dict[str, Any]:
        """Get information about all supported formats."""
        supported_formats = self.extractor.get_supported_formats()
        
        format_info = {}
        for fmt in supported_formats:
            info = self.extractor.get_format_info(f"test{fmt}")
            format_info[fmt] = {
                'name': info['info']['name'],
                'description': info['info']['description'],
                'max_size_mb': self.format_limits.get(fmt, {}).get('max_size', 0) / 1024 / 1024,
                'extraction_quality': info['info'].get('extraction_quality', 'Unknown'),
                'available': fmt in supported_formats
            }
        
        return format_info

