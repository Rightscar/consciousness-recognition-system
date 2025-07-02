"""
Enhanced Text Validation Framework with Comprehensive Input Normalization
=========================================================================

Bulletproof validation system that ensures all text inputs are properly normalized
and validated before processing. Prevents "expected string or bytes-like object, 
got 'list'" errors system-wide with multiple defensive layers.

Author: Consciousness Recognition System
Version: 2.0 - Enhanced with comprehensive normalization
"""

import streamlit as st
from typing import Any, Optional, Union, List, Dict, Tuple
import logging
import traceback
import re
from pathlib import Path
import json

# Configure logging for validation events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextValidationError(Exception):
    """Custom exception for text validation failures."""
    pass


class EnhancedTextValidator:
    """
    Enhanced text validation and normalization system with comprehensive input handling.
    
    Provides multiple layers of defense against type-related errors and ensures
    all text inputs are properly formatted strings before processing.
    """
    
    def __init__(self, strict_mode: bool = False, debug_mode: bool = False):
        """
        Initialize the enhanced text validator.
        
        Args:
            strict_mode: If True, raises exceptions on validation failures.
                        If False, attempts graceful recovery with warnings.
            debug_mode: If True, provides detailed debugging information.
        """
        self.strict_mode = strict_mode
        self.debug_mode = debug_mode
        self.validation_stats = {
            'total_validations': 0,
            'list_conversions': 0,
            'type_conversions': 0,
            'failures': 0,
            'warnings': 0,
            'emergency_fixes': 0,
            'encoding_fixes': 0,
            'content_repairs': 0
        }
        
        # Validation history for debugging
        self.validation_history = []
        
        # Content repair patterns
        self.repair_patterns = [
            (r'\n{3,}', '\n\n'),  # Multiple newlines to double
            (r'\s{3,}', ' '),     # Multiple spaces to single
            (r'\t+', ' '),        # Tabs to spaces
            (r'\r\n', '\n'),      # Windows line endings
            (r'\r', '\n'),        # Mac line endings
        ]
    
    def validate_and_normalize_text(
        self, 
        text: Any, 
        source: str = "unknown",
        show_ui_feedback: bool = True,
        emergency_mode: bool = False
    ) -> Optional[str]:
        """
        Enhanced validation and normalization with comprehensive input handling.
        
        Args:
            text: Input text of any type
            source: Description of where the text came from (for logging)
            show_ui_feedback: Whether to show Streamlit UI feedback
            emergency_mode: If True, always returns a string (never fails)
            
        Returns:
            Normalized string or None if validation fails in strict mode
            
        Raises:
            TextValidationError: If validation fails and strict_mode is True
        """
        self.validation_stats['total_validations'] += 1
        validation_start_time = time.time() if 'time' in globals() else 0
        
        try:
            # Record validation attempt
            validation_record = {
                'source': source,
                'input_type': type(text).__name__,
                'input_length': self._safe_length(text),
                'timestamp': validation_start_time
            }
            
            if self.debug_mode and show_ui_feedback:
                st.write(f"🔍 **DEBUG ({source}):** Input type: `{type(text)}`, Length: {self._safe_length(text)}")
            
            # Emergency mode - always return something
            if emergency_mode:
                result = self._emergency_normalize(text, source, show_ui_feedback)
                validation_record['result'] = 'emergency_success'
                validation_record['output_length'] = len(result) if result else 0
                self.validation_history.append(validation_record)
                return result
            
            # Layer 1: Type-specific validation and normalization
            normalized_text = self._normalize_by_type(text, source, show_ui_feedback)
            
            # Layer 2: Content validation and repair
            validated_text = self._validate_and_repair_content(normalized_text, source, show_ui_feedback)
            
            # Layer 3: Encoding validation and cleanup
            final_text = self._validate_encoding(validated_text, source, show_ui_feedback)
            
            # Layer 4: Final quality checks
            quality_checked_text = self._final_quality_check(final_text, source, show_ui_feedback)
            
            # Record successful validation
            validation_record['result'] = 'success'
            validation_record['output_length'] = len(quality_checked_text) if quality_checked_text else 0
            validation_record['processing_time'] = time.time() - validation_start_time if validation_start_time else 0
            self.validation_history.append(validation_record)
            
            return quality_checked_text
                
        except Exception as e:
            self.validation_stats['failures'] += 1
            error_msg = f"Enhanced text validation failed for {source}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Record failed validation
            validation_record['result'] = 'failure'
            validation_record['error'] = str(e)
            self.validation_history.append(validation_record)
            
            if show_ui_feedback:
                st.error(f"❌ **Enhanced Validation Error ({source}):** {str(e)}")
                if self.debug_mode:
                    st.code(traceback.format_exc())
            
            # Emergency fallback
            if emergency_mode or not self.strict_mode:
                emergency_result = self._emergency_normalize(text, source, show_ui_feedback)
                return emergency_result
            
            if self.strict_mode:
                raise TextValidationError(error_msg)
            
            return None
    
    def _normalize_by_type(self, text: Any, source: str, show_ui_feedback: bool) -> str:
        """Layer 1: Type-specific normalization."""
        
        # Case 1: Already a string
        if isinstance(text, str):
            if show_ui_feedback and self.debug_mode:
                st.success(f"✅ **String Input ({source}):** {len(text)} characters")
            return text
        
        # Case 2: List of items
        elif isinstance(text, (list, tuple)):
            return self._normalize_list_input(text, source, show_ui_feedback)
        
        # Case 3: Dictionary (extraction result)
        elif isinstance(text, dict):
            return self._normalize_dict_input(text, source, show_ui_feedback)
        
        # Case 4: Bytes
        elif isinstance(text, bytes):
            return self._normalize_bytes_input(text, source, show_ui_feedback)
        
        # Case 5: Other types
        else:
            return self._normalize_other_types(text, source, show_ui_feedback)
    
    def _normalize_list_input(self, text_list: Union[List, Tuple], source: str, show_ui_feedback: bool) -> str:
        """Normalize list/tuple input with comprehensive handling."""
        
        self.validation_stats['list_conversions'] += 1
        
        if show_ui_feedback:
            st.info(f"📄 **List Conversion ({source}):** Processing {len(text_list)} items")
        
        # Handle nested structures
        flattened_items = []
        for item in text_list:
            if isinstance(item, (list, tuple)):
                # Recursively flatten nested lists
                flattened_items.extend(self._flatten_nested_list(item))
            elif isinstance(item, dict):
                # Extract text from dictionary items
                text_content = self._extract_text_from_dict(item)
                if text_content:
                    flattened_items.append(text_content)
            elif item is not None:
                # Convert to string and clean
                item_str = str(item).strip()
                if item_str:
                    flattened_items.append(item_str)
        
        if not flattened_items:
            warning_msg = f"List from {source} contains no valid text items"
            self.validation_stats['warnings'] += 1
            
            if show_ui_feedback:
                st.warning(f"⚠️ **Empty List Warning ({source}):** No valid text items found")
            
            if self.strict_mode:
                raise TextValidationError(warning_msg)
            
            return ""
        
        # Join with appropriate separators
        if len(flattened_items) == 1:
            result = flattened_items[0]
            separator_used = "none"
        else:
            # Use double newlines for paragraph separation
            result = "\n\n".join(flattened_items)
            separator_used = "double_newline"
        
        if show_ui_feedback:
            st.success(f"✅ **List Conversion Complete ({source}):** {len(flattened_items)} items → {len(result)} characters (separator: {separator_used})")
        
        logger.info(f"Converted list to string for {source}: {len(text_list)} items → {len(result)} chars")
        
        return result
    
    def _normalize_dict_input(self, text_dict: Dict, source: str, show_ui_feedback: bool) -> str:
        """Normalize dictionary input (e.g., extraction results)."""
        
        if show_ui_feedback:
            st.info(f"📋 **Dictionary Conversion ({source}):** Extracting text content")
        
        # Try common text fields
        text_fields = ['text', 'content', 'body', 'message', 'data', 'result']
        
        for field in text_fields:
            if field in text_dict:
                extracted_text = text_dict[field]
                if extracted_text:
                    if show_ui_feedback:
                        st.success(f"✅ **Dictionary Text Extracted ({source}):** From '{field}' field")
                    return self._normalize_by_type(extracted_text, f"{source}_dict_{field}", False)
        
        # If no standard fields, try to extract all string values
        string_values = []
        for key, value in text_dict.items():
            if isinstance(value, str) and value.strip():
                string_values.append(value.strip())
        
        if string_values:
            result = "\n\n".join(string_values)
            if show_ui_feedback:
                st.success(f"✅ **Dictionary Conversion ({source}):** Extracted {len(string_values)} text values")
            return result
        
        # Last resort: convert entire dict to string
        result = str(text_dict)
        if show_ui_feedback:
            st.warning(f"⚠️ **Dictionary Fallback ({source}):** Converted entire dict to string")
        
        return result
    
    def _normalize_bytes_input(self, text_bytes: bytes, source: str, show_ui_feedback: bool) -> str:
        """Normalize bytes input with encoding detection."""
        
        if show_ui_feedback:
            st.info(f"🔤 **Bytes Conversion ({source}):** Decoding {len(text_bytes)} bytes")
        
        # Try common encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'ascii', 'cp1252']
        
        for encoding in encodings:
            try:
                result = text_bytes.decode(encoding)
                if show_ui_feedback:
                    st.success(f"✅ **Bytes Decoded ({source}):** Using {encoding} encoding")
                return result
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Fallback: decode with error handling
        result = text_bytes.decode('utf-8', errors='replace')
        if show_ui_feedback:
            st.warning(f"⚠️ **Bytes Fallback ({source}):** Used UTF-8 with error replacement")
        
        return result
    
    def _normalize_other_types(self, text: Any, source: str, show_ui_feedback: bool) -> str:
        """Normalize other types with comprehensive conversion."""
        
        self.validation_stats['type_conversions'] += 1
        
        if show_ui_feedback:
            st.warning(f"⚠️ **Type Conversion ({source}):** Converting {type(text)} to string")
        
        try:
            # Handle None
            if text is None:
                if show_ui_feedback:
                    st.warning(f"⚠️ **None Input ({source}):** Converting to empty string")
                return ""
            
            # Handle numeric types
            if isinstance(text, (int, float, complex)):
                result = str(text)
                if show_ui_feedback:
                    st.success(f"✅ **Numeric Conversion ({source}):** {type(text).__name__} → string")
                return result
            
            # Handle boolean
            if isinstance(text, bool):
                result = str(text)
                if show_ui_feedback:
                    st.success(f"✅ **Boolean Conversion ({source}):** {text} → string")
                return result
            
            # Generic string conversion
            result = str(text)
            
            if show_ui_feedback:
                st.success(f"✅ **Generic Conversion ({source}):** {type(text).__name__} → string ({len(result)} chars)")
            
            logger.info(f"Converted {type(text)} to string for {source}: {len(result)} chars")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to convert {type(text)} to string for {source}: {str(e)}"
            
            if show_ui_feedback:
                st.error(f"❌ **Type Conversion Failed ({source}):** {str(e)}")
            
            raise TextValidationError(error_msg)
    
    def _validate_and_repair_content(self, text: str, source: str, show_ui_feedback: bool) -> str:
        """Layer 2: Content validation and repair."""
        
        if not text:
            return text
        
        original_length = len(text)
        repaired_text = text
        repairs_made = []
        
        # Apply repair patterns
        for pattern, replacement in self.repair_patterns:
            before_repair = repaired_text
            repaired_text = re.sub(pattern, replacement, repaired_text)
            if repaired_text != before_repair:
                repairs_made.append(f"{pattern} → {replacement}")
        
        # Remove control characters except newlines and tabs
        control_chars_removed = 0
        clean_text = ""
        for char in repaired_text:
            if ord(char) < 32 and char not in '\n\t':
                control_chars_removed += 1
            else:
                clean_text += char
        
        if control_chars_removed > 0:
            repairs_made.append(f"Removed {control_chars_removed} control characters")
            repaired_text = clean_text
        
        # Report repairs
        if repairs_made:
            self.validation_stats['content_repairs'] += 1
            if show_ui_feedback:
                st.info(f"🔧 **Content Repair ({source}):** {len(repairs_made)} repairs made")
                if self.debug_mode:
                    for repair in repairs_made:
                        st.write(f"  • {repair}")
        
        # Check for significant content loss
        final_length = len(repaired_text)
        if original_length > 0 and final_length < original_length * 0.5:
            warning_msg = f"Significant content loss during repair: {original_length} → {final_length} chars"
            self.validation_stats['warnings'] += 1
            if show_ui_feedback:
                st.warning(f"⚠️ **Content Loss Warning ({source}):** {warning_msg}")
        
        return repaired_text
    
    def _validate_encoding(self, text: str, source: str, show_ui_feedback: bool) -> str:
        """Layer 3: Encoding validation and cleanup."""
        
        if not text:
            return text
        
        # Check for encoding issues
        encoding_issues = []
        
        # Check for replacement characters
        replacement_chars = text.count('�')
        if replacement_chars > 0:
            encoding_issues.append(f"{replacement_chars} replacement characters (�)")
        
        # Check for common encoding artifacts
        artifacts = [
            ('â€™', "'"),  # Smart apostrophe
            ('â€œ', '"'),  # Smart quote open
            ('â€', '"'),   # Smart quote close
            ('â€"', '—'),  # Em dash
            ('â€"', '–'),  # En dash
        ]
        
        fixed_text = text
        fixes_applied = []
        
        for artifact, replacement in artifacts:
            if artifact in fixed_text:
                fixed_text = fixed_text.replace(artifact, replacement)
                fixes_applied.append(f"{artifact} → {replacement}")
        
        if fixes_applied:
            self.validation_stats['encoding_fixes'] += 1
            if show_ui_feedback:
                st.info(f"🔤 **Encoding Fix ({source}):** {len(fixes_applied)} artifacts fixed")
                if self.debug_mode:
                    for fix in fixes_applied:
                        st.write(f"  • {fix}")
        
        # Report encoding issues
        if encoding_issues and show_ui_feedback:
            st.warning(f"⚠️ **Encoding Issues ({source}):** {', '.join(encoding_issues)}")
        
        return fixed_text
    
    def _final_quality_check(self, text: str, source: str, show_ui_feedback: bool) -> str:
        """Layer 4: Final quality checks."""
        
        if not text:
            if show_ui_feedback:
                st.warning(f"⚠️ **Empty Text ({source}):** No content after validation")
            return text
        
        # Quality metrics
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.splitlines())
        
        # Quality assessment
        quality_issues = []
        
        if char_count < 10:
            quality_issues.append("Very short content")
        
        if word_count < 3:
            quality_issues.append("Very few words")
        
        if char_count > 0 and word_count / char_count < 0.1:
            quality_issues.append("Low word density")
        
        # Report quality
        if show_ui_feedback:
            if quality_issues:
                st.warning(f"⚠️ **Quality Issues ({source}):** {', '.join(quality_issues)}")
            else:
                st.success(f"✅ **Quality Check Passed ({source}):** {char_count} chars, {word_count} words, {line_count} lines")
        
        return text
    
    def _emergency_normalize(self, text: Any, source: str, show_ui_feedback: bool) -> str:
        """Emergency normalization that never fails."""
        
        self.validation_stats['emergency_fixes'] += 1
        
        try:
            if isinstance(text, str):
                return text
            elif isinstance(text, (list, tuple)):
                return "\n\n".join(str(item) for item in text if item)
            elif isinstance(text, dict):
                return str(text.get('text', str(text)))
            elif isinstance(text, bytes):
                return text.decode('utf-8', errors='replace')
            else:
                return str(text) if text is not None else ""
        except Exception:
            if show_ui_feedback:
                st.error(f"🚨 **Emergency Fix Failed ({source}):** Returning empty string")
            return ""
    
    def _safe_length(self, obj: Any) -> int:
        """Safely get length of any object."""
        try:
            if hasattr(obj, '__len__'):
                return len(obj)
            else:
                return len(str(obj))
        except:
            return 0
    
    def _flatten_nested_list(self, nested_list: Union[List, Tuple]) -> List[str]:
        """Recursively flatten nested lists."""
        result = []
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                result.extend(self._flatten_nested_list(item))
            else:
                result.append(str(item))
        return result
    
    def _extract_text_from_dict(self, item_dict: Dict) -> Optional[str]:
        """Extract text content from dictionary."""
        text_fields = ['text', 'content', 'body', 'message', 'value']
        for field in text_fields:
            if field in item_dict and item_dict[field]:
                return str(item_dict[field])
        return None
    
    def validate_extraction_result(
        self, 
        extraction_result: Dict[str, Any], 
        source: str = "extraction"
    ) -> Optional[str]:
        """
        Enhanced validation for extraction results with comprehensive error handling.
        
        Args:
            extraction_result: Dictionary containing extraction results
            source: Description of the extraction source
            
        Returns:
            Normalized text string or None if validation fails
        """
        
        if not isinstance(extraction_result, dict):
            error_msg = f"Extraction result from {source} is not a dictionary: {type(extraction_result)}"
            st.error(f"❌ **Invalid Extraction Result ({source}):** {error_msg}")
            
            # Try to extract text anyway
            return self.validate_and_normalize_text(extraction_result, f"{source}_invalid_dict", emergency_mode=True)
        
        # Check success status
        if not extraction_result.get('success', False):
            error_msg = extraction_result.get('error', 'Unknown extraction error')
            st.error(f"❌ **Extraction Failed ({source}):** {error_msg}")
            
            # Try to extract any available text
            text = extraction_result.get('text', '')
            if text:
                st.info(f"🔄 **Attempting Recovery ({source}):** Found text despite failure status")
                return self.validate_and_normalize_text(text, f"{source}_recovery", emergency_mode=True)
            
            return None
        
        # Extract text with validation
        text = extraction_result.get('text', '')
        return self.validate_and_normalize_text(text, f"{source}_extraction")
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get comprehensive validation statistics."""
        return self.validation_stats.copy()
    
    def get_validation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent validation history for debugging."""
        return self.validation_history[-limit:] if self.validation_history else []
    
    def reset_stats(self):
        """Reset validation statistics and history."""
        self.validation_stats = {
            'total_validations': 0,
            'list_conversions': 0,
            'type_conversions': 0,
            'failures': 0,
            'warnings': 0,
            'emergency_fixes': 0,
            'encoding_fixes': 0,
            'content_repairs': 0
        }
        self.validation_history = []
    
    def display_comprehensive_stats(self):
        """Display comprehensive validation statistics in Streamlit UI."""
        stats = self.get_validation_stats()
        
        st.subheader("📊 Enhanced Text Validation Statistics")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Validations", stats['total_validations'])
        with col2:
            st.metric("Success Rate", 
                     f"{((stats['total_validations'] - stats['failures']) / max(stats['total_validations'], 1) * 100):.1f}%")
        with col3:
            st.metric("Emergency Fixes", stats['emergency_fixes'])
        with col4:
            st.metric("Failures", stats['failures'])
        
        # Detailed metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("List Conversions", stats['list_conversions'])
        with col6:
            st.metric("Type Conversions", stats['type_conversions'])
        with col7:
            st.metric("Content Repairs", stats['content_repairs'])
        with col8:
            st.metric("Encoding Fixes", stats['encoding_fixes'])
        
        # Recent validation history
        if self.debug_mode:
            st.subheader("🔍 Recent Validation History")
            history = self.get_validation_history()
            if history:
                for i, record in enumerate(reversed(history)):
                    with st.expander(f"Validation {len(history) - i}: {record['source']} ({record['result']})"):
                        st.json(record)
            else:
                st.info("No validation history available")


# Global enhanced validator instance
_global_enhanced_validator = EnhancedTextValidator(strict_mode=False, debug_mode=False)


def validate_text_enhanced(
    text: Any, 
    source: str = "unknown", 
    show_ui_feedback: bool = True,
    emergency_mode: bool = False
) -> Optional[str]:
    """
    Enhanced global function for text validation with comprehensive normalization.
    
    Args:
        text: Input text of any type
        source: Description of where the text came from
        show_ui_feedback: Whether to show Streamlit UI feedback
        emergency_mode: If True, always returns a string (never fails)
        
    Returns:
        Normalized string or None if validation fails
    """
    return _global_enhanced_validator.validate_and_normalize_text(
        text, source, show_ui_feedback, emergency_mode
    )


def validate_extraction_enhanced(
    extraction_result: Dict[str, Any], 
    source: str = "extraction"
) -> Optional[str]:
    """
    Enhanced global function for extraction result validation.
    
    Args:
        extraction_result: Dictionary containing extraction results
        source: Description of the extraction source
        
    Returns:
        Normalized text string or None if validation fails
    """
    return _global_enhanced_validator.validate_extraction_result(extraction_result, source)


def emergency_text_fix_enhanced(text: Any, context: str = "emergency") -> str:
    """
    Enhanced emergency text normalization that never fails.
    
    Args:
        text: Input of any type
        context: Context description for logging
        
    Returns:
        String representation of the input
    """
    return _global_enhanced_validator._emergency_normalize(text, context, False)


def get_enhanced_validator_stats() -> Dict[str, int]:
    """Get enhanced validation statistics."""
    return _global_enhanced_validator.get_validation_stats()


def display_enhanced_validator_stats():
    """Display enhanced validation statistics in Streamlit UI."""
    _global_enhanced_validator.display_comprehensive_stats()


def reset_enhanced_validator_stats():
    """Reset enhanced validation statistics."""
    _global_enhanced_validator.reset_stats()


def enable_debug_mode():
    """Enable debug mode for detailed validation feedback."""
    _global_enhanced_validator.debug_mode = True


def disable_debug_mode():
    """Disable debug mode."""
    _global_enhanced_validator.debug_mode = False


# Compatibility layer - enhanced versions of original functions
def validate_text(text: Any, source: str = "unknown", show_ui_feedback: bool = True) -> Optional[str]:
    """Compatibility wrapper for enhanced validation."""
    return validate_text_enhanced(text, source, show_ui_feedback, emergency_mode=False)


def validate_extraction(extraction_result: Dict[str, Any], source: str = "extraction") -> Optional[str]:
    """Compatibility wrapper for enhanced extraction validation."""
    return validate_extraction_enhanced(extraction_result, source)


def emergency_text_fix(text: Any, context: str = "emergency") -> str:
    """Compatibility wrapper for enhanced emergency fix."""
    return emergency_text_fix_enhanced(text, context)

