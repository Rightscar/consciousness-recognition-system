"""
Comprehensive Text Validation Framework
=======================================

Ensures all text inputs are properly normalized and validated before processing.
Prevents "expected string or bytes-like object, got 'list'" errors system-wide.

Author: Consciousness Recognition System
"""

import streamlit as st
from typing import Any, Optional, Union, List, Dict
import logging

# Configure logging for validation events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextValidationError(Exception):
    """Custom exception for text validation failures."""
    pass


class TextValidator:
    """
    Comprehensive text validation and normalization system.
    
    Ensures all text inputs are properly formatted strings before processing.
    Provides multiple layers of defense against type-related errors.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the text validator.
        
        Args:
            strict_mode: If True, raises exceptions on validation failures.
                        If False, attempts graceful recovery with warnings.
        """
        self.strict_mode = strict_mode
        self.validation_stats = {
            'total_validations': 0,
            'list_conversions': 0,
            'type_conversions': 0,
            'failures': 0,
            'warnings': 0
        }
    
    def validate_and_normalize_text(
        self, 
        text: Any, 
        source: str = "unknown",
        show_ui_feedback: bool = True
    ) -> Optional[str]:
        """
        Validate and normalize text input to ensure it's a proper string.
        
        Args:
            text: Input text of any type
            source: Description of where the text came from (for logging)
            show_ui_feedback: Whether to show Streamlit UI feedback
            
        Returns:
            Normalized string or None if validation fails in strict mode
            
        Raises:
            TextValidationError: If validation fails and strict_mode is True
        """
        self.validation_stats['total_validations'] += 1
        
        try:
            # Log the validation attempt
            logger.info(f"Validating text from {source}: type={type(text)}")
            
            # Case 1: Already a string - validate content
            if isinstance(text, str):
                return self._validate_string_content(text, source, show_ui_feedback)
            
            # Case 2: List of strings - convert to single string
            elif isinstance(text, list):
                return self._convert_list_to_string(text, source, show_ui_feedback)
            
            # Case 3: Other types - attempt conversion
            else:
                return self._convert_other_types(text, source, show_ui_feedback)
                
        except Exception as e:
            self.validation_stats['failures'] += 1
            error_msg = f"Text validation failed for {source}: {str(e)}"
            logger.error(error_msg)
            
            if show_ui_feedback:
                st.error(f"❌ **Text Validation Error ({source}):** {str(e)}")
            
            if self.strict_mode:
                raise TextValidationError(error_msg)
            
            return None
    
    def _validate_string_content(
        self, 
        text: str, 
        source: str, 
        show_ui_feedback: bool
    ) -> str:
        """Validate string content and provide feedback."""
        
        # Check for empty or whitespace-only strings
        if not text or not text.strip():
            warning_msg = f"Empty or whitespace-only text from {source}"
            self.validation_stats['warnings'] += 1
            
            if show_ui_feedback:
                st.warning(f"⚠️ **Empty Text Warning ({source}):** No meaningful content found")
            
            logger.warning(warning_msg)
            
            if self.strict_mode:
                raise TextValidationError(warning_msg)
        
        # Check for very short content
        elif len(text.strip()) < 10:
            warning_msg = f"Very short text from {source}: {len(text)} characters"
            self.validation_stats['warnings'] += 1
            
            if show_ui_feedback:
                st.warning(f"⚠️ **Short Text Warning ({source}):** Only {len(text)} characters found")
            
            logger.warning(warning_msg)
        
        # Success case
        else:
            if show_ui_feedback:
                st.success(f"✅ **Text Validated ({source}):** {len(text)} characters, type: string")
        
        return text
    
    def _convert_list_to_string(
        self, 
        text_list: List[Any], 
        source: str, 
        show_ui_feedback: bool
    ) -> str:
        """Convert list to string with proper joining."""
        
        self.validation_stats['list_conversions'] += 1
        
        if show_ui_feedback:
            st.info(f"📄 **Converting List to String ({source}):** Found {len(text_list)} items")
        
        # Filter out empty/None items and convert to strings
        valid_items = []
        for item in text_list:
            if item is not None:
                item_str = str(item).strip()
                if item_str:  # Only add non-empty strings
                    valid_items.append(item_str)
        
        if not valid_items:
            warning_msg = f"List from {source} contains no valid text items"
            self.validation_stats['warnings'] += 1
            
            if show_ui_feedback:
                st.warning(f"⚠️ **Empty List Warning ({source}):** No valid text items found")
            
            if self.strict_mode:
                raise TextValidationError(warning_msg)
            
            return ""
        
        # Join with double newlines to preserve structure
        result = "\n\n".join(valid_items)
        
        if show_ui_feedback:
            st.success(f"✅ **List Conversion Complete ({source}):** {len(valid_items)} items → {len(result)} characters")
        
        logger.info(f"Converted list to string for {source}: {len(text_list)} items → {len(result)} chars")
        
        return result
    
    def _convert_other_types(
        self, 
        text: Any, 
        source: str, 
        show_ui_feedback: bool
    ) -> str:
        """Convert other types to string with validation."""
        
        self.validation_stats['type_conversions'] += 1
        
        if show_ui_feedback:
            st.warning(f"⚠️ **Type Conversion ({source}):** Converting {type(text)} to string")
        
        # Attempt conversion
        try:
            result = str(text)
            
            if show_ui_feedback:
                st.success(f"✅ **Type Conversion Complete ({source}):** {type(text)} → string ({len(result)} chars)")
            
            logger.info(f"Converted {type(text)} to string for {source}: {len(result)} chars")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to convert {type(text)} to string for {source}: {str(e)}"
            
            if show_ui_feedback:
                st.error(f"❌ **Type Conversion Failed ({source}):** {str(e)}")
            
            raise TextValidationError(error_msg)
    
    def validate_extraction_result(
        self, 
        extraction_result: Dict[str, Any], 
        source: str = "extraction"
    ) -> Optional[str]:
        """
        Validate and normalize text from extraction results.
        
        Args:
            extraction_result: Dictionary containing extraction results
            source: Description of the extraction source
            
        Returns:
            Normalized text string or None if validation fails
        """
        
        if not isinstance(extraction_result, dict):
            if self.strict_mode:
                raise TextValidationError(f"Extraction result from {source} is not a dictionary")
            return None
        
        if not extraction_result.get('success', False):
            error_msg = extraction_result.get('error', 'Unknown extraction error')
            st.error(f"❌ **Extraction Failed ({source}):** {error_msg}")
            return None
        
        text = extraction_result.get('text', '')
        return self.validate_and_normalize_text(text, f"{source}_extraction")
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics for monitoring."""
        return self.validation_stats.copy()
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'list_conversions': 0,
            'type_conversions': 0,
            'failures': 0,
            'warnings': 0
        }
    
    def display_stats(self):
        """Display validation statistics in Streamlit UI."""
        stats = self.get_validation_stats()
        
        st.subheader("📊 Text Validation Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Validations", stats['total_validations'])
        with col2:
            st.metric("List Conversions", stats['list_conversions'])
        with col3:
            st.metric("Type Conversions", stats['type_conversions'])
        with col4:
            st.metric("Warnings", stats['warnings'])
        with col5:
            st.metric("Failures", stats['failures'])


# Global validator instance for system-wide use
_global_validator = TextValidator(strict_mode=False)


def validate_text(
    text: Any, 
    source: str = "unknown", 
    show_ui_feedback: bool = True
) -> Optional[str]:
    """
    Global function for text validation - convenience wrapper.
    
    Args:
        text: Input text of any type
        source: Description of where the text came from
        show_ui_feedback: Whether to show Streamlit UI feedback
        
    Returns:
        Normalized string or None if validation fails
    """
    return _global_validator.validate_and_normalize_text(text, source, show_ui_feedback)


def validate_extraction(
    extraction_result: Dict[str, Any], 
    source: str = "extraction"
) -> Optional[str]:
    """
    Global function for extraction result validation - convenience wrapper.
    
    Args:
        extraction_result: Dictionary containing extraction results
        source: Description of the extraction source
        
    Returns:
        Normalized text string or None if validation fails
    """
    return _global_validator.validate_extraction_result(extraction_result, source)


def get_validator_stats() -> Dict[str, int]:
    """Get global validation statistics."""
    return _global_validator.get_validation_stats()


def display_validator_stats():
    """Display global validation statistics in Streamlit UI."""
    _global_validator.display_stats()


def reset_validator_stats():
    """Reset global validation statistics."""
    _global_validator.reset_stats()


# Emergency validation function for critical paths
def emergency_text_fix(text: Any, context: str = "emergency") -> str:
    """
    Emergency text normalization for critical code paths.
    Always returns a string, never fails.
    
    Args:
        text: Input of any type
        context: Context description for logging
        
    Returns:
        String representation of the input
    """
    try:
        if isinstance(text, str):
            return text
        elif isinstance(text, list):
            return "\n\n".join(str(item) for item in text if item)
        else:
            return str(text)
    except Exception:
        logger.error(f"Emergency text fix failed for {context}")
        return ""  # Return empty string as last resort

