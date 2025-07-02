"""
Early Type Checking and Emergency Conversion System
==================================================

Proactive type checking system that catches potential issues before they reach
the main processing pipeline. Provides automatic conversion and emergency
fallback mechanisms to prevent system crashes.

Author: Consciousness Recognition System
Version: 1.0 - Early Detection and Emergency Conversion
"""

import streamlit as st
from typing import Any, Optional, Union, List, Dict, Tuple, Callable
import logging
import traceback
import time
from functools import wraps
import inspect

# Configure logging
logger = logging.getLogger(__name__)


class TypeCheckError(Exception):
    """Custom exception for type checking failures."""
    pass


class EarlyTypeChecker:
    """
    Early type checking system with automatic conversion and emergency fallbacks.
    
    Provides proactive detection of type issues before they cause system failures.
    """
    
    def __init__(self, auto_convert: bool = True, emergency_mode: bool = True):
        """
        Initialize the early type checker.
        
        Args:
            auto_convert: If True, automatically converts types when possible
            emergency_mode: If True, provides emergency fallbacks for critical failures
        """
        self.auto_convert = auto_convert
        self.emergency_mode = emergency_mode
        self.check_stats = {
            'total_checks': 0,
            'type_mismatches': 0,
            'auto_conversions': 0,
            'emergency_conversions': 0,
            'failures': 0
        }
        self.type_history = []
    
    def check_text_input(
        self, 
        value: Any, 
        source: str = "unknown",
        expected_type: type = str,
        show_feedback: bool = True
    ) -> Tuple[bool, Any, str]:
        """
        Early type check for text inputs with automatic conversion.
        
        Args:
            value: Input value to check
            source: Source description for logging
            expected_type: Expected type (default: str)
            show_feedback: Whether to show UI feedback
            
        Returns:
            Tuple of (is_valid, converted_value, message)
        """
        self.check_stats['total_checks'] += 1
        
        # Record check
        check_record = {
            'source': source,
            'input_type': type(value).__name__,
            'expected_type': expected_type.__name__,
            'timestamp': time.time()
        }
        
        try:
            # Check if already correct type
            if isinstance(value, expected_type):
                check_record['result'] = 'correct_type'
                self.type_history.append(check_record)
                
                if show_feedback:
                    st.success(f"✅ **Type Check Passed ({source}):** {type(value).__name__} as expected")
                
                return True, value, f"Type check passed: {type(value).__name__}"
            
            # Type mismatch detected
            self.check_stats['type_mismatches'] += 1
            
            if show_feedback:
                st.warning(f"⚠️ **Type Mismatch ({source}):** Expected {expected_type.__name__}, got {type(value).__name__}")
            
            # Attempt automatic conversion
            if self.auto_convert:
                converted_value, conversion_message = self._auto_convert(
                    value, expected_type, source, show_feedback
                )
                
                if converted_value is not None:
                    check_record['result'] = 'auto_converted'
                    check_record['conversion_message'] = conversion_message
                    self.type_history.append(check_record)
                    return True, converted_value, conversion_message
            
            # Emergency conversion if enabled
            if self.emergency_mode:
                emergency_value, emergency_message = self._emergency_convert(
                    value, expected_type, source, show_feedback
                )
                
                check_record['result'] = 'emergency_converted'
                check_record['emergency_message'] = emergency_message
                self.type_history.append(check_record)
                return True, emergency_value, emergency_message
            
            # No conversion possible
            check_record['result'] = 'failed'
            self.type_history.append(check_record)
            self.check_stats['failures'] += 1
            
            error_message = f"Type mismatch: expected {expected_type.__name__}, got {type(value).__name__}"
            
            if show_feedback:
                st.error(f"❌ **Type Check Failed ({source}):** {error_message}")
            
            return False, value, error_message
            
        except Exception as e:
            self.check_stats['failures'] += 1
            error_message = f"Type check error for {source}: {str(e)}"
            
            check_record['result'] = 'error'
            check_record['error'] = str(e)
            self.type_history.append(check_record)
            
            if show_feedback:
                st.error(f"❌ **Type Check Error ({source}):** {str(e)}")
            
            logger.error(f"Type check error: {error_message}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Emergency fallback
            if self.emergency_mode:
                emergency_value = self._ultimate_fallback(value, expected_type)
                return True, emergency_value, f"Emergency fallback applied: {error_message}"
            
            return False, value, error_message
    
    def _auto_convert(
        self, 
        value: Any, 
        target_type: type, 
        source: str, 
        show_feedback: bool
    ) -> Tuple[Optional[Any], str]:
        """Attempt automatic type conversion."""
        
        self.check_stats['auto_conversions'] += 1
        
        try:
            # String conversion (most common case)
            if target_type == str:
                return self._convert_to_string(value, source, show_feedback)
            
            # List conversion
            elif target_type == list:
                return self._convert_to_list(value, source, show_feedback)
            
            # Dict conversion
            elif target_type == dict:
                return self._convert_to_dict(value, source, show_feedback)
            
            # Numeric conversions
            elif target_type in (int, float):
                return self._convert_to_numeric(value, target_type, source, show_feedback)
            
            # Boolean conversion
            elif target_type == bool:
                return self._convert_to_boolean(value, source, show_feedback)
            
            # Generic conversion attempt
            else:
                converted = target_type(value)
                message = f"Generic conversion to {target_type.__name__} successful"
                
                if show_feedback:
                    st.success(f"✅ **Auto Conversion ({source}):** {message}")
                
                return converted, message
                
        except Exception as e:
            error_message = f"Auto conversion failed: {str(e)}"
            
            if show_feedback:
                st.warning(f"⚠️ **Auto Conversion Failed ({source}):** {error_message}")
            
            return None, error_message
    
    def _convert_to_string(self, value: Any, source: str, show_feedback: bool) -> Tuple[str, str]:
        """Convert any value to string."""
        
        if isinstance(value, str):
            return value, "Already a string"
        
        elif isinstance(value, (list, tuple)):
            # Join list/tuple elements
            string_items = []
            for item in value:
                if item is not None:
                    item_str = str(item).strip()
                    if item_str:
                        string_items.append(item_str)
            
            result = "\n\n".join(string_items)
            message = f"Converted {type(value).__name__} with {len(value)} items to string"
            
            if show_feedback:
                st.success(f"✅ **List→String Conversion ({source}):** {message}")
            
            return result, message
        
        elif isinstance(value, dict):
            # Extract text from dictionary
            text_fields = ['text', 'content', 'body', 'message', 'data']
            
            for field in text_fields:
                if field in value and value[field]:
                    result = str(value[field])
                    message = f"Extracted text from dict field '{field}'"
                    
                    if show_feedback:
                        st.success(f"✅ **Dict→String Conversion ({source}):** {message}")
                    
                    return result, message
            
            # Fallback: convert entire dict
            result = str(value)
            message = "Converted entire dictionary to string"
            
            if show_feedback:
                st.warning(f"⚠️ **Dict→String Fallback ({source}):** {message}")
            
            return result, message
        
        elif isinstance(value, bytes):
            # Decode bytes
            try:
                result = value.decode('utf-8')
                message = "Decoded bytes to string using UTF-8"
            except UnicodeDecodeError:
                result = value.decode('utf-8', errors='replace')
                message = "Decoded bytes to string with error replacement"
            
            if show_feedback:
                st.success(f"✅ **Bytes→String Conversion ({source}):** {message}")
            
            return result, message
        
        else:
            # Generic string conversion
            result = str(value)
            message = f"Converted {type(value).__name__} to string"
            
            if show_feedback:
                st.success(f"✅ **Generic→String Conversion ({source}):** {message}")
            
            return result, message
    
    def _convert_to_list(self, value: Any, source: str, show_feedback: bool) -> Tuple[List, str]:
        """Convert any value to list."""
        
        if isinstance(value, list):
            return value, "Already a list"
        
        elif isinstance(value, (tuple, set)):
            result = list(value)
            message = f"Converted {type(value).__name__} to list"
            
            if show_feedback:
                st.success(f"✅ **{type(value).__name__}→List Conversion ({source}):** {message}")
            
            return result, message
        
        elif isinstance(value, str):
            # Split string into list
            if '\n' in value:
                result = [line.strip() for line in value.split('\n') if line.strip()]
                message = "Split string by newlines into list"
            else:
                result = [value]
                message = "Wrapped string in list"
            
            if show_feedback:
                st.success(f"✅ **String→List Conversion ({source}):** {message}")
            
            return result, message
        
        else:
            # Wrap single value in list
            result = [value]
            message = f"Wrapped {type(value).__name__} in list"
            
            if show_feedback:
                st.success(f"✅ **{type(value).__name__}→List Conversion ({source}):** {message}")
            
            return result, message
    
    def _convert_to_dict(self, value: Any, source: str, show_feedback: bool) -> Tuple[Dict, str]:
        """Convert any value to dictionary."""
        
        if isinstance(value, dict):
            return value, "Already a dictionary"
        
        elif isinstance(value, str):
            # Try to parse as JSON
            try:
                import json
                result = json.loads(value)
                message = "Parsed string as JSON dictionary"
                
                if show_feedback:
                    st.success(f"✅ **String→Dict Conversion ({source}):** {message}")
                
                return result, message
            except:
                # Create dict with text content
                result = {'text': value}
                message = "Wrapped string in dictionary with 'text' key"
                
                if show_feedback:
                    st.success(f"✅ **String→Dict Conversion ({source}):** {message}")
                
                return result, message
        
        elif isinstance(value, (list, tuple)):
            # Convert list to dict with indices
            result = {str(i): item for i, item in enumerate(value)}
            message = f"Converted {type(value).__name__} to dictionary with index keys"
            
            if show_feedback:
                st.success(f"✅ **{type(value).__name__}→Dict Conversion ({source}):** {message}")
            
            return result, message
        
        else:
            # Wrap value in dict
            result = {'value': value}
            message = f"Wrapped {type(value).__name__} in dictionary with 'value' key"
            
            if show_feedback:
                st.success(f"✅ **{type(value).__name__}→Dict Conversion ({source}):** {message}")
            
            return result, message
    
    def _convert_to_numeric(self, value: Any, target_type: type, source: str, show_feedback: bool) -> Tuple[Union[int, float], str]:
        """Convert value to numeric type."""
        
        if isinstance(value, target_type):
            return value, f"Already a {target_type.__name__}"
        
        elif isinstance(value, (int, float)):
            result = target_type(value)
            message = f"Converted {type(value).__name__} to {target_type.__name__}"
            
            if show_feedback:
                st.success(f"✅ **Numeric Conversion ({source}):** {message}")
            
            return result, message
        
        elif isinstance(value, str):
            # Try to parse string as number
            try:
                result = target_type(value.strip())
                message = f"Parsed string as {target_type.__name__}"
                
                if show_feedback:
                    st.success(f"✅ **String→{target_type.__name__} Conversion ({source}):** {message}")
                
                return result, message
            except ValueError:
                raise ValueError(f"Cannot convert string '{value}' to {target_type.__name__}")
        
        else:
            # Try generic conversion
            result = target_type(value)
            message = f"Converted {type(value).__name__} to {target_type.__name__}"
            
            if show_feedback:
                st.success(f"✅ **{type(value).__name__}→{target_type.__name__} Conversion ({source}):** {message}")
            
            return result, message
    
    def _convert_to_boolean(self, value: Any, source: str, show_feedback: bool) -> Tuple[bool, str]:
        """Convert value to boolean."""
        
        if isinstance(value, bool):
            return value, "Already a boolean"
        
        elif isinstance(value, str):
            # String to boolean conversion
            lower_value = value.lower().strip()
            if lower_value in ('true', 'yes', '1', 'on', 'enabled'):
                result = True
            elif lower_value in ('false', 'no', '0', 'off', 'disabled'):
                result = False
            else:
                result = bool(value)  # Non-empty string is True
            
            message = f"Converted string '{value}' to boolean {result}"
            
            if show_feedback:
                st.success(f"✅ **String→Boolean Conversion ({source}):** {message}")
            
            return result, message
        
        else:
            # Generic boolean conversion
            result = bool(value)
            message = f"Converted {type(value).__name__} to boolean {result}"
            
            if show_feedback:
                st.success(f"✅ **{type(value).__name__}→Boolean Conversion ({source}):** {message}")
            
            return result, message
    
    def _emergency_convert(
        self, 
        value: Any, 
        target_type: type, 
        source: str, 
        show_feedback: bool
    ) -> Tuple[Any, str]:
        """Emergency conversion that never fails."""
        
        self.check_stats['emergency_conversions'] += 1
        
        try:
            if target_type == str:
                result = self._ultimate_string_conversion(value)
                message = f"Emergency string conversion applied"
            elif target_type == list:
                result = [value] if not isinstance(value, (list, tuple)) else list(value)
                message = f"Emergency list conversion applied"
            elif target_type == dict:
                result = {'emergency_value': value}
                message = f"Emergency dict conversion applied"
            else:
                result = self._ultimate_fallback(value, target_type)
                message = f"Emergency {target_type.__name__} conversion applied"
            
            if show_feedback:
                st.warning(f"🚨 **Emergency Conversion ({source}):** {message}")
            
            return result, message
            
        except Exception as e:
            # Ultimate fallback
            result = self._ultimate_fallback(value, target_type)
            message = f"Ultimate fallback applied after emergency conversion failed: {str(e)}"
            
            if show_feedback:
                st.error(f"🚨 **Ultimate Fallback ({source}):** {message}")
            
            return result, message
    
    def _ultimate_string_conversion(self, value: Any) -> str:
        """Ultimate string conversion that never fails."""
        try:
            if value is None:
                return ""
            elif isinstance(value, str):
                return value
            elif isinstance(value, (list, tuple)):
                return "\n\n".join(str(item) for item in value if item is not None)
            elif isinstance(value, dict):
                return str(value.get('text', str(value)))
            elif isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')
            else:
                return str(value)
        except:
            return "<conversion_failed>"
    
    def _ultimate_fallback(self, value: Any, target_type: type) -> Any:
        """Ultimate fallback for any type conversion."""
        try:
            if target_type == str:
                return self._ultimate_string_conversion(value)
            elif target_type == list:
                return [value]
            elif target_type == dict:
                return {'value': value}
            elif target_type == bool:
                return bool(value)
            elif target_type in (int, float):
                return target_type(0)  # Safe default
            else:
                return target_type()  # Default constructor
        except:
            # Last resort defaults
            if target_type == str:
                return ""
            elif target_type == list:
                return []
            elif target_type == dict:
                return {}
            elif target_type == bool:
                return False
            elif target_type in (int, float):
                return 0
            else:
                return None
    
    def get_check_stats(self) -> Dict[str, int]:
        """Get type checking statistics."""
        return self.check_stats.copy()
    
    def get_type_history(self, limit: int = 10) -> List[Dict]:
        """Get recent type checking history."""
        return self.type_history[-limit:] if self.type_history else []
    
    def reset_stats(self):
        """Reset checking statistics and history."""
        self.check_stats = {
            'total_checks': 0,
            'type_mismatches': 0,
            'auto_conversions': 0,
            'emergency_conversions': 0,
            'failures': 0
        }
        self.type_history = []
    
    def display_stats(self):
        """Display type checking statistics in Streamlit UI."""
        stats = self.get_check_stats()
        
        st.subheader("🔍 Early Type Checking Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Checks", stats['total_checks'])
        with col2:
            st.metric("Type Mismatches", stats['type_mismatches'])
        with col3:
            st.metric("Auto Conversions", stats['auto_conversions'])
        with col4:
            st.metric("Emergency Conversions", stats['emergency_conversions'])
        with col5:
            st.metric("Failures", stats['failures'])
        
        # Success rate
        if stats['total_checks'] > 0:
            success_rate = ((stats['total_checks'] - stats['failures']) / stats['total_checks']) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")


# Global early type checker instance
_global_type_checker = EarlyTypeChecker(auto_convert=True, emergency_mode=True)


def check_text_type(
    value: Any, 
    source: str = "unknown", 
    show_feedback: bool = True
) -> Tuple[bool, str, str]:
    """
    Global function for early text type checking.
    
    Args:
        value: Input value to check
        source: Source description for logging
        show_feedback: Whether to show UI feedback
        
    Returns:
        Tuple of (is_valid, converted_value, message)
    """
    is_valid, converted_value, message = _global_type_checker.check_text_input(
        value, source, str, show_feedback
    )
    return is_valid, converted_value, message


def emergency_text_conversion(value: Any, source: str = "emergency") -> str:
    """
    Emergency text conversion that never fails.
    
    Args:
        value: Input value of any type
        source: Source description for logging
        
    Returns:
        String representation of the input
    """
    return _global_type_checker._ultimate_string_conversion(value)


def type_safe_wrapper(expected_type: type = str, emergency_mode: bool = True):
    """
    Decorator for type-safe function calls with automatic conversion.
    
    Args:
        expected_type: Expected type for the first argument
        emergency_mode: Whether to use emergency conversion on failure
        
    Returns:
        Decorated function with type safety
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if args:
                # Check first argument type
                first_arg = args[0]
                is_valid, converted_value, message = _global_type_checker.check_text_input(
                    first_arg, 
                    f"{func.__name__}_arg1", 
                    expected_type, 
                    show_feedback=False
                )
                
                if is_valid and converted_value != first_arg:
                    # Replace first argument with converted value
                    args = (converted_value,) + args[1:]
                elif not is_valid and emergency_mode:
                    # Emergency conversion
                    emergency_value = _global_type_checker._ultimate_fallback(first_arg, expected_type)
                    args = (emergency_value,) + args[1:]
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def get_type_checker_stats() -> Dict[str, int]:
    """Get global type checker statistics."""
    return _global_type_checker.get_check_stats()


def display_type_checker_stats():
    """Display global type checker statistics."""
    _global_type_checker.display_stats()


def reset_type_checker_stats():
    """Reset global type checker statistics."""
    _global_type_checker.reset_stats()


# Convenience functions for common type checks
def ensure_string(value: Any, source: str = "unknown") -> str:
    """Ensure value is a string, convert if necessary."""
    _, converted_value, _ = check_text_type(value, source, show_feedback=False)
    return converted_value


def ensure_list(value: Any, source: str = "unknown") -> List:
    """Ensure value is a list, convert if necessary."""
    is_valid, converted_value, _ = _global_type_checker.check_text_input(
        value, source, list, show_feedback=False
    )
    return converted_value


def ensure_dict(value: Any, source: str = "unknown") -> Dict:
    """Ensure value is a dictionary, convert if necessary."""
    is_valid, converted_value, _ = _global_type_checker.check_text_input(
        value, source, dict, show_feedback=False
    )
    return converted_value

