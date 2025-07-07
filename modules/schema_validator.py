"""
Export Schema Validator
======================

Auto-check for missing keys, special characters, newline issues
to save pain later during training.

Features:
- Comprehensive schema validation
- Format-specific validation rules
- Character encoding checks
- JSONL formatting validation
- Missing field detection
- Data type validation
- Custom validation rules
"""

import streamlit as st
import json
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
import unicodedata
from modules.logger import get_logger, log_event, log_user_action
import pandas as pd

class SchemaValidator:
    """
    Comprehensive schema validation system for export data
    
    Features:
    - Format-specific validation (OpenAI, Hugging Face, etc.)
    - Character encoding and special character detection
    - JSONL formatting validation
    - Missing field detection
    - Data type validation
    - Custom validation rules
    - Detailed error reporting
    """
    
    def __init__(self):
        self.logger = get_logger("schema_validator")
        
        # Initialize session state
        if 'validation_results' not in st.session_state:
            st.session_state['validation_results'] = {}
        
        if 'validation_settings' not in st.session_state:
            st.session_state['validation_settings'] = {
                'strict_mode': True,
                'check_encoding': True,
                'check_special_chars': True,
                'check_newlines': True,
                'max_field_length': 10000,
                'min_field_length': 1
            }
        
        # Validation schemas for different formats
        self.validation_schemas = {
            'openai_chat': {
                'required_fields': ['messages'],
                'field_types': {
                    'messages': list
                },
                'message_schema': {
                    'required_fields': ['role', 'content'],
                    'field_types': {
                        'role': str,
                        'content': str
                    },
                    'valid_roles': ['system', 'user', 'assistant']
                },
                'constraints': {
                    'max_messages': 100,
                    'max_content_length': 8000,
                    'min_content_length': 1
                }
            },
            'openai_completion': {
                'required_fields': ['prompt', 'completion'],
                'field_types': {
                    'prompt': str,
                    'completion': str
                },
                'constraints': {
                    'max_prompt_length': 4000,
                    'max_completion_length': 4000,
                    'min_prompt_length': 1,
                    'min_completion_length': 1
                }
            },
            'huggingface_instruction': {
                'required_fields': ['instruction', 'output'],
                'optional_fields': ['input'],
                'field_types': {
                    'instruction': str,
                    'input': str,
                    'output': str
                },
                'constraints': {
                    'max_instruction_length': 2000,
                    'max_input_length': 2000,
                    'max_output_length': 4000,
                    'min_instruction_length': 5,
                    'min_output_length': 1
                }
            },
            'huggingface_qa': {
                'required_fields': ['question', 'answer'],
                'optional_fields': ['context'],
                'field_types': {
                    'question': str,
                    'answer': str,
                    'context': str
                },
                'constraints': {
                    'max_question_length': 1000,
                    'max_answer_length': 2000,
                    'max_context_length': 4000,
                    'min_question_length': 5,
                    'min_answer_length': 1
                }
            },
            'alpaca': {
                'required_fields': ['instruction', 'input', 'output'],
                'field_types': {
                    'instruction': str,
                    'input': str,
                    'output': str
                },
                'constraints': {
                    'max_instruction_length': 2000,
                    'max_input_length': 2000,
                    'max_output_length': 4000,
                    'min_instruction_length': 5,
                    'min_output_length': 1
                }
            }
        }
        
        # Problematic characters and patterns
        self.problematic_patterns = {
            'control_chars': r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',
            'zero_width_chars': r'[\u200B-\u200D\uFEFF]',
            'bidi_chars': r'[\u202A-\u202E\u2066-\u2069]',
            'private_use': r'[\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]',
            'unassigned': r'[\uFDD0-\uFDEF\uFFFE\uFFFF]',
            'emoji_modifiers': r'[\U0001F3FB-\U0001F3FF]',
            'variation_selectors': r'[\uFE00-\uFE0F\U000E0100-\U000E01EF]'
        }
    
    def render_validation_interface(self, content_data: List[Dict], target_format: str = 'openai_chat'):
        """Render the complete schema validation interface"""
        
        st.subheader("🔍 Export Schema Validation")
        
        if not content_data:
            st.warning("⚠️ No content available for validation")
            return
        
        # Validation controls
        self.render_validation_controls(target_format)
        
        # Run validation
        validation_results = self.validate_content(content_data, target_format)
        
        # Display results
        self.render_validation_results(validation_results)
        
        # Detailed error analysis
        if validation_results['errors']:
            self.render_error_analysis(validation_results)
        
        # Validation report export
        self.render_validation_export_options(validation_results)
    
    def render_validation_controls(self, target_format: str):
        """Render validation control panel"""
        
        st.markdown("**🔧 Validation Settings:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strict_mode = st.checkbox(
                "Strict Mode",
                value=st.session_state['validation_settings']['strict_mode'],
                help="Enable strict validation with all checks"
            )
            st.session_state['validation_settings']['strict_mode'] = strict_mode
            
            check_encoding = st.checkbox(
                "Check Encoding",
                value=st.session_state['validation_settings']['check_encoding'],
                help="Validate character encoding and detect problematic characters"
            )
            st.session_state['validation_settings']['check_encoding'] = check_encoding
        
        with col2:
            check_special_chars = st.checkbox(
                "Check Special Characters",
                value=st.session_state['validation_settings']['check_special_chars'],
                help="Detect control characters, zero-width characters, etc."
            )
            st.session_state['validation_settings']['check_special_chars'] = check_special_chars
            
            check_newlines = st.checkbox(
                "Check Newlines",
                value=st.session_state['validation_settings']['check_newlines'],
                help="Detect problematic newline characters in JSONL format"
            )
            st.session_state['validation_settings']['check_newlines'] = check_newlines
        
        with col3:
            max_length = st.number_input(
                "Max Field Length",
                min_value=100,
                max_value=50000,
                value=st.session_state['validation_settings']['max_field_length'],
                help="Maximum allowed length for text fields"
            )
            st.session_state['validation_settings']['max_field_length'] = max_length
            
            min_length = st.number_input(
                "Min Field Length",
                min_value=0,
                max_value=100,
                value=st.session_state['validation_settings']['min_field_length'],
                help="Minimum required length for text fields"
            )
            st.session_state['validation_settings']['min_field_length'] = min_length
        
        # Quick validation presets
        st.markdown("**⚡ Quick Presets:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔒 Strict"):
                self.apply_validation_preset('strict')
                st.rerun()
        
        with col2:
            if st.button("⚖️ Balanced"):
                self.apply_validation_preset('balanced')
                st.rerun()
        
        with col3:
            if st.button("🚀 Fast"):
                self.apply_validation_preset('fast')
                st.rerun()
        
        with col4:
            if st.button("🎯 Custom"):
                self.apply_validation_preset('custom')
                st.rerun()
    
    def apply_validation_preset(self, preset_type: str):
        """Apply predefined validation settings"""
        
        if preset_type == 'strict':
            st.session_state['validation_settings'].update({
                'strict_mode': True,
                'check_encoding': True,
                'check_special_chars': True,
                'check_newlines': True,
                'max_field_length': 8000,
                'min_field_length': 1
            })
        
        elif preset_type == 'balanced':
            st.session_state['validation_settings'].update({
                'strict_mode': True,
                'check_encoding': True,
                'check_special_chars': False,
                'check_newlines': True,
                'max_field_length': 10000,
                'min_field_length': 1
            })
        
        elif preset_type == 'fast':
            st.session_state['validation_settings'].update({
                'strict_mode': False,
                'check_encoding': False,
                'check_special_chars': False,
                'check_newlines': True,
                'max_field_length': 15000,
                'min_field_length': 0
            })
        
        log_user_action("validation_preset_applied", {"preset": preset_type})
    
    def validate_content(self, content_data: List[Dict], target_format: str) -> Dict[str, Any]:
        """Perform comprehensive validation of content data"""
        
        validation_start = datetime.now()
        
        results = {
            'total_items': len(content_data),
            'valid_items': 0,
            'invalid_items': 0,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'validation_time': 0,
            'target_format': target_format
        }
        
        schema = self.validation_schemas.get(target_format, {})
        
        for i, item in enumerate(content_data):
            item_errors = []
            item_warnings = []
            
            try:
                # Schema validation
                schema_errors = self.validate_item_schema(item, schema, i)
                item_errors.extend(schema_errors)
                
                # Character encoding validation
                if st.session_state['validation_settings']['check_encoding']:
                    encoding_errors = self.validate_item_encoding(item, i)
                    item_errors.extend(encoding_errors)
                
                # Special character validation
                if st.session_state['validation_settings']['check_special_chars']:
                    special_char_errors = self.validate_special_characters(item, i)
                    item_errors.extend(special_char_errors)
                
                # Newline validation
                if st.session_state['validation_settings']['check_newlines']:
                    newline_errors = self.validate_newlines(item, i)
                    item_errors.extend(newline_errors)
                
                # JSONL serialization test
                jsonl_errors = self.validate_jsonl_serialization(item, i)
                item_errors.extend(jsonl_errors)
                
                # Length validation
                length_errors = self.validate_field_lengths(item, i)
                item_errors.extend(length_errors)
                
                # Data type validation
                type_errors = self.validate_data_types(item, schema, i)
                item_errors.extend(type_errors)
                
                if item_errors:
                    results['invalid_items'] += 1
                    results['errors'].extend(item_errors)
                else:
                    results['valid_items'] += 1
                
                if item_warnings:
                    results['warnings'].extend(item_warnings)
            
            except Exception as e:
                results['invalid_items'] += 1
                results['errors'].append({
                    'item_index': i,
                    'error_type': 'validation_exception',
                    'message': f"Validation failed: {str(e)}",
                    'severity': 'critical'
                })
        
        # Calculate statistics
        results['statistics'] = self.calculate_validation_statistics(results)
        results['validation_time'] = (datetime.now() - validation_start).total_seconds()
        
        # Store results in session state
        st.session_state['validation_results'] = results
        
        # Log validation
        log_event("schema_validation_completed", {
            "target_format": target_format,
            "total_items": results['total_items'],
            "valid_items": results['valid_items'],
            "error_count": len(results['errors']),
            "validation_time": results['validation_time']
        }, "schema_validator")
        
        return results
    
    def validate_item_schema(self, item: Dict, schema: Dict, item_index: int) -> List[Dict]:
        """Validate item against schema requirements"""
        
        errors = []
        
        if not schema:
            return errors
        
        # Check required fields
        required_fields = schema.get('required_fields', [])
        for field in required_fields:
            if field not in item:
                errors.append({
                    'item_index': item_index,
                    'error_type': 'missing_required_field',
                    'field': field,
                    'message': f"Missing required field: '{field}'",
                    'severity': 'critical'
                })
            elif item[field] is None:
                errors.append({
                    'item_index': item_index,
                    'error_type': 'null_required_field',
                    'field': field,
                    'message': f"Required field '{field}' is null",
                    'severity': 'critical'
                })
        
        # Check field types
        field_types = schema.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in item and item[field] is not None:
                if not isinstance(item[field], expected_type):
                    errors.append({
                        'item_index': item_index,
                        'error_type': 'invalid_field_type',
                        'field': field,
                        'expected_type': expected_type.__name__,
                        'actual_type': type(item[field]).__name__,
                        'message': f"Field '{field}' should be {expected_type.__name__}, got {type(item[field]).__name__}",
                        'severity': 'critical'
                    })
        
        # Format-specific validation
        if schema.get('message_schema') and 'messages' in item:
            # OpenAI chat format validation
            messages = item['messages']
            if isinstance(messages, list):
                for msg_idx, message in enumerate(messages):
                    msg_errors = self.validate_message_schema(message, schema['message_schema'], item_index, msg_idx)
                    errors.extend(msg_errors)
        
        return errors
    
    def validate_message_schema(self, message: Dict, message_schema: Dict, item_index: int, msg_index: int) -> List[Dict]:
        """Validate individual message in chat format"""
        
        errors = []
        
        # Check required fields
        required_fields = message_schema.get('required_fields', [])
        for field in required_fields:
            if field not in message:
                errors.append({
                    'item_index': item_index,
                    'message_index': msg_index,
                    'error_type': 'missing_message_field',
                    'field': field,
                    'message': f"Message {msg_index} missing required field: '{field}'",
                    'severity': 'critical'
                })
        
        # Check valid roles
        if 'role' in message:
            valid_roles = message_schema.get('valid_roles', [])
            if valid_roles and message['role'] not in valid_roles:
                errors.append({
                    'item_index': item_index,
                    'message_index': msg_index,
                    'error_type': 'invalid_role',
                    'field': 'role',
                    'value': message['role'],
                    'valid_values': valid_roles,
                    'message': f"Message {msg_index} has invalid role: '{message['role']}'. Valid roles: {valid_roles}",
                    'severity': 'critical'
                })
        
        return errors
    
    def validate_item_encoding(self, item: Dict, item_index: int) -> List[Dict]:
        """Validate character encoding in item"""
        
        errors = []
        
        for field, value in item.items():
            if isinstance(value, str):
                try:
                    # Test UTF-8 encoding
                    value.encode('utf-8')
                    
                    # Check for problematic Unicode categories
                    for char in value:
                        category = unicodedata.category(char)
                        if category in ['Cc', 'Cf', 'Co', 'Cs', 'Cn']:  # Control, Format, Private Use, Surrogate, Unassigned
                            errors.append({
                                'item_index': item_index,
                                'error_type': 'problematic_unicode',
                                'field': field,
                                'character': repr(char),
                                'category': category,
                                'message': f"Field '{field}' contains problematic Unicode character: {repr(char)} (category: {category})",
                                'severity': 'warning'
                            })
                
                except UnicodeEncodeError as e:
                    errors.append({
                        'item_index': item_index,
                        'error_type': 'encoding_error',
                        'field': field,
                        'message': f"Field '{field}' contains characters that cannot be encoded in UTF-8: {str(e)}",
                        'severity': 'critical'
                    })
        
        return errors
    
    def validate_special_characters(self, item: Dict, item_index: int) -> List[Dict]:
        """Validate for problematic special characters"""
        
        errors = []
        
        for field, value in item.items():
            if isinstance(value, str):
                for pattern_name, pattern in self.problematic_patterns.items():
                    matches = re.findall(pattern, value)
                    if matches:
                        errors.append({
                            'item_index': item_index,
                            'error_type': 'special_characters',
                            'field': field,
                            'pattern_type': pattern_name,
                            'characters': [repr(char) for char in matches],
                            'message': f"Field '{field}' contains {pattern_name}: {[repr(char) for char in matches]}",
                            'severity': 'warning'
                        })
        
        return errors
    
    def validate_newlines(self, item: Dict, item_index: int) -> List[Dict]:
        """Validate newline characters that can break JSONL format"""
        
        errors = []
        
        for field, value in item.items():
            if isinstance(value, str):
                # Check for unescaped newlines
                if '\n' in value or '\r' in value:
                    errors.append({
                        'item_index': item_index,
                        'error_type': 'newline_characters',
                        'field': field,
                        'message': f"Field '{field}' contains newline characters that may break JSONL format",
                        'severity': 'warning'
                    })
                
                # Check for other line separators
                line_separators = ['\u2028', '\u2029']  # Line Separator, Paragraph Separator
                for sep in line_separators:
                    if sep in value:
                        errors.append({
                            'item_index': item_index,
                            'error_type': 'line_separator',
                            'field': field,
                            'character': repr(sep),
                            'message': f"Field '{field}' contains Unicode line separator: {repr(sep)}",
                            'severity': 'warning'
                        })
        
        return errors
    
    def validate_jsonl_serialization(self, item: Dict, item_index: int) -> List[Dict]:
        """Test if item can be properly serialized to JSONL"""
        
        errors = []
        
        try:
            # Test JSON serialization
            json_str = json.dumps(item, ensure_ascii=False)
            
            # Test if it contains actual newlines (not escaped)
            if '\n' in json_str and '\\n' not in json_str:
                errors.append({
                    'item_index': item_index,
                    'error_type': 'jsonl_serialization',
                    'message': "Item contains unescaped newlines that will break JSONL format",
                    'severity': 'critical'
                })
            
            # Test deserialization
            json.loads(json_str)
        
        except (TypeError, ValueError) as e:
            errors.append({
                'item_index': item_index,
                'error_type': 'json_serialization',
                'message': f"Item cannot be serialized to JSON: {str(e)}",
                'severity': 'critical'
            })
        
        return errors
    
    def validate_field_lengths(self, item: Dict, item_index: int) -> List[Dict]:
        """Validate field lengths against constraints"""
        
        errors = []
        settings = st.session_state['validation_settings']
        
        for field, value in item.items():
            if isinstance(value, str):
                length = len(value)
                
                # Check maximum length
                if length > settings['max_field_length']:
                    errors.append({
                        'item_index': item_index,
                        'error_type': 'field_too_long',
                        'field': field,
                        'length': length,
                        'max_length': settings['max_field_length'],
                        'message': f"Field '{field}' is too long: {length} characters (max: {settings['max_field_length']})",
                        'severity': 'warning'
                    })
                
                # Check minimum length
                if length < settings['min_field_length']:
                    errors.append({
                        'item_index': item_index,
                        'error_type': 'field_too_short',
                        'field': field,
                        'length': length,
                        'min_length': settings['min_field_length'],
                        'message': f"Field '{field}' is too short: {length} characters (min: {settings['min_field_length']})",
                        'severity': 'warning'
                    })
                
                # Check for empty strings
                if not value.strip():
                    errors.append({
                        'item_index': item_index,
                        'error_type': 'empty_field',
                        'field': field,
                        'message': f"Field '{field}' is empty or contains only whitespace",
                        'severity': 'warning'
                    })
        
        return errors
    
    def validate_data_types(self, item: Dict, schema: Dict, item_index: int) -> List[Dict]:
        """Validate data types and structure"""
        
        errors = []
        
        for field, value in item.items():
            # Check for nested objects that might cause issues
            if isinstance(value, dict):
                errors.append({
                    'item_index': item_index,
                    'error_type': 'nested_object',
                    'field': field,
                    'message': f"Field '{field}' contains nested object which may not be supported by all training formats",
                    'severity': 'info'
                })
            
            elif isinstance(value, list):
                # Check if it's a valid list structure
                if field != 'messages':  # messages field is expected to be a list
                    errors.append({
                        'item_index': item_index,
                        'error_type': 'unexpected_list',
                        'field': field,
                        'message': f"Field '{field}' contains list which may not be supported by all training formats",
                        'severity': 'info'
                    })
        
        return errors
    
    def calculate_validation_statistics(self, results: Dict) -> Dict[str, Any]:
        """Calculate comprehensive validation statistics"""
        
        stats = {
            'success_rate': (results['valid_items'] / results['total_items'] * 100) if results['total_items'] > 0 else 0,
            'error_types': {},
            'field_errors': {},
            'severity_counts': {'critical': 0, 'warning': 0, 'info': 0}
        }
        
        # Count error types
        for error in results['errors']:
            error_type = error['error_type']
            stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
            
            # Count field errors
            if 'field' in error:
                field = error['field']
                stats['field_errors'][field] = stats['field_errors'].get(field, 0) + 1
            
            # Count severity
            severity = error.get('severity', 'warning')
            stats['severity_counts'][severity] += 1
        
        return stats
    
    def render_validation_results(self, results: Dict):
        """Render validation results overview"""
        
        st.markdown("---")
        st.markdown("**📊 Validation Results:**")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Items", results['total_items'])
        
        with col2:
            st.metric("Valid Items", results['valid_items'])
        
        with col3:
            st.metric("Invalid Items", results['invalid_items'])
        
        with col4:
            success_rate = results['statistics']['success_rate']
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Success rate visualization
        if results['total_items'] > 0:
            if success_rate >= 95:
                st.success(f"🎉 Excellent! {success_rate:.1f}% of items passed validation")
            elif success_rate >= 80:
                st.warning(f"⚠️ Good! {success_rate:.1f}% of items passed validation")
            else:
                st.error(f"❌ Issues found! Only {success_rate:.1f}% of items passed validation")
        
        # Error summary
        if results['errors']:
            st.markdown("**🚨 Error Summary:**")
            
            error_stats = results['statistics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**By Severity:**")
                for severity, count in error_stats['severity_counts'].items():
                    if count > 0:
                        severity_emoji = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}
                        st.write(f"{severity_emoji.get(severity, '⚪')} {severity.title()}: {count}")
            
            with col2:
                st.markdown("**Top Error Types:**")
                sorted_errors = sorted(error_stats['error_types'].items(), key=lambda x: x[1], reverse=True)
                for error_type, count in sorted_errors[:5]:
                    st.write(f"• {error_type.replace('_', ' ').title()}: {count}")
        
        # Performance info
        st.caption(f"Validation completed in {results['validation_time']:.2f} seconds")
    
    def render_error_analysis(self, results: Dict):
        """Render detailed error analysis"""
        
        st.markdown("---")
        st.markdown("**🔍 Detailed Error Analysis:**")
        
        # Error filtering
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.selectbox(
                "Filter by Severity",
                ["All", "Critical", "Warning", "Info"],
                help="Filter errors by severity level"
            )
        
        with col2:
            error_type_filter = st.selectbox(
                "Filter by Type",
                ["All"] + list(results['statistics']['error_types'].keys()),
                help="Filter errors by type"
            )
        
        with col3:
            max_errors = st.selectbox(
                "Show Errors",
                [10, 25, 50, 100, "All"],
                help="Maximum number of errors to display"
            )
        
        # Filter errors
        filtered_errors = results['errors']
        
        if severity_filter != "All":
            filtered_errors = [e for e in filtered_errors if e.get('severity', 'warning').lower() == severity_filter.lower()]
        
        if error_type_filter != "All":
            filtered_errors = [e for e in filtered_errors if e['error_type'] == error_type_filter]
        
        if max_errors != "All":
            filtered_errors = filtered_errors[:max_errors]
        
        # Display errors
        if filtered_errors:
            for i, error in enumerate(filtered_errors):
                severity = error.get('severity', 'warning')
                severity_color = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}
                
                with st.expander(
                    f"{severity_color.get(severity, '⚪')} Item {error['item_index'] + 1}: {error['error_type'].replace('_', ' ').title()}",
                    expanded=i < 3
                ):
                    st.write(f"**Message:** {error['message']}")
                    st.write(f"**Severity:** {severity.title()}")
                    
                    if 'field' in error:
                        st.write(f"**Field:** {error['field']}")
                    
                    if 'expected_type' in error:
                        st.write(f"**Expected Type:** {error['expected_type']}")
                        st.write(f"**Actual Type:** {error['actual_type']}")
                    
                    if 'characters' in error:
                        st.write(f"**Problematic Characters:** {', '.join(error['characters'])}")
                    
                    # Show fix suggestions
                    fix_suggestion = self.get_fix_suggestion(error)
                    if fix_suggestion:
                        st.info(f"💡 **Fix Suggestion:** {fix_suggestion}")
        else:
            st.info("No errors match the current filters")
    
    def get_fix_suggestion(self, error: Dict) -> Optional[str]:
        """Get fix suggestion for specific error types"""
        
        error_type = error['error_type']
        
        suggestions = {
            'missing_required_field': f"Add the required field '{error.get('field', '')}' to your data",
            'invalid_field_type': f"Convert field '{error.get('field', '')}' to {error.get('expected_type', 'correct')} type",
            'newline_characters': "Remove or escape newline characters in the field content",
            'field_too_long': f"Truncate field '{error.get('field', '')}' to under {error.get('max_length', 'limit')} characters",
            'field_too_short': f"Add more content to field '{error.get('field', '')}' (minimum {error.get('min_length', 1)} characters)",
            'empty_field': f"Add content to the empty field '{error.get('field', '')}'",
            'special_characters': "Remove or replace special characters that may cause encoding issues",
            'json_serialization': "Fix data structure to ensure proper JSON serialization",
            'invalid_role': f"Use valid role values: {error.get('valid_values', [])}",
            'encoding_error': "Fix character encoding issues or remove problematic characters"
        }
        
        return suggestions.get(error_type)
    
    def render_validation_export_options(self, results: Dict):
        """Render validation report export options"""
        
        st.markdown("---")
        st.markdown("**📥 Export Validation Report:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Full Report"):
                self.export_validation_report(results, 'full')
        
        with col2:
            if st.button("🚨 Export Errors Only"):
                self.export_validation_report(results, 'errors')
        
        with col3:
            if st.button("📈 Export Statistics"):
                self.export_validation_report(results, 'stats')
    
    def export_validation_report(self, results: Dict, report_type: str):
        """Export validation report in various formats"""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if report_type == 'full':
                # Full validation report
                report_data = {
                    'validation_summary': {
                        'timestamp': datetime.now().isoformat(),
                        'target_format': results['target_format'],
                        'total_items': results['total_items'],
                        'valid_items': results['valid_items'],
                        'invalid_items': results['invalid_items'],
                        'success_rate': results['statistics']['success_rate'],
                        'validation_time': results['validation_time']
                    },
                    'statistics': results['statistics'],
                    'errors': results['errors'],
                    'warnings': results['warnings'],
                    'validation_settings': st.session_state['validation_settings']
                }
                
                filename = f"validation_report_full_{timestamp}.json"
            
            elif report_type == 'errors':
                # Errors only
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'total_errors': len(results['errors']),
                    'errors': results['errors']
                }
                
                filename = f"validation_errors_{timestamp}.json"
            
            else:  # stats
                # Statistics only
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'validation_summary': {
                        'total_items': results['total_items'],
                        'valid_items': results['valid_items'],
                        'invalid_items': results['invalid_items'],
                        'success_rate': results['statistics']['success_rate']
                    },
                    'statistics': results['statistics']
                }
                
                filename = f"validation_stats_{timestamp}.json"
            
            # Create download
            report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                f"📥 Download {report_type.title()} Report",
                data=report_json,
                file_name=filename,
                mime="application/json"
            )
            
            log_event("validation_report_exported", {
                "report_type": report_type,
                "error_count": len(results['errors']),
                "success_rate": results['statistics']['success_rate']
            }, "schema_validator")
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            self.logger.error(f"Validation report export failed: {str(e)}")

# Integration function for main app
def render_schema_validation(content_data: List[Dict], target_format: str = 'openai_chat'):
    """
    Render schema validation in main app
    
    Usage:
    from modules.schema_validator import render_schema_validation
    
    render_schema_validation(enhanced_content, 'openai_chat')
    """
    
    validator = SchemaValidator()
    validator.render_validation_interface(content_data, target_format)

# Quick validation function
def quick_validate(content_data: List[Dict], target_format: str = 'openai_chat') -> Dict[str, Any]:
    """
    Quick validation without UI
    
    Usage:
    from modules.schema_validator import quick_validate
    
    results = quick_validate(content_data, 'openai_chat')
    """
    
    validator = SchemaValidator()
    return validator.validate_content(content_data, target_format)

if __name__ == "__main__":
    # Test the schema validator
    st.set_page_config(page_title="Schema Validator Test", layout="wide")
    
    st.title("Schema Validator Test")
    
    # Sample data with various issues
    sample_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Hello\nWorld"},  # Newline issue
                {"role": "assistant", "content": "Hi there!"}
            ]
        },
        {
            "prompt": "What is machine learning?",  # Wrong format
            "completion": "Machine learning is a subset of AI."
        }
    ]
    
    # Render schema validation
    render_schema_validation(sample_data, 'openai_chat')

