"""
Format Preview Engine
====================

Shows exactly how the final JSONL will look for OpenAI/Hugging Face
to boost user confidence and ensure proper formatting.

Features:
- Real-time JSONL preview
- OpenAI fine-tuning format validation
- Hugging Face dataset format preview
- Custom format templates
- Schema validation and error detection
- Export format optimization
"""

import streamlit as st
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from modules.logger import get_logger, log_event, log_user_action
import pandas as pd

class FormatPreviewEngine:
    """
    Comprehensive format preview engine for AI training data
    
    Features:
    - Multiple format previews (OpenAI, Hugging Face, Custom)
    - Real-time format validation
    - Schema compliance checking
    - Interactive format customization
    - Export optimization suggestions
    """
    
    def __init__(self):
        self.logger = get_logger("format_preview")
        
        # Initialize session state
        if 'preview_format' not in st.session_state:
            st.session_state['preview_format'] = 'openai_chat'
        
        if 'preview_sample_size' not in st.session_state:
            st.session_state['preview_sample_size'] = 3
        
        if 'custom_format_template' not in st.session_state:
            st.session_state['custom_format_template'] = {}
        
        # Format templates
        self.format_templates = {
            'openai_chat': {
                'name': 'OpenAI Chat Completion',
                'description': 'Format for OpenAI GPT fine-tuning with chat completions',
                'schema': {
                    'messages': [
                        {'role': 'system', 'content': 'string'},
                        {'role': 'user', 'content': 'string'},
                        {'role': 'assistant', 'content': 'string'}
                    ]
                },
                'example': {
                    'messages': [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': 'What is the capital of France?'},
                        {'role': 'assistant', 'content': 'The capital of France is Paris.'}
                    ]
                }
            },
            'openai_completion': {
                'name': 'OpenAI Text Completion',
                'description': 'Format for OpenAI GPT fine-tuning with text completions',
                'schema': {
                    'prompt': 'string',
                    'completion': 'string'
                },
                'example': {
                    'prompt': 'What is the capital of France?',
                    'completion': 'The capital of France is Paris.'
                }
            },
            'huggingface_instruction': {
                'name': 'Hugging Face Instruction',
                'description': 'Format for Hugging Face instruction-following datasets',
                'schema': {
                    'instruction': 'string',
                    'input': 'string (optional)',
                    'output': 'string'
                },
                'example': {
                    'instruction': 'Answer the following question.',
                    'input': 'What is the capital of France?',
                    'output': 'The capital of France is Paris.'
                }
            },
            'huggingface_qa': {
                'name': 'Hugging Face Q&A',
                'description': 'Format for Hugging Face question-answering datasets',
                'schema': {
                    'question': 'string',
                    'answer': 'string',
                    'context': 'string (optional)'
                },
                'example': {
                    'question': 'What is the capital of France?',
                    'answer': 'The capital of France is Paris.',
                    'context': 'France is a country in Western Europe.'
                }
            },
            'alpaca': {
                'name': 'Alpaca Format',
                'description': 'Stanford Alpaca instruction format',
                'schema': {
                    'instruction': 'string',
                    'input': 'string',
                    'output': 'string'
                },
                'example': {
                    'instruction': 'Answer the following question about geography.',
                    'input': 'What is the capital of France?',
                    'output': 'The capital of France is Paris.'
                }
            },
            'sharegpt': {
                'name': 'ShareGPT Format',
                'description': 'ShareGPT conversation format',
                'schema': {
                    'conversations': [
                        {'from': 'human', 'value': 'string'},
                        {'from': 'gpt', 'value': 'string'}
                    ]
                },
                'example': {
                    'conversations': [
                        {'from': 'human', 'value': 'What is the capital of France?'},
                        {'from': 'gpt', 'value': 'The capital of France is Paris.'}
                    ]
                }
            }
        }
    
    def render_format_preview_interface(self, content_data: List[Dict]):
        """Render the complete format preview interface"""
        
        st.subheader("📋 Fine-Tune Format Preview")
        
        if not content_data:
            st.warning("⚠️ No content available for preview")
            return
        
        # Format selection and controls
        self.render_format_controls()
        
        # Format information
        self.render_format_information()
        
        # Live preview
        self.render_live_preview(content_data)
        
        # Validation results
        self.render_validation_results(content_data)
        
        # Export options
        self.render_export_options(content_data)
    
    def render_format_controls(self):
        """Render format selection and control panel"""
        
        st.markdown("**Format Selection:**")
        
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            format_options = list(self.format_templates.keys())
            format_names = [self.format_templates[fmt]['name'] for fmt in format_options]
            
            selected_format_name = st.selectbox(
                "Target Format",
                format_names,
                index=format_options.index(st.session_state['preview_format']),
                help="Choose the target format for your training data"
            )
            
            # Update session state
            selected_format = format_options[format_names.index(selected_format_name)]
            st.session_state['preview_format'] = selected_format
        
        with col2:
            sample_size = st.selectbox(
                "Preview Items",
                [1, 3, 5, 10],
                index=1,
                help="Number of items to show in preview"
            )
            st.session_state['preview_sample_size'] = sample_size
        
        with col3:
            if st.button("🔄 Refresh Preview"):
                log_user_action("format_preview_refreshed", {
                    "format": st.session_state['preview_format']
                })
                st.rerun()
    
    def render_format_information(self):
        """Render information about the selected format"""
        
        selected_format = st.session_state['preview_format']
        format_info = self.format_templates[selected_format]
        
        st.markdown("---")
        st.markdown("**📖 Format Information:**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Format:** {format_info['name']}")
            st.markdown(f"**Description:** {format_info['description']}")
        
        with col2:
            st.markdown("**Schema:**")
            st.json(format_info['schema'])
        
        # Example
        with st.expander("📄 Format Example"):
            st.json(format_info['example'])
    
    def render_live_preview(self, content_data: List[Dict]):
        """Render live preview of formatted data"""
        
        st.markdown("---")
        st.markdown("**👀 Live Preview:**")
        
        selected_format = st.session_state['preview_format']
        sample_size = st.session_state['preview_sample_size']
        
        # Convert content to selected format
        formatted_data = self.convert_to_format(content_data[:sample_size], selected_format)
        
        if formatted_data:
            # Show formatted examples
            for i, item in enumerate(formatted_data):
                with st.expander(f"📄 Example {i + 1}", expanded=i == 0):
                    
                    # JSON preview
                    st.markdown("**JSON Format:**")
                    st.json(item)
                    
                    # JSONL preview
                    st.markdown("**JSONL Line:**")
                    jsonl_line = json.dumps(item, ensure_ascii=False)
                    st.code(jsonl_line, language='json')
                    
                    # Validation status
                    is_valid, validation_errors = self.validate_format(item, selected_format)
                    
                    if is_valid:
                        st.success("✅ Valid format")
                    else:
                        st.error("❌ Format errors:")
                        for error in validation_errors:
                            st.write(f"• {error}")
        else:
            st.warning("Could not convert content to selected format")
    
    def render_validation_results(self, content_data: List[Dict]):
        """Render comprehensive validation results"""
        
        st.markdown("---")
        st.markdown("**🔍 Validation Results:**")
        
        selected_format = st.session_state['preview_format']
        
        # Validate all content
        validation_results = self.validate_all_content(content_data, selected_format)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Items", validation_results['total_items'])
        
        with col2:
            st.metric("Valid Items", validation_results['valid_items'])
        
        with col3:
            st.metric("Invalid Items", validation_results['invalid_items'])
        
        with col4:
            if validation_results['total_items'] > 0:
                success_rate = (validation_results['valid_items'] / validation_results['total_items']) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Detailed validation results
        if validation_results['invalid_items'] > 0:
            with st.expander(f"⚠️ Validation Issues ({validation_results['invalid_items']} items)"):
                
                # Group errors by type
                error_summary = {}
                for error_info in validation_results['errors']:
                    error_type = error_info['error']
                    if error_type not in error_summary:
                        error_summary[error_type] = []
                    error_summary[error_type].append(error_info['item_index'])
                
                for error_type, item_indices in error_summary.items():
                    st.write(f"**{error_type}:** Items {', '.join(map(str, item_indices))}")
        
        # Format-specific recommendations
        self.render_format_recommendations(validation_results, selected_format)
    
    def render_format_recommendations(self, validation_results: Dict, format_type: str):
        """Render format-specific recommendations"""
        
        st.markdown("**💡 Recommendations:**")
        
        recommendations = []
        
        # General recommendations
        if validation_results['invalid_items'] > 0:
            recommendations.append("🔧 Fix validation errors before export")
        
        if validation_results['total_items'] < 100:
            recommendations.append("📊 Consider adding more training examples (recommended: 100+ for fine-tuning)")
        
        # Format-specific recommendations
        if format_type == 'openai_chat':
            recommendations.extend([
                "💬 Ensure conversations have clear system, user, and assistant roles",
                "📝 Keep messages concise and focused",
                "🎯 Include diverse conversation patterns"
            ])
        elif format_type == 'openai_completion':
            recommendations.extend([
                "📝 Keep prompts and completions consistent in style",
                "🎯 Ensure completions directly answer prompts",
                "⚖️ Balance prompt and completion lengths"
            ])
        elif format_type.startswith('huggingface'):
            recommendations.extend([
                "📚 Include diverse instruction types",
                "🔍 Ensure outputs are comprehensive and accurate",
                "📖 Add context when helpful for understanding"
            ])
        
        for rec in recommendations:
            st.write(f"• {rec}")
    
    def render_export_options(self, content_data: List[Dict]):
        """Render export options with format optimization"""
        
        st.markdown("---")
        st.markdown("**📦 Export Options:**")
        
        selected_format = st.session_state['preview_format']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export JSONL"):
                self.export_formatted_data(content_data, selected_format, 'jsonl')
        
        with col2:
            if st.button("📥 Export JSON"):
                self.export_formatted_data(content_data, selected_format, 'json')
        
        with col3:
            if st.button("📥 Export CSV"):
                self.export_formatted_data(content_data, selected_format, 'csv')
        
        # Export settings
        with st.expander("⚙️ Export Settings"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                include_metadata = st.checkbox(
                    "Include Metadata",
                    value=False,
                    help="Include quality scores and other metadata"
                )
                
                validate_before_export = st.checkbox(
                    "Validate Before Export",
                    value=True,
                    help="Only export valid items"
                )
            
            with col2:
                pretty_print = st.checkbox(
                    "Pretty Print JSON",
                    value=False,
                    help="Format JSON with indentation (larger file size)"
                )
                
                ensure_ascii = st.checkbox(
                    "Ensure ASCII",
                    value=False,
                    help="Escape non-ASCII characters"
                )
    
    def convert_to_format(self, content_data: List[Dict], format_type: str) -> List[Dict]:
        """Convert content data to specified format"""
        
        formatted_data = []
        
        for item in content_data:
            try:
                formatted_item = self.convert_single_item(item, format_type)
                if formatted_item:
                    formatted_data.append(formatted_item)
            except Exception as e:
                self.logger.error(f"Failed to convert item to {format_type}: {str(e)}")
                continue
        
        return formatted_data
    
    def convert_single_item(self, item: Dict, format_type: str) -> Optional[Dict]:
        """Convert a single item to the specified format"""
        
        # Extract content from item
        question = item.get('question', item.get('input', ''))
        answer = item.get('answer', item.get('output', item.get('response', '')))
        context = item.get('context', '')
        instruction = item.get('instruction', '')
        
        if format_type == 'openai_chat':
            return {
                'messages': [
                    {'role': 'system', 'content': instruction or 'You are a helpful assistant.'},
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': answer}
                ]
            }
        
        elif format_type == 'openai_completion':
            return {
                'prompt': question,
                'completion': answer
            }
        
        elif format_type == 'huggingface_instruction':
            result = {
                'instruction': instruction or 'Answer the following question.',
                'output': answer
            }
            if question:
                result['input'] = question
            return result
        
        elif format_type == 'huggingface_qa':
            result = {
                'question': question,
                'answer': answer
            }
            if context:
                result['context'] = context
            return result
        
        elif format_type == 'alpaca':
            return {
                'instruction': instruction or 'Answer the following question.',
                'input': question,
                'output': answer
            }
        
        elif format_type == 'sharegpt':
            return {
                'conversations': [
                    {'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': answer}
                ]
            }
        
        return None
    
    def validate_format(self, item: Dict, format_type: str) -> Tuple[bool, List[str]]:
        """Validate a single item against format requirements"""
        
        errors = []
        
        if format_type == 'openai_chat':
            if 'messages' not in item:
                errors.append("Missing 'messages' field")
            else:
                messages = item['messages']
                if not isinstance(messages, list):
                    errors.append("'messages' must be a list")
                else:
                    for i, msg in enumerate(messages):
                        if not isinstance(msg, dict):
                            errors.append(f"Message {i} must be a dictionary")
                        elif 'role' not in msg or 'content' not in msg:
                            errors.append(f"Message {i} missing 'role' or 'content'")
                        elif msg['role'] not in ['system', 'user', 'assistant']:
                            errors.append(f"Message {i} has invalid role: {msg['role']}")
        
        elif format_type == 'openai_completion':
            if 'prompt' not in item:
                errors.append("Missing 'prompt' field")
            if 'completion' not in item:
                errors.append("Missing 'completion' field")
        
        elif format_type == 'huggingface_instruction':
            if 'instruction' not in item:
                errors.append("Missing 'instruction' field")
            if 'output' not in item:
                errors.append("Missing 'output' field")
        
        elif format_type == 'huggingface_qa':
            if 'question' not in item:
                errors.append("Missing 'question' field")
            if 'answer' not in item:
                errors.append("Missing 'answer' field")
        
        elif format_type == 'alpaca':
            required_fields = ['instruction', 'input', 'output']
            for field in required_fields:
                if field not in item:
                    errors.append(f"Missing '{field}' field")
        
        elif format_type == 'sharegpt':
            if 'conversations' not in item:
                errors.append("Missing 'conversations' field")
            else:
                conversations = item['conversations']
                if not isinstance(conversations, list):
                    errors.append("'conversations' must be a list")
                else:
                    for i, conv in enumerate(conversations):
                        if not isinstance(conv, dict):
                            errors.append(f"Conversation {i} must be a dictionary")
                        elif 'from' not in conv or 'value' not in conv:
                            errors.append(f"Conversation {i} missing 'from' or 'value'")
        
        # Common validations
        for key, value in item.items():
            if isinstance(value, str):
                # Check for problematic characters
                if '\n' in value and format_type.startswith('openai'):
                    errors.append(f"Field '{key}' contains newlines (may cause JSONL parsing issues)")
                
                # Check for empty content
                if not value.strip():
                    errors.append(f"Field '{key}' is empty or whitespace only")
        
        return len(errors) == 0, errors
    
    def validate_all_content(self, content_data: List[Dict], format_type: str) -> Dict[str, Any]:
        """Validate all content against format requirements"""
        
        total_items = len(content_data)
        valid_items = 0
        invalid_items = 0
        errors = []
        
        for i, item in enumerate(content_data):
            try:
                formatted_item = self.convert_single_item(item, format_type)
                if formatted_item:
                    is_valid, item_errors = self.validate_format(formatted_item, format_type)
                    
                    if is_valid:
                        valid_items += 1
                    else:
                        invalid_items += 1
                        for error in item_errors:
                            errors.append({
                                'item_index': i + 1,
                                'error': error
                            })
                else:
                    invalid_items += 1
                    errors.append({
                        'item_index': i + 1,
                        'error': 'Failed to convert to target format'
                    })
            
            except Exception as e:
                invalid_items += 1
                errors.append({
                    'item_index': i + 1,
                    'error': f'Conversion error: {str(e)}'
                })
        
        return {
            'total_items': total_items,
            'valid_items': valid_items,
            'invalid_items': invalid_items,
            'errors': errors
        }
    
    def export_formatted_data(self, content_data: List[Dict], format_type: str, export_format: str):
        """Export formatted data in specified format"""
        
        try:
            # Convert to target format
            formatted_data = self.convert_to_format(content_data, format_type)
            
            if not formatted_data:
                st.error("No valid data to export")
                return
            
            # Generate export data
            if export_format == 'jsonl':
                export_data = '\n'.join(json.dumps(item, ensure_ascii=False) for item in formatted_data)
                mime_type = 'application/jsonl'
                file_extension = 'jsonl'
            
            elif export_format == 'json':
                export_data = json.dumps(formatted_data, indent=2, ensure_ascii=False)
                mime_type = 'application/json'
                file_extension = 'json'
            
            elif export_format == 'csv':
                # Flatten data for CSV
                flattened_data = []
                for item in formatted_data:
                    flat_item = self.flatten_dict(item)
                    flattened_data.append(flat_item)
                
                df = pd.DataFrame(flattened_data)
                export_data = df.to_csv(index=False)
                mime_type = 'text/csv'
                file_extension = 'csv'
            
            else:
                st.error(f"Unsupported export format: {export_format}")
                return
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"training_data_{format_type}_{timestamp}.{file_extension}"
            
            # Download button
            st.download_button(
                f"📥 Download {export_format.upper()}",
                data=export_data,
                file_name=filename,
                mime=mime_type
            )
            
            # Log export
            log_event("format_preview_export", {
                "format_type": format_type,
                "export_format": export_format,
                "item_count": len(formatted_data)
            }, "format_preview")
            
            st.success(f"✅ {len(formatted_data)} items ready for download!")
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            self.logger.error(f"Format preview export failed: {str(e)}")
    
    def flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV export"""
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)

# Integration function for main app
def render_format_preview(content_data: List[Dict]):
    """
    Render format preview in main app
    
    Usage:
    from modules.format_preview_engine import render_format_preview
    
    render_format_preview(enhanced_content)
    """
    
    preview_engine = FormatPreviewEngine()
    preview_engine.render_format_preview_interface(content_data)

# Compact format preview for sidebar
def render_compact_format_preview(content_data: List[Dict]):
    """
    Render compact format preview for sidebar
    
    Usage:
    with st.sidebar:
        render_compact_format_preview(enhanced_content)
    """
    
    if not content_data:
        return
    
    st.markdown("### 📋 Format Preview")
    
    # Quick format selection
    format_options = ['openai_chat', 'openai_completion', 'huggingface_instruction']
    format_names = ['OpenAI Chat', 'OpenAI Completion', 'HF Instruction']
    
    selected_format = st.selectbox(
        "Format",
        format_options,
        format_func=lambda x: format_names[format_options.index(x)]
    )
    
    # Quick validation
    preview_engine = FormatPreviewEngine()
    validation_results = preview_engine.validate_all_content(content_data[:10], selected_format)
    
    if validation_results['total_items'] > 0:
        success_rate = (validation_results['valid_items'] / validation_results['total_items']) * 100
        
        if success_rate == 100:
            st.success(f"✅ {success_rate:.0f}% valid")
        elif success_rate >= 80:
            st.warning(f"⚠️ {success_rate:.0f}% valid")
        else:
            st.error(f"❌ {success_rate:.0f}% valid")
    
    if st.button("📋 View Full Preview"):
        st.session_state['show_format_preview'] = True

if __name__ == "__main__":
    # Test the format preview engine
    st.set_page_config(page_title="Format Preview Test", layout="wide")
    
    st.title("Format Preview Engine Test")
    
    # Sample data
    sample_data = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "context": "France is a country in Western Europe."
        },
        {
            "question": "How do you make coffee?",
            "answer": "To make coffee, you need coffee beans, hot water, and a brewing method like a coffee maker or French press.",
            "instruction": "Provide a helpful cooking instruction."
        }
    ]
    
    # Render format preview
    render_format_preview(sample_data)

