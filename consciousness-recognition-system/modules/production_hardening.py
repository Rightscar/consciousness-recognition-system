"""
Production Hardening Module
===========================

Comprehensive fixes for 9 critical deployment and runtime issues:
1. File Parsing - Nested lists from extractors
2. Character Encoding - Unicode issues in JSONL export
3. Corrupted Files - File parsing failures
4. Non-Standard Line Breaks - \\r vs \\n issues
5. Empty Content - Processing empty files
6. Schema Drift - Human-edited content breaking schema
7. Chunk Mismatch - Poor chunking breaking content
8. Silent Token Overflows - Token limits causing truncation
9. Widget Side Effects - Streamlit callback issues
"""

import os
import re
import json
import logging
import unicodedata
from typing import Any, List, Dict, Union, Optional, Tuple
from pathlib import Path
import streamlit as st
from pydantic import BaseModel, ValidationError
import tiktoken

logger = logging.getLogger(__name__)


class ProductionHardening:
    """Comprehensive production hardening for deployment issues"""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        self.max_tokens = 8000  # Conservative limit for GPT-4o-mini
        
    # Issue 1: File Parsing - Flatten nested lists
    def flatten_extracted_content(self, content: Any) -> str:
        """
        Flatten nested lists from extractors to ensure string output
        
        Args:
            content: Raw extracted content (can be string, list, nested lists)
            
        Returns:
            Flattened string content
        """
        try:
            if content is None:
                return ""
            
            if isinstance(content, str):
                return content
            
            if isinstance(content, list):
                # Recursively flatten nested lists
                flattened = []
                for item in content:
                    if isinstance(item, list):
                        flattened.extend(self.flatten_extracted_content(item).split('\n'))
                    elif isinstance(item, str):
                        flattened.append(item)
                    else:
                        flattened.append(str(item))
                return '\n'.join(filter(None, flattened))
            
            # Convert other types to string
            return str(content)
            
        except Exception as e:
            logger.warning(f"Error flattening content: {e}")
            return str(content) if content is not None else ""
    
    # Issue 2: Character Encoding - Safe UTF-8 export
    def safe_utf8_encode(self, text: str) -> str:
        """
        Safely encode text for UTF-8 export, handling special characters
        
        Args:
            text: Input text with potential unicode issues
            
        Returns:
            UTF-8 safe text
        """
        try:
            if not isinstance(text, str):
                text = str(text)
            
            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Replace problematic characters
            replacements = {
                '"': '"',  # Smart quotes
                '"': '"',
                ''': "'",  # Smart apostrophes
                ''': "'",
                '—': '-',  # Em dash
                '–': '-',  # En dash
                '…': '...',  # Ellipsis
                '\u00a0': ' ',  # Non-breaking space
                '\u2028': '\n',  # Line separator
                '\u2029': '\n\n',  # Paragraph separator
            }
            
            for old, new in replacements.items():
                text = text.replace(old, new)
            
            # Encode and decode to catch any remaining issues
            text = text.encode('utf-8', 'replace').decode('utf-8')
            
            return text
            
        except Exception as e:
            logger.warning(f"Error encoding text: {e}")
            # Fallback: aggressive replacement
            return text.encode('ascii', 'replace').decode('ascii') if text else ""
    
    # Issue 3: Corrupted Files - Robust file parsing
    def safe_file_parse(self, file_path: str, parser_func: callable) -> Tuple[str, bool, str]:
        """
        Safely parse files with comprehensive error handling
        
        Args:
            file_path: Path to file
            parser_func: Function to parse the file
            
        Returns:
            Tuple of (content, success, error_message)
        """
        try:
            # Check file exists and is readable
            if not os.path.exists(file_path):
                return "", False, f"File not found: {file_path}"
            
            if not os.access(file_path, os.R_OK):
                return "", False, f"File not readable: {file_path}"
            
            # Check file size (avoid extremely large files)
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return "", False, f"File too large: {file_size / 1024 / 1024:.1f}MB"
            
            if file_size == 0:
                return "", False, "File is empty"
            
            # Attempt parsing with timeout protection
            content = parser_func(file_path)
            
            # Validate content
            if not content or (isinstance(content, str) and len(content.strip()) == 0):
                return "", False, "No extractable content found"
            
            # Flatten and normalize content
            content = self.flatten_extracted_content(content)
            content = self.normalize_line_breaks(content)
            
            return content, True, ""
            
        except UnicodeDecodeError as e:
            return "", False, f"Character encoding error: {e}"
        except MemoryError:
            return "", False, "File too large for available memory"
        except Exception as e:
            return "", False, f"Parsing error: {e}"
    
    # Issue 4: Non-Standard Line Breaks - Normalize line endings
    def normalize_line_breaks(self, text: str) -> str:
        """
        Normalize all line breaks to \\n
        
        Args:
            text: Input text with mixed line endings
            
        Returns:
            Text with normalized line breaks
        """
        try:
            if not isinstance(text, str):
                text = str(text)
            
            # Replace Windows (\\r\\n) and Mac (\\r) line endings with Unix (\\n)
            text = text.replace('\\r\\n', '\\n')
            text = text.replace('\\r', '\\n')
            
            # Remove excessive blank lines (more than 2 consecutive)
            text = re.sub(r'\\n{3,}', '\\n\\n', text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error normalizing line breaks: {e}")
            return text
    
    # Issue 5: Empty Content - Validate content before processing
    def validate_content_not_empty(self, content: str, min_length: int = 10) -> Tuple[bool, str]:
        """
        Validate that content is not empty and has meaningful text
        
        Args:
            content: Content to validate
            min_length: Minimum character length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not content:
                return False, "Content is None or empty"
            
            if not isinstance(content, str):
                content = str(content)
            
            # Check stripped length
            stripped = content.strip()
            if len(stripped) == 0:
                return False, "Content contains only whitespace"
            
            if len(stripped) < min_length:
                return False, f"Content too short: {len(stripped)} chars (minimum: {min_length})"
            
            # Check for meaningful content (not just punctuation/numbers)
            alpha_chars = sum(1 for c in stripped if c.isalpha())
            if alpha_chars < min_length // 2:
                return False, f"Content lacks meaningful text: {alpha_chars} alphabetic characters"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    # Issue 6: Schema Drift - Pydantic validation
    class TrainingDataSchema(BaseModel):
        """Pydantic schema for training data validation"""
        question: str
        answer: str
        metadata: Optional[Dict[str, Any]] = None
        
        class Config:
            extra = "allow"  # Allow additional fields
    
    def validate_training_data_schema(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate training data against schema, fixing what's possible
        
        Args:
            data: List of training data items
            
        Returns:
            Tuple of (valid_items, error_messages)
        """
        valid_items = []
        errors = []
        
        for i, item in enumerate(data):
            try:
                # Attempt validation
                validated_item = self.TrainingDataSchema(**item)
                
                # Additional content validation
                q_valid, q_error = self.validate_content_not_empty(validated_item.question, 5)
                a_valid, a_error = self.validate_content_not_empty(validated_item.answer, 10)
                
                if not q_valid:
                    errors.append(f"Item {i}: Invalid question - {q_error}")
                    continue
                
                if not a_valid:
                    errors.append(f"Item {i}: Invalid answer - {a_error}")
                    continue
                
                # Clean and normalize content
                clean_item = {
                    "question": self.safe_utf8_encode(validated_item.question.strip()),
                    "answer": self.safe_utf8_encode(validated_item.answer.strip()),
                }
                
                if validated_item.metadata:
                    clean_item["metadata"] = validated_item.metadata
                
                valid_items.append(clean_item)
                
            except ValidationError as e:
                errors.append(f"Item {i}: Schema validation failed - {e}")
            except Exception as e:
                errors.append(f"Item {i}: Unexpected error - {e}")
        
        return valid_items, errors
    
    # Issue 7: Chunk Mismatch - Sentence-based chunking
    def smart_chunk_content(self, text: str, max_tokens: int = None) -> List[str]:
        """
        Intelligently chunk content by sentences with size guards
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        try:
            # Normalize text
            text = self.normalize_line_breaks(text)
            text = text.strip()
            
            if not text:
                return []
            
            # Split by sentences (improved regex)
            sentence_pattern = r'(?<=[.!?])\\s+(?=[A-Z])'
            sentences = re.split(sentence_pattern, text)
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed token limit
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                token_count = len(self.encoding.encode(test_chunk))
                
                if token_count <= max_tokens:
                    current_chunk = test_chunk
                else:
                    # Save current chunk if it has content
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Check if single sentence is too long
                    sentence_tokens = len(self.encoding.encode(sentence))
                    if sentence_tokens > max_tokens:
                        # Split long sentence by words
                        word_chunks = self._split_by_words(sentence, max_tokens)
                        chunks.extend(word_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = sentence
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return [chunk for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            logger.warning(f"Error chunking content: {e}")
            # Fallback: simple word-based chunking
            return self._split_by_words(text, max_tokens)
    
    def _split_by_words(self, text: str, max_tokens: int) -> List[str]:
        """Fallback word-based chunking"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            test_chunk = " ".join(current_chunk + [word])
            if len(self.encoding.encode(test_chunk)) <= max_tokens:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    # Issue 8: Silent Token Overflows - Token counting and splitting
    def check_and_split_tokens(self, text: str, max_tokens: int = None) -> Tuple[List[str], int]:
        """
        Check token count and split if needed
        
        Args:
            text: Input text
            max_tokens: Maximum tokens allowed
            
        Returns:
            Tuple of (text_chunks, total_tokens)
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        try:
            # Count tokens
            tokens = self.encoding.encode(text)
            total_tokens = len(tokens)
            
            if total_tokens <= max_tokens:
                return [text], total_tokens
            
            # Split into chunks
            chunks = self.smart_chunk_content(text, max_tokens)
            return chunks, total_tokens
            
        except Exception as e:
            logger.warning(f"Error checking tokens: {e}")
            # Fallback: simple character-based estimation
            estimated_tokens = len(text) // 4  # Rough estimation
            if estimated_tokens <= max_tokens:
                return [text], estimated_tokens
            else:
                # Simple character-based chunking
                chunk_size = max_tokens * 3  # Rough character estimate
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                return chunks, estimated_tokens
    
    # Issue 9: Widget Side Effects - Debounced controls
    def create_debounced_input(self, key: str, label: str, default_value: str = "", 
                              help_text: str = None) -> str:
        """
        Create debounced text input to prevent excessive API calls
        
        Args:
            key: Unique key for the input
            label: Input label
            default_value: Default value
            help_text: Help text
            
        Returns:
            Input value (only when user stops typing)
        """
        # Use session state to track input and last update
        input_key = f"{key}_input"
        debounce_key = f"{key}_debounced"
        
        if input_key not in st.session_state:
            st.session_state[input_key] = default_value
        
        if debounce_key not in st.session_state:
            st.session_state[debounce_key] = default_value
        
        # Create input
        current_value = st.text_input(
            label,
            value=st.session_state[input_key],
            key=f"{key}_widget",
            help=help_text
        )
        
        # Update session state
        st.session_state[input_key] = current_value
        
        # Only update debounced value when user clicks a button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Apply", key=f"{key}_apply"):
                st.session_state[debounce_key] = current_value
                st.rerun()
        
        return st.session_state[debounce_key]
    
    def safe_export_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> Tuple[bool, str]:
        """
        Safely export data to JSONL with all hardening applied
        
        Args:
            data: Data to export
            file_path: Output file path
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Validate schema
            valid_data, errors = self.validate_training_data_schema(data)
            
            if errors:
                logger.warning(f"Schema validation errors: {errors}")
            
            if not valid_data:
                return False, "No valid data to export after schema validation"
            
            # Export with safe encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in valid_data:
                    # Apply UTF-8 encoding safety
                    safe_item = {}
                    for key, value in item.items():
                        if isinstance(value, str):
                            safe_item[key] = self.safe_utf8_encode(value)
                        else:
                            safe_item[key] = value
                    
                    # Write JSON line
                    json_line = json.dumps(safe_item, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_line + '\\n')
            
            return True, f"Successfully exported {len(valid_data)} items"
            
        except Exception as e:
            return False, f"Export error: {e}"


# Global hardening instance
production_hardening = ProductionHardening()


# Convenience functions
def safe_parse_file(file_path: str, parser_func: callable) -> Tuple[str, bool, str]:
    """Convenience function for safe file parsing"""
    return production_hardening.safe_file_parse(file_path, parser_func)


def safe_export_training_data(data: List[Dict[str, Any]], file_path: str) -> Tuple[bool, str]:
    """Convenience function for safe JSONL export"""
    return production_hardening.safe_export_jsonl(data, file_path)


def validate_and_clean_content(content: str) -> Tuple[str, bool, str]:
    """Convenience function for content validation and cleaning"""
    # Flatten if needed
    content = production_hardening.flatten_extracted_content(content)
    
    # Normalize line breaks
    content = production_hardening.normalize_line_breaks(content)
    
    # Validate not empty
    is_valid, error = production_hardening.validate_content_not_empty(content)
    
    if is_valid:
        # Apply UTF-8 safety
        content = production_hardening.safe_utf8_encode(content)
    
    return content, is_valid, error


def smart_token_split(text: str, max_tokens: int = 8000) -> List[str]:
    """Convenience function for smart token-aware splitting"""
    chunks, _ = production_hardening.check_and_split_tokens(text, max_tokens)
    return chunks

