"""
Shared Text Utilities
====================

Single source of truth for all text processing operations across QA modules.
Eliminates code duplication and ensures consistent text handling.

Features:
- Text normalization and cleaning
- Content extraction and parsing
- Format validation and conversion
- Feature extraction for analysis
- Encoding and decoding utilities
"""

import re
import string
import unicodedata
from typing import Dict, List, Tuple, Any, Optional, Union
import json
from datetime import datetime
import hashlib

def normalize_text(text: str, 
                  remove_extra_whitespace: bool = True,
                  normalize_unicode: bool = True,
                  remove_special_chars: bool = False,
                  lowercase: bool = False) -> str:
    """
    Normalize text with configurable options.
    
    Args:
        text: Input text to normalize
        remove_extra_whitespace: Remove extra spaces, tabs, newlines
        normalize_unicode: Normalize unicode characters
        remove_special_chars: Remove special characters
        lowercase: Convert to lowercase
    
    Returns:
        Normalized text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Unicode normalization
    if normalize_unicode:
        text = unicodedata.normalize('NFKC', text)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    # Remove special characters (keep basic punctuation)
    if remove_special_chars:
        # Keep letters, numbers, basic punctuation, and spaces
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+', '', text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    return text

def clean_content(content: str, 
                 aggressive: bool = False) -> str:
    """
    Clean content for processing with different levels of aggressiveness.
    
    Args:
        content: Input content to clean
        aggressive: Use aggressive cleaning (removes more formatting)
    
    Returns:
        Cleaned content string
    """
    if not content:
        return ""
    
    # Basic cleaning
    content = normalize_text(content, remove_extra_whitespace=True)
    
    # Remove common artifacts
    content = re.sub(r'\r\n', '\n', content)  # Normalize line endings
    content = re.sub(r'\n{3,}', '\n\n', content)  # Limit consecutive newlines
    
    if aggressive:
        # Remove formatting artifacts
        content = re.sub(r'[\u200b-\u200d\ufeff]', '', content)  # Zero-width chars
        content = re.sub(r'[^\S\n]+', ' ', content)  # Normalize whitespace except newlines
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Clean paragraph breaks
    
    return content.strip()

def extract_qa_pairs(text: str) -> List[Dict[str, str]]:
    """
    Extract Q&A pairs from text using multiple patterns.
    
    Args:
        text: Input text containing Q&A pairs
    
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    qa_pairs = []
    
    # Pattern 1: Q: ... A: ...
    pattern1 = r'Q[:\-\s]*(.+?)\s*A[:\-\s]*(.+?)(?=Q[:\-\s]|$)'
    matches1 = re.findall(pattern1, text, re.DOTALL | re.IGNORECASE)
    
    for q, a in matches1:
        qa_pairs.append({
            'question': clean_content(q),
            'answer': clean_content(a)
        })
    
    # Pattern 2: Question: ... Answer: ...
    pattern2 = r'Question[:\-\s]*(.+?)\s*Answer[:\-\s]*(.+?)(?=Question[:\-\s]|$)'
    matches2 = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
    
    for q, a in matches2:
        qa_pairs.append({
            'question': clean_content(q),
            'answer': clean_content(a)
        })
    
    # Pattern 3: Numbered Q&A (1. Q: ... A: ...)
    pattern3 = r'\d+\.\s*Q[:\-\s]*(.+?)\s*A[:\-\s]*(.+?)(?=\d+\.\s*Q|$)'
    matches3 = re.findall(pattern3, text, re.DOTALL | re.IGNORECASE)
    
    for q, a in matches3:
        qa_pairs.append({
            'question': clean_content(q),
            'answer': clean_content(a)
        })
    
    return qa_pairs

def detect_content_structure(text: str) -> Dict[str, Any]:
    """
    Detect the structure and format of content.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Dictionary with structure information
    """
    if not text:
        return {"type": "empty", "confidence": 1.0}
    
    text_clean = clean_content(text)
    
    # Count various patterns
    qa_patterns = len(re.findall(r'[Qq][:\-\s]', text_clean))
    answer_patterns = len(re.findall(r'[Aa][:\-\s]', text_clean))
    question_marks = text_clean.count('?')
    paragraphs = len([p for p in text_clean.split('\n\n') if p.strip()])
    
    # Dialogue patterns
    dialogue_patterns = len(re.findall(r'^[A-Z][a-z]*:', text_clean, re.MULTILINE))
    
    # Calculate confidence scores
    qa_confidence = min(1.0, (qa_patterns + answer_patterns) / 10)
    dialogue_confidence = min(1.0, dialogue_patterns / 5)
    monologue_confidence = 1.0 - max(qa_confidence, dialogue_confidence)
    
    # Determine primary type
    if qa_confidence > 0.3 and qa_patterns > 0 and answer_patterns > 0:
        content_type = "qa"
        confidence = qa_confidence
    elif dialogue_confidence > 0.3:
        content_type = "dialogue"
        confidence = dialogue_confidence
    elif paragraphs > 1:
        content_type = "monologue"
        confidence = monologue_confidence
    else:
        content_type = "mixed"
        confidence = 0.5
    
    return {
        "type": content_type,
        "confidence": confidence,
        "stats": {
            "qa_patterns": qa_patterns,
            "answer_patterns": answer_patterns,
            "question_marks": question_marks,
            "paragraphs": paragraphs,
            "dialogue_patterns": dialogue_patterns,
            "word_count": len(text_clean.split()),
            "character_count": len(text_clean)
        }
    }

def extract_text_features(text: str) -> Dict[str, float]:
    """
    Extract numerical features from text for analysis.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Dictionary of numerical features
    """
    if not text:
        return {}
    
    text_clean = clean_content(text)
    words = text_clean.split()
    sentences = re.split(r'[.!?]+', text_clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Basic metrics
    word_count = len(words)
    char_count = len(text_clean)
    sentence_count = len(sentences)
    
    # Advanced metrics
    avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Vocabulary diversity
    unique_words = len(set(word.lower() for word in words))
    vocabulary_diversity = unique_words / max(word_count, 1)
    
    # Punctuation density
    punctuation_count = sum(1 for char in text_clean if char in string.punctuation)
    punctuation_density = punctuation_count / max(char_count, 1)
    
    # Question density
    question_count = text_clean.count('?')
    question_density = question_count / max(sentence_count, 1)
    
    return {
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "vocabulary_diversity": vocabulary_diversity,
        "punctuation_density": punctuation_density,
        "question_density": question_density,
        "readability_score": calculate_readability_score(text_clean)
    }

def calculate_readability_score(text: str) -> float:
    """
    Calculate a simple readability score (0-1, higher = more readable).
    
    Args:
        text: Input text to analyze
    
    Returns:
        Readability score between 0 and 1
    """
    if not text:
        return 0.0
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not words or not sentences:
        return 0.0
    
    # Simple readability metrics
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Normalize to 0-1 scale (lower values = more readable)
    sentence_score = max(0, 1 - (avg_sentence_length - 10) / 20)
    word_score = max(0, 1 - (avg_word_length - 4) / 6)
    
    return (sentence_score + word_score) / 2

def validate_text_format(text: str, expected_format: str) -> Dict[str, Any]:
    """
    Validate text against expected format.
    
    Args:
        text: Input text to validate
        expected_format: Expected format (qa, dialogue, monologue, json, etc.)
    
    Returns:
        Validation result dictionary
    """
    if not text:
        return {"valid": False, "errors": ["Empty text"], "confidence": 0.0}
    
    errors = []
    warnings = []
    
    if expected_format == "qa":
        qa_pairs = extract_qa_pairs(text)
        if not qa_pairs:
            errors.append("No Q&A pairs found")
        elif len(qa_pairs) < 2:
            warnings.append("Only one Q&A pair found")
        
        # Check for incomplete pairs
        for i, pair in enumerate(qa_pairs):
            if not pair['question'].strip():
                errors.append(f"Empty question in pair {i+1}")
            if not pair['answer'].strip():
                errors.append(f"Empty answer in pair {i+1}")
    
    elif expected_format == "json":
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
    
    elif expected_format == "dialogue":
        dialogue_patterns = re.findall(r'^[A-Z][a-z]*:', text, re.MULTILINE)
        if len(dialogue_patterns) < 2:
            errors.append("Insufficient dialogue patterns found")
    
    # Calculate confidence
    confidence = 1.0 if not errors else (0.5 if not warnings else 0.0)
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "confidence": confidence
    }

def convert_text_format(text: str, 
                       from_format: str, 
                       to_format: str) -> str:
    """
    Convert text from one format to another.
    
    Args:
        text: Input text to convert
        from_format: Source format
        to_format: Target format
    
    Returns:
        Converted text
    """
    if from_format == to_format:
        return text
    
    if from_format == "qa" and to_format == "json":
        qa_pairs = extract_qa_pairs(text)
        return json.dumps(qa_pairs, indent=2, ensure_ascii=False)
    
    elif from_format == "json" and to_format == "qa":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                result = []
                for item in data:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        result.append(f"Q: {item['question']}")
                        result.append(f"A: {item['answer']}")
                        result.append("")
                return "\n".join(result)
        except:
            pass
    
    # Default: return original text
    return text

def generate_text_hash(text: str) -> str:
    """
    Generate a hash for text content (useful for deduplication).
    
    Args:
        text: Input text
    
    Returns:
        SHA-256 hash of normalized text
    """
    normalized = normalize_text(text, lowercase=True, remove_extra_whitespace=True)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def split_text_into_chunks(text: str, 
                          max_chunk_size: int = 1000,
                          overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Input text to split
        max_chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending within last 100 characters
            search_start = max(start, end - 100)
            sentence_end = -1
            
            for i in range(end, search_start, -1):
                if text[i] in '.!?':
                    sentence_end = i + 1
                    break
            
            if sentence_end > 0:
                end = sentence_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

def merge_text_chunks(chunks: List[str], separator: str = "\n\n") -> str:
    """
    Merge text chunks back into single text.
    
    Args:
        chunks: List of text chunks
        separator: Separator between chunks
    
    Returns:
        Merged text
    """
    return separator.join(chunk.strip() for chunk in chunks if chunk.strip())

# Utility functions for common operations
def is_empty_or_whitespace(text: str) -> bool:
    """Check if text is empty or contains only whitespace."""
    return not text or not text.strip()

def count_words(text: str) -> int:
    """Count words in text."""
    return len(clean_content(text).split())

def count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r'[.!?]+', clean_content(text))
    return len([s for s in sentences if s.strip()])

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text (simple frequency-based)."""
    words = normalize_text(text, lowercase=True).split()
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        word = word.strip(string.punctuation)
        if len(word) > 2 and word.lower() not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]

