"""
Intelligent Text Processor
==========================

Handles automatic language detection and unicode normalization to prevent
garbled text from non-UTF-8 sources and international documents.

Features:
- Automatic language detection using multiple methods
- Unicode normalization with language-specific rules
- Encoding detection and correction
- Text quality assessment and improvement
- Multi-language support with fallbacks
"""

import unicodedata
import re
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# Language detection libraries
try:
    import langdetect
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

from modules.logger import get_logger
from shared.text_utils import clean_content

class IntelligentTextProcessor:
    """
    Intelligent text processor with language detection and unicode normalization.
    
    Automatically detects document language and applies appropriate normalization
    rules to ensure clean, properly encoded text for AI processing.
    """
    
    def __init__(self, 
                 default_language: str = 'en',
                 confidence_threshold: float = 0.7,
                 fallback_encoding: str = 'utf-8'):
        """
        Initialize the intelligent text processor.
        
        Args:
            default_language: Default language code (ISO 639-1)
            confidence_threshold: Minimum confidence for language detection
            fallback_encoding: Fallback encoding if detection fails
        """
        self.logger = get_logger("intelligent_text_processor")
        self.default_language = default_language
        self.confidence_threshold = confidence_threshold
        self.fallback_encoding = fallback_encoding
        
        # Language-specific processing rules
        self.language_rules = self._initialize_language_rules()
        
        # Unicode normalization forms
        self.normalization_forms = ['NFC', 'NFKC', 'NFD', 'NFKD']
        
        self.logger.info("Intelligent text processor initialized")
    
    def normalize_multilingual_text(self, text: str, 
                                   detected_language: str = None,
                                   encoding: str = None) -> Dict[str, Any]:
        """
        Normalize multilingual text with language detection and unicode normalization.
        
        Args:
            text: Input text to normalize
            detected_language: Pre-detected language (optional)
            encoding: Known encoding (optional)
        
        Returns:
            Dictionary with normalized text and processing metadata
        """
        try:
            if not text or not text.strip():
                return {
                    'normalized_text': '',
                    'language': self.default_language,
                    'confidence': 0.0,
                    'encoding': 'utf-8',
                    'normalization_applied': [],
                    'issues_found': [],
                    'metadata': {}
                }
            
            self.logger.info(f"Processing text: {len(text)} characters")
            
            # Step 1: Detect and fix encoding issues
            encoding_result = self._detect_and_fix_encoding(text, encoding)
            text = encoding_result['text']
            detected_encoding = encoding_result['encoding']
            
            # Step 2: Detect language
            language_result = self._detect_language(text, detected_language)
            language = language_result['language']
            language_confidence = language_result['confidence']
            
            # Step 3: Apply unicode normalization
            normalization_result = self._apply_unicode_normalization(text, language)
            normalized_text = normalization_result['text']
            normalization_applied = normalization_result['forms_applied']
            
            # Step 4: Apply language-specific rules
            language_result = self._apply_language_specific_rules(normalized_text, language)
            final_text = language_result['text']
            language_rules_applied = language_result['rules_applied']
            
            # Step 5: Quality assessment and final cleanup
            quality_result = self._assess_and_improve_quality(final_text)
            final_text = quality_result['text']
            issues_found = quality_result['issues_found']
            improvements_made = quality_result['improvements_made']
            
            # Compile results
            result = {
                'normalized_text': final_text,
                'language': language,
                'confidence': language_confidence,
                'encoding': detected_encoding,
                'normalization_applied': normalization_applied,
                'language_rules_applied': language_rules_applied,
                'issues_found': issues_found,
                'improvements_made': improvements_made,
                'metadata': {
                    'original_length': len(text),
                    'final_length': len(final_text),
                    'processing_steps': [
                        'encoding_detection',
                        'language_detection', 
                        'unicode_normalization',
                        'language_specific_rules',
                        'quality_improvement'
                    ],
                    'quality_score': quality_result['quality_score']
                }
            }
            
            self.logger.info(f"Text processing complete: {language} ({language_confidence:.2f}), "
                           f"{len(final_text)} chars, quality: {quality_result['quality_score']:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text normalization failed: {str(e)}")
            return {
                'normalized_text': text,  # Return original on failure
                'language': self.default_language,
                'confidence': 0.0,
                'encoding': 'unknown',
                'normalization_applied': [],
                'issues_found': [f"Processing error: {str(e)}"],
                'metadata': {'error': str(e)}
            }
    
    def _detect_and_fix_encoding(self, text: str, known_encoding: str = None) -> Dict[str, Any]:
        """Detect and fix text encoding issues."""
        if known_encoding:
            return {'text': text, 'encoding': known_encoding, 'confidence': 1.0}
        
        # If chardet is available, use it for encoding detection
        if CHARDET_AVAILABLE and isinstance(text, bytes):
            detection = chardet.detect(text)
            if detection['confidence'] > 0.8:
                try:
                    decoded_text = text.decode(detection['encoding'])
                    return {
                        'text': decoded_text,
                        'encoding': detection['encoding'],
                        'confidence': detection['confidence']
                    }
                except:
                    pass
        
        # Handle common encoding issues in string text
        if isinstance(text, str):
            # Fix common encoding artifacts
            fixed_text = self._fix_common_encoding_issues(text)
            return {
                'text': fixed_text,
                'encoding': 'utf-8',
                'confidence': 0.9 if fixed_text != text else 1.0
            }
        
        # Fallback
        return {
            'text': str(text),
            'encoding': self.fallback_encoding,
            'confidence': 0.5
        }
    
    def _fix_common_encoding_issues(self, text: str) -> str:
        """Fix common encoding artifacts in text."""
        # Common encoding fixes
        fixes = {
            'â€™': "'",  # Smart apostrophe
            'â€œ': '"',  # Smart quote open
            'â€': '"',   # Smart quote close
            'â€"': '—',  # Em dash
            'â€"': '–',  # En dash
            'Ã¡': 'á',   # á with encoding issue
            'Ã©': 'é',   # é with encoding issue
            'Ã­': 'í',   # í with encoding issue
            'Ã³': 'ó',   # ó with encoding issue
            'Ãº': 'ú',   # ú with encoding issue
            'Ã±': 'ñ',   # ñ with encoding issue
        }
        
        for broken, fixed in fixes.items():
            text = text.replace(broken, fixed)
        
        return text
    
    def _detect_language(self, text: str, known_language: str = None) -> Dict[str, Any]:
        """Detect the language of the text."""
        if known_language:
            return {'language': known_language, 'confidence': 1.0, 'method': 'provided'}
        
        if not LANGDETECT_AVAILABLE:
            self.logger.warning("langdetect not available, using default language")
            return {'language': self.default_language, 'confidence': 0.5, 'method': 'default'}
        
        try:
            # Use langdetect for primary detection
            detected_langs = detect_langs(text)
            
            if detected_langs and detected_langs[0].prob >= self.confidence_threshold:
                primary_lang = detected_langs[0]
                return {
                    'language': primary_lang.lang,
                    'confidence': primary_lang.prob,
                    'method': 'langdetect',
                    'alternatives': [(lang.lang, lang.prob) for lang in detected_langs[1:3]]
                }
            
            # Fallback to simple heuristics
            heuristic_result = self._detect_language_heuristic(text)
            return heuristic_result
            
        except LangDetectException as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            return {'language': self.default_language, 'confidence': 0.3, 'method': 'fallback'}
    
    def _detect_language_heuristic(self, text: str) -> Dict[str, Any]:
        """Simple heuristic-based language detection."""
        # Character frequency analysis for common languages
        language_patterns = {
            'en': [r'\bthe\b', r'\band\b', r'\bof\b', r'\bto\b', r'\ba\b'],
            'es': [r'\bel\b', r'\bla\b', r'\bde\b', r'\by\b', r'\ben\b'],
            'fr': [r'\ble\b', r'\bde\b', r'\bet\b', r'\bà\b', r'\bun\b'],
            'de': [r'\bder\b', r'\bdie\b', r'\bund\b', r'\bin\b', r'\bden\b'],
            'it': [r'\bil\b', r'\bdi\b', r'\be\b', r'\bla\b', r'\bche\b'],
            'pt': [r'\bo\b', r'\bde\b', r'\be\b', r'\ba\b', r'\bque\b'],
        }
        
        text_lower = text.lower()
        scores = {}
        
        for lang, patterns in language_patterns.items():
            score = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            scores[lang] = score
        
        if scores:
            best_lang = max(scores, key=scores.get)
            max_score = scores[best_lang]
            total_words = len(text.split())
            confidence = min(0.8, max_score / max(total_words * 0.1, 1))
            
            return {
                'language': best_lang,
                'confidence': confidence,
                'method': 'heuristic',
                'scores': scores
            }
        
        return {'language': self.default_language, 'confidence': 0.2, 'method': 'default'}
    
    def _apply_unicode_normalization(self, text: str, language: str) -> Dict[str, Any]:
        """Apply appropriate unicode normalization."""
        forms_applied = []
        normalized_text = text
        
        # Choose normalization form based on language and content
        if language in ['ar', 'he', 'fa']:  # RTL languages
            # Use NFC for RTL languages to preserve character composition
            normalized_text = unicodedata.normalize('NFC', normalized_text)
            forms_applied.append('NFC')
        elif language in ['ja', 'ko', 'zh']:  # CJK languages
            # Use NFKC for CJK to handle compatibility characters
            normalized_text = unicodedata.normalize('NFKC', normalized_text)
            forms_applied.append('NFKC')
        else:
            # Use NFKC for most Western languages
            normalized_text = unicodedata.normalize('NFKC', normalized_text)
            forms_applied.append('NFKC')
        
        # Additional normalization for specific issues
        if self._has_combining_characters(text):
            # Apply NFC to properly compose combining characters
            normalized_text = unicodedata.normalize('NFC', normalized_text)
            if 'NFC' not in forms_applied:
                forms_applied.append('NFC')
        
        return {
            'text': normalized_text,
            'forms_applied': forms_applied,
            'changes_made': text != normalized_text
        }
    
    def _has_combining_characters(self, text: str) -> bool:
        """Check if text contains combining characters."""
        return any(unicodedata.combining(char) for char in text)
    
    def _apply_language_specific_rules(self, text: str, language: str) -> Dict[str, Any]:
        """Apply language-specific text processing rules."""
        rules_applied = []
        processed_text = text
        
        if language in self.language_rules:
            rules = self.language_rules[language]
            
            for rule_name, rule_func in rules.items():
                try:
                    processed_text = rule_func(processed_text)
                    rules_applied.append(rule_name)
                except Exception as e:
                    self.logger.warning(f"Language rule '{rule_name}' failed: {str(e)}")
        
        return {
            'text': processed_text,
            'rules_applied': rules_applied,
            'changes_made': text != processed_text
        }
    
    def _initialize_language_rules(self) -> Dict[str, Dict[str, callable]]:
        """Initialize language-specific processing rules."""
        return {
            'en': {
                'fix_contractions': self._fix_english_contractions,
                'normalize_quotes': self._normalize_quotes,
                'fix_spacing': self._fix_english_spacing,
            },
            'es': {
                'fix_accents': self._fix_spanish_accents,
                'normalize_punctuation': self._normalize_spanish_punctuation,
            },
            'fr': {
                'fix_accents': self._fix_french_accents,
                'normalize_apostrophes': self._normalize_french_apostrophes,
            },
            'de': {
                'fix_umlauts': self._fix_german_umlauts,
                'normalize_eszett': self._normalize_german_eszett,
            },
            'ar': {
                'normalize_arabic': self._normalize_arabic_text,
                'fix_diacritics': self._fix_arabic_diacritics,
            },
            'zh': {
                'normalize_chinese': self._normalize_chinese_text,
                'fix_punctuation': self._fix_chinese_punctuation,
            },
        }
    
    # Language-specific rule implementations
    def _fix_english_contractions(self, text: str) -> str:
        """Fix common English contraction issues."""
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize various quote characters to standard ASCII."""
        quote_map = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes
            '«': '"', '»': '"',  # French quotes
            '„': '"', '"': '"',  # German quotes
        }
        
        for smart_quote, ascii_quote in quote_map.items():
            text = text.replace(smart_quote, ascii_quote)
        
        return text
    
    def _fix_english_spacing(self, text: str) -> str:
        """Fix common English spacing issues."""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)  # Ensure space after punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
        
        return text.strip()
    
    def _fix_spanish_accents(self, text: str) -> str:
        """Fix common Spanish accent issues."""
        # Common Spanish accent corrections
        fixes = {
            'mas': 'más',
            'si': 'sí',
            'tu': 'tú',
            'el': 'él',
            'mi': 'mí',
        }
        
        # Apply fixes with word boundaries
        for unaccented, accented in fixes.items():
            text = re.sub(rf'\b{unaccented}\b', accented, text)
        
        return text
    
    def _normalize_spanish_punctuation(self, text: str) -> str:
        """Normalize Spanish punctuation marks."""
        # Ensure proper Spanish punctuation
        text = re.sub(r'¿\s*', '¿', text)  # Fix spacing after ¿
        text = re.sub(r'\s*\?', '?', text)  # Fix spacing before ?
        text = re.sub(r'¡\s*', '¡', text)  # Fix spacing after ¡
        text = re.sub(r'\s*!', '!', text)  # Fix spacing before !
        
        return text
    
    def _fix_french_accents(self, text: str) -> str:
        """Fix common French accent issues."""
        # Common French accent corrections
        fixes = {
            'a': 'à',
            'ou': 'où',
            'la': 'là',
            'deja': 'déjà',
            'etre': 'être',
        }
        
        # Apply contextual fixes
        for unaccented, accented in fixes.items():
            # Only apply in specific contexts to avoid over-correction
            if unaccented in ['a', 'ou', 'la']:
                text = re.sub(rf'\b{unaccented}\b(?=\s)', accented, text)
        
        return text
    
    def _normalize_french_apostrophes(self, text: str) -> str:
        """Normalize French apostrophes and contractions."""
        # Fix French contractions
        text = re.sub(r"\bl'", "l'", text)  # Normalize l'
        text = re.sub(r"\bd'", "d'", text)  # Normalize d'
        text = re.sub(r"\bc'", "c'", text)  # Normalize c'
        text = re.sub(r"\bj'", "j'", text)  # Normalize j'
        
        return text
    
    def _fix_german_umlauts(self, text: str) -> str:
        """Fix German umlaut encoding issues."""
        umlaut_fixes = {
            'ae': 'ä',
            'oe': 'ö',
            'ue': 'ü',
            'Ae': 'Ä',
            'Oe': 'Ö',
            'Ue': 'Ü',
        }
        
        # Apply umlaut fixes carefully to avoid false positives
        for ascii_form, umlaut in umlaut_fixes.items():
            # Only replace in German-looking contexts
            text = re.sub(rf'\b\w*{ascii_form}\w*\b', 
                         lambda m: m.group().replace(ascii_form, umlaut), text)
        
        return text
    
    def _normalize_german_eszett(self, text: str) -> str:
        """Normalize German eszett (ß) usage."""
        # Convert ss to ß in appropriate contexts
        # This is a simplified rule - real German orthography is more complex
        text = re.sub(r'\bss\b', 'ß', text)
        
        return text
    
    def _normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text."""
        # Basic Arabic normalization
        # Remove tatweel (Arabic elongation character)
        text = text.replace('\u0640', '')
        
        # Normalize Arabic-Indic digits to ASCII
        arabic_digits = '٠١٢٣٤٥٦٧٨٩'
        ascii_digits = '0123456789'
        for arabic, ascii_digit in zip(arabic_digits, ascii_digits):
            text = text.replace(arabic, ascii_digit)
        
        return text
    
    def _fix_arabic_diacritics(self, text: str) -> str:
        """Handle Arabic diacritics appropriately."""
        # Option to remove diacritics for better processing
        # Arabic diacritics range: U+064B to U+065F
        diacritics = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065A\u065B\u065C\u065D\u065E\u065F'
        
        # Remove diacritics (optional - depends on use case)
        for diacritic in diacritics:
            text = text.replace(diacritic, '')
        
        return text
    
    def _normalize_chinese_text(self, text: str) -> str:
        """Normalize Chinese text."""
        # Convert traditional to simplified (optional)
        # This would require additional libraries like opencc
        # For now, just basic cleanup
        
        # Remove excessive whitespace (Chinese text often doesn't use spaces)
        text = re.sub(r'\s+', '', text)
        
        return text
    
    def _fix_chinese_punctuation(self, text: str) -> str:
        """Fix Chinese punctuation marks."""
        # Normalize Chinese punctuation to standard forms
        punctuation_map = {
            '，': ',',  # Chinese comma to ASCII
            '。': '.',  # Chinese period to ASCII
            '？': '?',  # Chinese question mark to ASCII
            '！': '!',  # Chinese exclamation to ASCII
            '：': ':',  # Chinese colon to ASCII
            '；': ';',  # Chinese semicolon to ASCII
        }
        
        for chinese_punct, ascii_punct in punctuation_map.items():
            text = text.replace(chinese_punct, ascii_punct)
        
        return text
    
    def _assess_and_improve_quality(self, text: str) -> Dict[str, Any]:
        """Assess text quality and apply final improvements."""
        issues_found = []
        improvements_made = []
        improved_text = text
        
        # Check for common quality issues
        if self._has_excessive_whitespace(text):
            issues_found.append('excessive_whitespace')
            improved_text = self._fix_excessive_whitespace(improved_text)
            improvements_made.append('whitespace_normalization')
        
        if self._has_broken_words(text):
            issues_found.append('broken_words')
            improved_text = self._fix_broken_words(improved_text)
            improvements_made.append('word_reconstruction')
        
        if self._has_inconsistent_line_breaks(text):
            issues_found.append('inconsistent_line_breaks')
            improved_text = self._fix_line_breaks(improved_text)
            improvements_made.append('line_break_normalization')
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(improved_text)
        
        return {
            'text': improved_text,
            'issues_found': issues_found,
            'improvements_made': improvements_made,
            'quality_score': quality_score
        }
    
    def _has_excessive_whitespace(self, text: str) -> bool:
        """Check if text has excessive whitespace."""
        return bool(re.search(r'\s{3,}', text))
    
    def _fix_excessive_whitespace(self, text: str) -> str:
        """Fix excessive whitespace in text."""
        # Normalize multiple spaces to single space
        text = re.sub(r' {2,}', ' ', text)
        # Normalize multiple line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove trailing/leading whitespace
        text = text.strip()
        
        return text
    
    def _has_broken_words(self, text: str) -> bool:
        """Check if text has broken words (common in OCR)."""
        # Look for single characters followed by spaces (potential broken words)
        broken_pattern = r'\b[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\b'
        return bool(re.search(broken_pattern, text))
    
    def _fix_broken_words(self, text: str) -> str:
        """Attempt to fix broken words."""
        # This is a simple heuristic - real word reconstruction is complex
        # Join single characters that might be broken words
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2\3', text)
        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2', text)
        
        return text
    
    def _has_inconsistent_line_breaks(self, text: str) -> bool:
        """Check if text has inconsistent line breaks."""
        # Look for lines that end mid-word (common in PDF extraction)
        return bool(re.search(r'[a-z]-\n[a-z]', text))
    
    def _fix_line_breaks(self, text: str) -> str:
        """Fix inconsistent line breaks."""
        # Fix hyphenated words split across lines
        text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)
        # Join lines that don't end with punctuation
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
        
        return text
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate a quality score for the text."""
        if not text:
            return 0.0
        
        score = 1.0
        
        # Penalize excessive whitespace
        whitespace_ratio = len(re.findall(r'\s', text)) / len(text)
        if whitespace_ratio > 0.3:
            score -= 0.2
        
        # Penalize very short or very long words (potential OCR errors)
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length < 2 or avg_word_length > 15:
                score -= 0.1
        
        # Penalize excessive punctuation
        punct_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
        if punct_ratio > 0.1:
            score -= 0.1
        
        # Bonus for proper sentence structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 5 <= avg_sentence_length <= 30:  # Reasonable sentence length
                score += 0.1
        
        return max(0.0, min(1.0, score))

# Convenience functions
def normalize_text_intelligent(text: str, 
                             language: str = None,
                             encoding: str = None) -> Dict[str, Any]:
    """
    Convenience function for intelligent text normalization.
    
    Args:
        text: Input text to normalize
        language: Known language (optional)
        encoding: Known encoding (optional)
    
    Returns:
        Normalization result dictionary
    """
    processor = IntelligentTextProcessor()
    return processor.normalize_multilingual_text(text, language, encoding)

def detect_text_language(text: str) -> str:
    """
    Quick language detection for text.
    
    Args:
        text: Input text
    
    Returns:
        Detected language code
    """
    processor = IntelligentTextProcessor()
    result = processor._detect_language(text)
    return result['language']

# Example usage
if __name__ == "__main__":
    processor = IntelligentTextProcessor()
    
    # Example with multilingual text
    sample_text = "Hola, ¿cómo estás? This is a mixed language text with encoding issues like â€™smart quotesâ€."
    
    result = processor.normalize_multilingual_text(sample_text)
    
    print(f"Original: {sample_text}")
    print(f"Normalized: {result['normalized_text']}")
    print(f"Language: {result['language']} (confidence: {result['confidence']:.2f})")
    print(f"Issues found: {result['issues_found']}")
    print(f"Improvements: {result['improvements_made']}")
    print(f"Quality score: {result['metadata']['quality_score']:.2f}")

