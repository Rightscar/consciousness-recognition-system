"""
Content Analyzer for Consciousness Recognition System

Analyzes spiritual texts to determine content type and extract meaningful content
from both dialogue and non-dialogue formats.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class ContentType(Enum):
    """Types of spiritual content."""
    DIALOGUE_HEAVY = "dialogue_heavy"      # >30% Q&A patterns
    MIXED = "mixed"                        # 10-30% Q&A patterns  
    PROSE_HEAVY = "prose_heavy"           # <10% Q&A patterns
    POETRY = "poetry"                     # Verse/poetic structure
    INSTRUCTIONAL = "instructional"       # How-to/practice guides


class ContentAnalyzer:
    """Analyzes and categorizes spiritual text content."""
    
    def __init__(self):
        """Initialize the content analyzer."""
        self.qa_patterns = [
            r'Q:\s*.*?A:\s*',
            r'Question:\s*.*?Answer:\s*',
            r'Questioner:\s*.*?(?:Maharaj|Ramana|Teacher):\s*',
            r'Visitor:\s*.*?(?:Maharaj|Ramana):\s*',
            r'\?\s*.*?(?:\n\n|\n[A-Z])',  # Question followed by answer
        ]
        
        self.poetry_indicators = [
            r'\n\s*\n\s*[A-Z].*\n\s*[A-Z]',  # Verse structure
            r'^\s*[A-Z].*\n\s*[A-Z].*\n\s*[A-Z]',  # Multiple short lines
            r'\n\s{4,}',  # Indented lines (poetry formatting)
        ]
        
        self.instructional_indicators = [
            r'step\s+\d+', r'first.*second.*third',
            r'practice:', r'exercise:', r'meditation:',
            r'how to', r'method', r'technique',
            r'instruction', r'guide', r'procedure'
        ]
        
        self.consciousness_themes = [
            # Direct pointing
            'awareness', 'consciousness', 'being', 'presence', 'witness',
            'you are', 'what you are', 'true nature', 'real self',
            
            # Non-dual concepts
            'not two', 'one consciousness', 'no separation', 'unity',
            'absolute', 'brahman', 'atman', 'self',
            
            # Present moment
            'now', 'here', 'this moment', 'immediate', 'direct',
            'right now', 'present', 'current',
            
            # Disidentification
            'not the body', 'not thoughts', 'not mind', 'not person',
            'beyond', 'prior to', 'before',
            
            # Spiritual inquiry
            'who am i', 'what am i', 'self inquiry', 'investigation',
            'look within', 'turn attention', 'observe'
        ]
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to determine content type and characteristics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Analysis results with content type and metrics
        """
        # Basic text metrics
        total_chars = len(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        total_paragraphs = len(paragraphs)
        
        if total_paragraphs == 0:
            return {
                'content_type': ContentType.PROSE_HEAVY,
                'confidence': 0.0,
                'dialogue_ratio': 0.0,
                'consciousness_density': 0.0,
                'extractable_content': False,
                'recommendations': ['Text appears to be empty or poorly formatted']
            }
        
        # Detect Q&A patterns
        qa_count = self._count_qa_patterns(text)
        dialogue_ratio = qa_count / total_paragraphs
        
        # Detect poetry structure
        poetry_score = self._detect_poetry(text)
        
        # Detect instructional content
        instructional_score = self._detect_instructional(text)
        
        # Calculate consciousness content density
        consciousness_density = self._calculate_consciousness_density(text)
        
        # Determine primary content type
        content_type, confidence = self._determine_content_type(
            dialogue_ratio, poetry_score, instructional_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            content_type, dialogue_ratio, consciousness_density
        )
        
        return {
            'content_type': content_type,
            'confidence': confidence,
            'dialogue_ratio': dialogue_ratio,
            'poetry_score': poetry_score,
            'instructional_score': instructional_score,
            'consciousness_density': consciousness_density,
            'total_paragraphs': total_paragraphs,
            'qa_patterns_found': qa_count,
            'extractable_content': consciousness_density > 0.1,
            'recommendations': recommendations,
            'processing_strategy': self._get_processing_strategy(content_type)
        }
    
    def _count_qa_patterns(self, text: str) -> int:
        """Count Q&A patterns in text."""
        count = 0
        for pattern in self.qa_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            count += len(matches)
        return count
    
    def _detect_poetry(self, text: str) -> float:
        """Detect poetic structure in text."""
        score = 0.0
        total_indicators = len(self.poetry_indicators)
        
        for pattern in self.poetry_indicators:
            if re.search(pattern, text, re.MULTILINE):
                score += 1.0
        
        return score / total_indicators if total_indicators > 0 else 0.0
    
    def _detect_instructional(self, text: str) -> float:
        """Detect instructional content."""
        score = 0.0
        text_lower = text.lower()
        
        for indicator in self.instructional_indicators:
            if re.search(indicator, text_lower):
                score += 1.0
        
        return min(score / 5.0, 1.0)  # Normalize to 0-1
    
    def _calculate_consciousness_density(self, text: str) -> float:
        """Calculate density of consciousness-related content."""
        text_lower = text.lower()
        total_words = len(text_lower.split())
        
        if total_words == 0:
            return 0.0
        
        consciousness_words = 0
        for theme in self.consciousness_themes:
            consciousness_words += len(re.findall(r'\b' + re.escape(theme) + r'\b', text_lower))
        
        return min(consciousness_words / total_words * 10, 1.0)  # Scale and cap at 1.0
    
    def _determine_content_type(
        self, 
        dialogue_ratio: float, 
        poetry_score: float, 
        instructional_score: float
    ) -> Tuple[ContentType, float]:
        """Determine primary content type with confidence."""
        
        # Poetry detection
        if poetry_score > 0.3:
            return ContentType.POETRY, poetry_score
        
        # Instructional detection
        if instructional_score > 0.4:
            return ContentType.INSTRUCTIONAL, instructional_score
        
        # Dialogue-based detection
        if dialogue_ratio > 0.3:
            return ContentType.DIALOGUE_HEAVY, dialogue_ratio
        elif dialogue_ratio > 0.1:
            return ContentType.MIXED, dialogue_ratio
        else:
            return ContentType.PROSE_HEAVY, 1.0 - dialogue_ratio
    
    def _generate_recommendations(
        self, 
        content_type: ContentType, 
        dialogue_ratio: float, 
        consciousness_density: float
    ) -> List[str]:
        """Generate processing recommendations."""
        recommendations = []
        
        if content_type == ContentType.DIALOGUE_HEAVY:
            recommendations.append("Use standard dialogue extraction")
            if dialogue_ratio > 0.7:
                recommendations.append("High dialogue density - expect good Q&A extraction")
        
        elif content_type == ContentType.MIXED:
            recommendations.append("Use hybrid processing (dialogues + passages)")
            recommendations.append("Extract both Q&A patterns and meaningful passages")
        
        elif content_type == ContentType.PROSE_HEAVY:
            recommendations.append("Use passage extraction mode")
            recommendations.append("Create synthetic Q&A from meaningful passages")
            if consciousness_density > 0.3:
                recommendations.append("High consciousness content - good for training data")
        
        elif content_type == ContentType.POETRY:
            recommendations.append("Use verse-aware passage extraction")
            recommendations.append("Preserve poetic structure in extracted content")
        
        elif content_type == ContentType.INSTRUCTIONAL:
            recommendations.append("Extract practice instructions and guidance")
            recommendations.append("Create Q&A from instructional content")
        
        # General recommendations
        if consciousness_density < 0.1:
            recommendations.append("⚠️ Low consciousness content - may not be suitable for training")
        elif consciousness_density > 0.5:
            recommendations.append("✅ High consciousness content - excellent for training")
        
        return recommendations
    
    def _get_processing_strategy(self, content_type: ContentType) -> Dict[str, Any]:
        """Get recommended processing strategy for content type."""
        strategies = {
            ContentType.DIALOGUE_HEAVY: {
                'primary_method': 'dialogue_extraction',
                'secondary_method': None,
                'chunk_size': 'standard',
                'post_processing': 'score_dialogues'
            },
            ContentType.MIXED: {
                'primary_method': 'dialogue_extraction',
                'secondary_method': 'passage_extraction',
                'chunk_size': 'standard',
                'post_processing': 'merge_and_score'
            },
            ContentType.PROSE_HEAVY: {
                'primary_method': 'passage_extraction',
                'secondary_method': 'synthetic_qa_generation',
                'chunk_size': 'large',
                'post_processing': 'score_passages'
            },
            ContentType.POETRY: {
                'primary_method': 'verse_extraction',
                'secondary_method': 'passage_extraction',
                'chunk_size': 'preserve_structure',
                'post_processing': 'score_verses'
            },
            ContentType.INSTRUCTIONAL: {
                'primary_method': 'instruction_extraction',
                'secondary_method': 'synthetic_qa_generation',
                'chunk_size': 'by_section',
                'post_processing': 'score_instructions'
            }
        }
        
        return strategies.get(content_type, strategies[ContentType.PROSE_HEAVY])


class PassageExtractor:
    """Extracts meaningful passages from non-dialogue spiritual texts."""
    
    def __init__(self):
        """Initialize the passage extractor."""
        self.min_passage_length = 100  # Minimum characters
        self.max_passage_length = 1000  # Maximum characters
        self.consciousness_themes = ContentAnalyzer().consciousness_themes
    
    def extract_passages(
        self, 
        text: str, 
        content_type: ContentType,
        min_consciousness_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Extract meaningful passages from text.
        
        Args:
            text: Input text
            content_type: Type of content being processed
            min_consciousness_score: Minimum score for passage inclusion
            
        Returns:
            List of extracted passages with metadata
        """
        if content_type == ContentType.POETRY:
            return self._extract_verses(text, min_consciousness_score)
        elif content_type == ContentType.INSTRUCTIONAL:
            return self._extract_instructions(text, min_consciousness_score)
        else:
            return self._extract_prose_passages(text, min_consciousness_score)
    
    def _extract_prose_passages(self, text: str, min_score: float) -> List[Dict[str, Any]]:
        """Extract passages from prose text."""
        passages = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Group paragraphs into passages
        current_passage = ""
        current_paragraphs = []
        
        for para in paragraphs:
            # Check if adding this paragraph would exceed max length
            if len(current_passage + para) > self.max_passage_length and current_passage:
                # Process current passage
                if len(current_passage) >= self.min_passage_length:
                    score = self._score_consciousness_content(current_passage)
                    if score >= min_score:
                        passages.append({
                            'text': current_passage.strip(),
                            'consciousness_score': score,
                            'paragraph_count': len(current_paragraphs),
                            'type': 'prose_passage'
                        })
                
                # Start new passage
                current_passage = para + "\n\n"
                current_paragraphs = [para]
            else:
                current_passage += para + "\n\n"
                current_paragraphs.append(para)
        
        # Process final passage
        if len(current_passage) >= self.min_passage_length:
            score = self._score_consciousness_content(current_passage)
            if score >= min_score:
                passages.append({
                    'text': current_passage.strip(),
                    'consciousness_score': score,
                    'paragraph_count': len(current_paragraphs),
                    'type': 'prose_passage'
                })
        
        return passages
    
    def _extract_verses(self, text: str, min_score: float) -> List[Dict[str, Any]]:
        """Extract verses from poetic text."""
        passages = []
        
        # Split by double newlines (verse breaks)
        verses = [v.strip() for v in text.split('\n\n') if v.strip()]
        
        for verse in verses:
            if len(verse) >= self.min_passage_length:
                score = self._score_consciousness_content(verse)
                if score >= min_score:
                    passages.append({
                        'text': verse,
                        'consciousness_score': score,
                        'line_count': len(verse.split('\n')),
                        'type': 'verse'
                    })
        
        return passages
    
    def _extract_instructions(self, text: str, min_score: float) -> List[Dict[str, Any]]:
        """Extract instructional content."""
        passages = []
        
        # Look for numbered steps or sections
        sections = re.split(r'\n(?=\d+\.|\w+:)', text)
        
        for section in sections:
            section = section.strip()
            if len(section) >= self.min_passage_length:
                score = self._score_consciousness_content(section)
                if score >= min_score:
                    passages.append({
                        'text': section,
                        'consciousness_score': score,
                        'type': 'instruction'
                    })
        
        return passages
    
    def _score_consciousness_content(self, text: str) -> float:
        """Score text for consciousness-related content."""
        text_lower = text.lower()
        total_words = len(text_lower.split())
        
        if total_words == 0:
            return 0.0
        
        consciousness_words = 0
        for theme in self.consciousness_themes:
            consciousness_words += len(re.findall(r'\b' + re.escape(theme) + r'\b', text_lower))
        
        # Base score from keyword density
        base_score = min(consciousness_words / total_words * 5, 0.8)
        
        # Bonus for direct pointing language
        direct_pointing_bonus = 0.0
        direct_phrases = ['you are', 'what you are', 'remain as', 'be as you are', 'just be']
        for phrase in direct_phrases:
            if phrase in text_lower:
                direct_pointing_bonus += 0.1
        
        return min(base_score + direct_pointing_bonus, 1.0)


class SyntheticQAGenerator:
    """Generates synthetic Q&A pairs from spiritual passages."""
    
    def __init__(self):
        """Initialize the synthetic Q&A generator."""
        self.question_templates = [
            "What is the nature of {concept}?",
            "How can one understand {concept}?",
            "What does it mean to {action}?",
            "How does one {action}?",
            "What is {concept}?",
            "Can you explain {concept}?",
            "What is the significance of {concept}?",
            "How should one approach {concept}?"
        ]
        
        self.concept_extractors = [
            r'\b(awareness|consciousness|being|presence)\b',
            r'\b(self|true nature|real self)\b',
            r'\b(meditation|inquiry|investigation)\b',
            r'\b(mind|thoughts|ego)\b',
            r'\b(reality|truth|absolute)\b'
        ]
    
    def generate_qa_from_passage(self, passage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate synthetic Q&A pairs from a passage.
        
        Args:
            passage: Passage dictionary with text and metadata
            
        Returns:
            List of synthetic Q&A pairs
        """
        text = passage['text']
        qa_pairs = []
        
        # Extract key concepts from the passage
        concepts = self._extract_concepts(text)
        
        # Generate questions for each concept
        for concept in concepts[:3]:  # Limit to 3 questions per passage
            question = self._generate_question(concept, text)
            if question:
                qa_pairs.append({
                    'question': question,
                    'answer': text,
                    'source': 'synthetic',
                    'original_type': passage.get('type', 'passage'),
                    'consciousness_score': passage.get('consciousness_score', 0.5),
                    'generation_method': 'concept_based'
                })
        
        # Generate a general question if no concepts found
        if not qa_pairs:
            general_question = self._generate_general_question(text)
            if general_question:
                qa_pairs.append({
                    'question': general_question,
                    'answer': text,
                    'source': 'synthetic',
                    'original_type': passage.get('type', 'passage'),
                    'consciousness_score': passage.get('consciousness_score', 0.5),
                    'generation_method': 'general'
                })
        
        return qa_pairs
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key spiritual concepts from text."""
        concepts = []
        text_lower = text.lower()
        
        for pattern in self.concept_extractors:
            matches = re.findall(pattern, text_lower)
            concepts.extend(matches)
        
        # Remove duplicates and return
        return list(set(concepts))
    
    def _generate_question(self, concept: str, text: str) -> Optional[str]:
        """Generate a question based on a concept and context."""
        import random
        
        # Choose appropriate template based on concept
        if concept in ['awareness', 'consciousness', 'being']:
            templates = [
                f"What is the nature of {concept}?",
                f"How can one recognize {concept}?",
                f"What does it mean to be {concept}?"
            ]
        elif concept in ['meditation', 'inquiry']:
            templates = [
                f"How should one practice {concept}?",
                f"What is the purpose of {concept}?",
                f"How does {concept} lead to understanding?"
            ]
        else:
            templates = [
                f"What is {concept}?",
                f"How can one understand {concept}?",
                f"What is the significance of {concept}?"
            ]
        
        return random.choice(templates) if templates else None
    
    def _generate_general_question(self, text: str) -> Optional[str]:
        """Generate a general question for passages without clear concepts."""
        general_questions = [
            "What is the essence of this teaching?",
            "How can one apply this understanding?",
            "What is the deeper meaning here?",
            "How does this relate to spiritual understanding?",
            "What insight is being shared?",
            "How can this guide one's practice?",
            "What is the practical wisdom in this teaching?"
        ]
        
        import random
        return random.choice(general_questions)

