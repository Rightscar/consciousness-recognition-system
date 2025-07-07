"""
Quality Threshold Handler Module
==============================

Implements comprehensive quality scoring and threshold handling for AI-enhanced content.
Automatically flags low-quality outputs for manual review using multiple quality metrics.

Features:
- Coherence scoring using semantic embeddings
- Length ratio analysis with intelligent thresholds
- Content quality metrics (readability, sentiment, vocabulary)
- Automated flagging for manual review
- Configurable quality thresholds
- Comprehensive quality visualization
- Quality trend analysis and monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import json
import re
from datetime import datetime, timedelta
import math

# Core dependencies
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Sentence transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Text analysis libraries
try:
    import textstat
    from textblob import TextBlob
    from nltk.tokenize import sent_tokenize, word_tokenize
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    TEXT_ANALYSIS_AVAILABLE = True
except ImportError:
    TEXT_ANALYSIS_AVAILABLE = False

@dataclass
class QualityScore:
    """Structured quality score result"""
    overall_score: float
    component_scores: Dict[str, float]
    quality_status: str
    needs_review: bool
    review_reasons: List[str]
    detailed_analysis: Dict[str, Any]
    thresholds_used: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

class CoherenceAnalyzer:
    """Analyzes semantic coherence between original and enhanced text"""
    
    def __init__(self):
        self.model = None
        self.tfidf_vectorizer = None
        self.is_loaded = False
        self.fallback_method = "tfidf"  # Fallback if sentence transformers unavailable
    
    def lazy_load_model(self) -> bool:
        """Load semantic similarity model on demand"""
        if self.is_loaded:
            return True
        
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use sentence transformers for best results
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.fallback_method = "sentence_transformers"
                logging.info("Loaded sentence transformers model for coherence analysis")
            elif SKLEARN_AVAILABLE:
                # Fallback to TF-IDF similarity
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.fallback_method = "tfidf"
                logging.info("Using TF-IDF fallback for coherence analysis")
            else:
                # Basic word overlap fallback
                self.fallback_method = "word_overlap"
                logging.warning("Using basic word overlap for coherence analysis")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logging.error(f"Failed to load coherence analysis model: {e}")
            self.fallback_method = "word_overlap"
            self.is_loaded = True
            return False
    
    def calculate_coherence(self, original: str, enhanced: str) -> float:
        """
        Calculate semantic coherence between original and enhanced text
        
        Returns:
            Float between 0.0 and 1.0 representing coherence score
        """
        if not original.strip() or not enhanced.strip():
            return 0.0
        
        self.lazy_load_model()
        
        try:
            if self.fallback_method == "sentence_transformers" and self.model:
                return self._calculate_sentence_transformer_similarity(original, enhanced)
            elif self.fallback_method == "tfidf" and SKLEARN_AVAILABLE:
                return self._calculate_tfidf_similarity(original, enhanced)
            else:
                return self._calculate_word_overlap_similarity(original, enhanced)
                
        except Exception as e:
            logging.error(f"Coherence calculation failed: {e}")
            return 0.5  # Neutral score on failure
    
    def _calculate_sentence_transformer_similarity(self, original: str, enhanced: str) -> float:
        """Calculate similarity using sentence transformers"""
        try:
            # Generate embeddings
            embeddings = self.model.encode([original, enhanced])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Ensure result is in valid range
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logging.error(f"Sentence transformer similarity failed: {e}")
            return self._calculate_word_overlap_similarity(original, enhanced)
    
    def _calculate_tfidf_similarity(self, original: str, enhanced: str) -> float:
        """Calculate similarity using TF-IDF vectors"""
        try:
            # Fit TF-IDF on both texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([original, enhanced])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logging.error(f"TF-IDF similarity failed: {e}")
            return self._calculate_word_overlap_similarity(original, enhanced)
    
    def _calculate_word_overlap_similarity(self, original: str, enhanced: str) -> float:
        """Basic word overlap similarity as fallback"""
        try:
            # Tokenize and normalize
            original_words = set(word.lower() for word in re.findall(r'\b\w+\b', original))
            enhanced_words = set(word.lower() for word in re.findall(r'\b\w+\b', enhanced))
            
            if not original_words or not enhanced_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(original_words & enhanced_words)
            union = len(original_words | enhanced_words)
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # Adjust for length differences (penalize extreme changes)
            length_ratio = min(len(enhanced_words), len(original_words)) / max(len(enhanced_words), len(original_words))
            
            # Combine Jaccard similarity with length penalty
            final_similarity = jaccard_similarity * (0.7 + 0.3 * length_ratio)
            
            return max(0.0, min(1.0, final_similarity))
            
        except Exception as e:
            logging.error(f"Word overlap similarity failed: {e}")
            return 0.5

class LengthAnalyzer:
    """Analyzes length changes and their implications for quality"""
    
    def __init__(self):
        # Configurable thresholds
        self.optimal_ratio_range = (0.8, 1.3)    # Ideal enhancement range
        self.acceptable_ratio_range = (0.6, 1.8)  # Acceptable range
        self.warning_ratio_range = (0.4, 2.2)     # Warning range
        self.critical_threshold = 2.5              # Critical threshold
    
    def analyze_length_ratio(self, original: str, enhanced: str) -> Dict[str, Any]:
        """
        Analyze length changes and their quality implications
        
        Returns:
            Dictionary with ratio analysis and quality assessment
        """
        original_length = len(original.strip())
        enhanced_length = len(enhanced.strip())
        
        if original_length == 0:
            return {
                "ratio": 0,
                "status": "ERROR",
                "message": "Original text is empty",
                "original_length": 0,
                "enhanced_length": enhanced_length,
                "length_change": enhanced_length,
                "quality_score": 0.0
            }
        
        ratio = enhanced_length / original_length
        
        # Determine status and quality implications
        status, message, quality_score = self._assess_length_ratio(ratio)
        
        # Additional metrics
        word_count_original = len(original.split())
        word_count_enhanced = len(enhanced.split())
        word_ratio = word_count_enhanced / max(word_count_original, 1)
        
        return {
            "ratio": ratio,
            "status": status,
            "message": message,
            "quality_score": quality_score,
            "original_length": original_length,
            "enhanced_length": enhanced_length,
            "length_change": enhanced_length - original_length,
            "length_change_pct": ((enhanced_length - original_length) / original_length) * 100,
            "word_count_original": word_count_original,
            "word_count_enhanced": word_count_enhanced,
            "word_ratio": word_ratio,
            "thresholds": {
                "optimal_range": self.optimal_ratio_range,
                "acceptable_range": self.acceptable_ratio_range,
                "warning_range": self.warning_ratio_range,
                "critical_threshold": self.critical_threshold
            }
        }
    
    def _assess_length_ratio(self, ratio: float) -> Tuple[str, str, float]:
        """Assess length ratio and return status, message, and quality score"""
        
        if ratio < 0.2:
            return "CRITICAL", "Enhanced text severely truncated - possible API failure", 0.1
        elif ratio < 0.4:
            return "CRITICAL", "Enhanced text too short - significant content loss", 0.2
        elif ratio < self.acceptable_ratio_range[0]:
            return "WARNING", "Enhanced text shorter than expected - review for content loss", 0.4
        elif ratio > self.critical_threshold:
            return "CRITICAL", "Enhanced text excessively long - possible hallucination", 0.1
        elif ratio > self.acceptable_ratio_range[1]:
            return "WARNING", "Enhanced text significantly longer - review for relevance", 0.5
        elif self.optimal_ratio_range[0] <= ratio <= self.optimal_ratio_range[1]:
            return "OPTIMAL", "Length change within optimal range", 1.0
        else:
            return "ACCEPTABLE", "Length change acceptable but monitor", 0.8

class ContentQualityAnalyzer:
    """Analyzes various content quality metrics"""
    
    def __init__(self):
        self.text_analysis_available = TEXT_ANALYSIS_AVAILABLE
    
    def analyze_content_quality(self, original: str, enhanced: str) -> Dict[str, Any]:
        """
        Comprehensive content quality analysis
        
        Returns:
            Dictionary with various quality metrics
        """
        analysis = {
            "readability": self._analyze_readability(original, enhanced),
            "sentiment": self._analyze_sentiment(original, enhanced),
            "vocabulary": self._analyze_vocabulary(original, enhanced),
            "structure": self._analyze_structure(original, enhanced),
            "complexity": self._analyze_complexity(original, enhanced)
        }
        
        return analysis
    
    def _analyze_readability(self, original: str, enhanced: str) -> Dict[str, Any]:
        """Analyze readability changes"""
        try:
            if self.text_analysis_available:
                original_score = textstat.flesch_reading_ease(original)
                enhanced_score = textstat.flesch_reading_ease(enhanced)
                change = enhanced_score - original_score
                
                # Assess readability change
                if change >= 5:
                    status = "IMPROVED"
                    quality_score = 1.0
                elif change >= 0:
                    status = "MAINTAINED"
                    quality_score = 0.9
                elif change >= -5:
                    status = "SLIGHT_DECLINE"
                    quality_score = 0.7
                elif change >= -10:
                    status = "MODERATE_DECLINE"
                    quality_score = 0.4
                else:
                    status = "SIGNIFICANT_DECLINE"
                    quality_score = 0.1
                
                return {
                    "original_score": original_score,
                    "enhanced_score": enhanced_score,
                    "change": change,
                    "status": status,
                    "quality_score": quality_score,
                    "available": True
                }
            else:
                # Fallback: simple sentence length analysis
                orig_sentences = len(re.split(r'[.!?]+', original))
                enh_sentences = len(re.split(r'[.!?]+', enhanced))
                
                orig_avg_length = len(original.split()) / max(orig_sentences, 1)
                enh_avg_length = len(enhanced.split()) / max(enh_sentences, 1)
                
                change = enh_avg_length - orig_avg_length
                
                return {
                    "original_avg_sentence_length": orig_avg_length,
                    "enhanced_avg_sentence_length": enh_avg_length,
                    "change": change,
                    "status": "ESTIMATED",
                    "quality_score": 0.7,  # Neutral score
                    "available": False
                }
                
        except Exception as e:
            logging.error(f"Readability analysis failed: {e}")
            return {"available": False, "error": str(e), "quality_score": 0.5}
    
    def _analyze_sentiment(self, original: str, enhanced: str) -> Dict[str, Any]:
        """Analyze sentiment preservation"""
        try:
            if self.text_analysis_available:
                original_blob = TextBlob(original)
                enhanced_blob = TextBlob(enhanced)
                
                original_sentiment = original_blob.sentiment.polarity
                enhanced_sentiment = enhanced_blob.sentiment.polarity
                
                drift = abs(original_sentiment - enhanced_sentiment)
                
                # Assess sentiment preservation
                if drift <= 0.1:
                    status = "EXCELLENT"
                    quality_score = 1.0
                elif drift <= 0.2:
                    status = "GOOD"
                    quality_score = 0.8
                elif drift <= 0.3:
                    status = "ACCEPTABLE"
                    quality_score = 0.6
                elif drift <= 0.5:
                    status = "CONCERNING"
                    quality_score = 0.3
                else:
                    status = "SIGNIFICANT_DRIFT"
                    quality_score = 0.1
                
                return {
                    "original_sentiment": original_sentiment,
                    "enhanced_sentiment": enhanced_sentiment,
                    "drift": drift,
                    "status": status,
                    "quality_score": quality_score,
                    "available": True
                }
            else:
                # Fallback: basic sentiment word counting
                positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'like']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible']
                
                def simple_sentiment(text):
                    words = text.lower().split()
                    pos_count = sum(1 for word in words if word in positive_words)
                    neg_count = sum(1 for word in words if word in negative_words)
                    return (pos_count - neg_count) / max(len(words), 1)
                
                orig_sentiment = simple_sentiment(original)
                enh_sentiment = simple_sentiment(enhanced)
                drift = abs(orig_sentiment - enh_sentiment)
                
                return {
                    "original_sentiment": orig_sentiment,
                    "enhanced_sentiment": enh_sentiment,
                    "drift": drift,
                    "status": "ESTIMATED",
                    "quality_score": 0.7,
                    "available": False
                }
                
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {e}")
            return {"available": False, "error": str(e), "quality_score": 0.5}
    
    def _analyze_vocabulary(self, original: str, enhanced: str) -> Dict[str, Any]:
        """Analyze vocabulary overlap and richness"""
        try:
            # Tokenize and normalize
            original_words = set(word.lower() for word in re.findall(r'\b\w+\b', original))
            enhanced_words = set(word.lower() for word in re.findall(r'\b\w+\b', enhanced))
            
            if not original_words:
                return {"available": False, "error": "No words in original text", "quality_score": 0.0}
            
            # Calculate overlap metrics
            intersection = original_words & enhanced_words
            union = original_words | enhanced_words
            
            overlap_ratio = len(intersection) / len(original_words)
            jaccard_similarity = len(intersection) / len(union) if union else 0
            
            # Vocabulary richness (unique words / total words)
            original_total_words = len(re.findall(r'\b\w+\b', original.lower()))
            enhanced_total_words = len(re.findall(r'\b\w+\b', enhanced.lower()))
            
            original_richness = len(original_words) / max(original_total_words, 1)
            enhanced_richness = len(enhanced_words) / max(enhanced_total_words, 1)
            
            # Assess vocabulary preservation
            if overlap_ratio >= 0.7:
                status = "EXCELLENT"
                quality_score = 1.0
            elif overlap_ratio >= 0.5:
                status = "GOOD"
                quality_score = 0.8
            elif overlap_ratio >= 0.3:
                status = "ACCEPTABLE"
                quality_score = 0.6
            elif overlap_ratio >= 0.2:
                status = "CONCERNING"
                quality_score = 0.3
            else:
                status = "POOR_OVERLAP"
                quality_score = 0.1
            
            return {
                "overlap_ratio": overlap_ratio,
                "jaccard_similarity": jaccard_similarity,
                "original_unique_words": len(original_words),
                "enhanced_unique_words": len(enhanced_words),
                "original_richness": original_richness,
                "enhanced_richness": enhanced_richness,
                "status": status,
                "quality_score": quality_score,
                "available": True
            }
            
        except Exception as e:
            logging.error(f"Vocabulary analysis failed: {e}")
            return {"available": False, "error": str(e), "quality_score": 0.5}
    
    def _analyze_structure(self, original: str, enhanced: str) -> Dict[str, Any]:
        """Analyze structural preservation"""
        try:
            # Sentence analysis
            if self.text_analysis_available:
                original_sentences = sent_tokenize(original)
                enhanced_sentences = sent_tokenize(enhanced)
            else:
                original_sentences = re.split(r'[.!?]+', original)
                enhanced_sentences = re.split(r'[.!?]+', enhanced)
            
            original_sentence_count = len([s for s in original_sentences if s.strip()])
            enhanced_sentence_count = len([s for s in enhanced_sentences if s.strip()])
            
            sentence_ratio = enhanced_sentence_count / max(original_sentence_count, 1)
            
            # Paragraph analysis
            original_paragraphs = len([p for p in original.split('\n\n') if p.strip()])
            enhanced_paragraphs = len([p for p in enhanced.split('\n\n') if p.strip()])
            
            paragraph_ratio = enhanced_paragraphs / max(original_paragraphs, 1)
            
            # Assess structural preservation
            if 0.9 <= sentence_ratio <= 1.1:
                status = "EXCELLENT"
                quality_score = 1.0
            elif 0.8 <= sentence_ratio <= 1.3:
                status = "GOOD"
                quality_score = 0.8
            elif 0.7 <= sentence_ratio <= 1.5:
                status = "ACCEPTABLE"
                quality_score = 0.6
            elif 0.5 <= sentence_ratio <= 2.0:
                status = "CONCERNING"
                quality_score = 0.3
            else:
                status = "POOR_STRUCTURE"
                quality_score = 0.1
            
            return {
                "sentence_ratio": sentence_ratio,
                "paragraph_ratio": paragraph_ratio,
                "original_sentences": original_sentence_count,
                "enhanced_sentences": enhanced_sentence_count,
                "original_paragraphs": original_paragraphs,
                "enhanced_paragraphs": enhanced_paragraphs,
                "status": status,
                "quality_score": quality_score,
                "available": True
            }
            
        except Exception as e:
            logging.error(f"Structure analysis failed: {e}")
            return {"available": False, "error": str(e), "quality_score": 0.5}
    
    def _analyze_complexity(self, original: str, enhanced: str) -> Dict[str, Any]:
        """Analyze text complexity changes"""
        try:
            # Word length analysis
            original_words = re.findall(r'\b\w+\b', original)
            enhanced_words = re.findall(r'\b\w+\b', enhanced)
            
            if not original_words:
                return {"available": False, "error": "No words found", "quality_score": 0.0}
            
            original_avg_word_length = np.mean([len(word) for word in original_words])
            enhanced_avg_word_length = np.mean([len(word) for word in enhanced_words])
            
            # Sentence length analysis
            original_sentences = re.split(r'[.!?]+', original)
            enhanced_sentences = re.split(r'[.!?]+', enhanced)
            
            original_sentences = [s.strip() for s in original_sentences if s.strip()]
            enhanced_sentences = [s.strip() for s in enhanced_sentences if s.strip()]
            
            if original_sentences:
                original_avg_sentence_length = np.mean([len(s.split()) for s in original_sentences])
            else:
                original_avg_sentence_length = 0
            
            if enhanced_sentences:
                enhanced_avg_sentence_length = np.mean([len(s.split()) for s in enhanced_sentences])
            else:
                enhanced_avg_sentence_length = 0
            
            # Calculate complexity change
            word_length_change = enhanced_avg_word_length - original_avg_word_length
            sentence_length_change = enhanced_avg_sentence_length - original_avg_sentence_length
            
            return {
                "original_avg_word_length": original_avg_word_length,
                "enhanced_avg_word_length": enhanced_avg_word_length,
                "word_length_change": word_length_change,
                "original_avg_sentence_length": original_avg_sentence_length,
                "enhanced_avg_sentence_length": enhanced_avg_sentence_length,
                "sentence_length_change": sentence_length_change,
                "available": True,
                "quality_score": 0.8  # Neutral assessment for complexity
            }
            
        except Exception as e:
            logging.error(f"Complexity analysis failed: {e}")
            return {"available": False, "error": str(e), "quality_score": 0.5}

class ComprehensiveQualityScorer:
    """Main quality scoring system combining all analysis components"""
    
    def __init__(self, custom_thresholds: Optional[Dict[str, float]] = None):
        self.coherence_analyzer = CoherenceAnalyzer()
        self.length_analyzer = LengthAnalyzer()
        self.content_analyzer = ContentQualityAnalyzer()
        
        # Default quality thresholds (configurable)
        self.thresholds = {
            "coherence_min": 0.75,
            "length_ratio_max": 1.8,
            "length_ratio_min": 0.6,
            "sentiment_drift_max": 0.3,
            "vocabulary_overlap_min": 0.4,
            "overall_quality_min": 0.7,
            "readability_decline_max": 10.0,
            "structure_ratio_min": 0.7,
            "structure_ratio_max": 1.5
        }
        
        # Update with custom thresholds if provided
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
        
        # Component weights for overall score
        self.weights = {
            "coherence": 0.30,      # Most important - semantic preservation
            "length": 0.20,         # Important - reasonable enhancement
            "readability": 0.15,    # Moderate - should improve or maintain
            "sentiment": 0.15,      # Moderate - tone preservation
            "vocabulary": 0.10,     # Lower - some change expected
            "structure": 0.10       # Lower - structure can change
        }
    
    def calculate_quality_score(self, original: str, enhanced: str, 
                              content_type: Optional[str] = None) -> QualityScore:
        """
        Calculate comprehensive quality score with detailed breakdown
        
        Args:
            original: Original text
            enhanced: AI-enhanced text
            content_type: Optional content type for adaptive thresholds
            
        Returns:
            QualityScore object with comprehensive analysis
        """
        try:
            # Adapt thresholds based on content type
            adapted_thresholds = self._adapt_thresholds_for_content_type(content_type)
            
            # Calculate individual component scores
            coherence_score = self.coherence_analyzer.calculate_coherence(original, enhanced)
            length_analysis = self.length_analyzer.analyze_length_ratio(original, enhanced)
            content_analysis = self.content_analyzer.analyze_content_quality(original, enhanced)
            
            # Convert component analyses to 0-1 scores
            component_scores = {
                "coherence": coherence_score,
                "length": length_analysis["quality_score"],
                "readability": content_analysis["readability"].get("quality_score", 0.5),
                "sentiment": content_analysis["sentiment"].get("quality_score", 0.5),
                "vocabulary": content_analysis["vocabulary"].get("quality_score", 0.5),
                "structure": content_analysis["structure"].get("quality_score", 0.5)
            }
            
            # Calculate weighted overall score
            overall_score = sum(
                component_scores[component] * self.weights[component] 
                for component in component_scores
            )
            
            # Determine quality status
            quality_status = self._determine_quality_status(overall_score)
            
            # Check if manual review is needed
            needs_review, review_reasons = self._should_flag_for_review(
                overall_score, component_scores, length_analysis, content_analysis, adapted_thresholds
            )
            
            # Compile detailed analysis
            detailed_analysis = {
                "coherence": coherence_score,
                "length_analysis": length_analysis,
                "content_analysis": content_analysis,
                "component_weights": self.weights,
                "adapted_thresholds": adapted_thresholds
            }
            
            return QualityScore(
                overall_score=overall_score,
                component_scores=component_scores,
                quality_status=quality_status,
                needs_review=needs_review,
                review_reasons=review_reasons,
                detailed_analysis=detailed_analysis,
                thresholds_used=adapted_thresholds
            )
            
        except Exception as e:
            logging.error(f"Quality score calculation failed: {e}")
            # Return minimal quality score on failure
            return QualityScore(
                overall_score=0.0,
                component_scores={},
                quality_status="ERROR",
                needs_review=True,
                review_reasons=[f"Quality calculation failed: {str(e)}"],
                detailed_analysis={"error": str(e)},
                thresholds_used=self.thresholds
            )
    
    def _adapt_thresholds_for_content_type(self, content_type: Optional[str]) -> Dict[str, float]:
        """Adapt quality thresholds based on content type"""
        adapted = self.thresholds.copy()
        
        if content_type == "qa_pair":
            # Q&A should maintain higher coherence
            adapted["coherence_min"] = 0.8
            adapted["vocabulary_overlap_min"] = 0.5
        elif content_type == "dialogue":
            # Dialogue can expand more naturally
            adapted["length_ratio_max"] = 2.0
            adapted["structure_ratio_max"] = 2.0
        elif content_type == "monologue":
            # Monologue enhancement can be more creative
            adapted["coherence_min"] = 0.7
            adapted["vocabulary_overlap_min"] = 0.3
        elif content_type == "mixed":
            # Mixed content needs balanced approach
            adapted["coherence_min"] = 0.72
            adapted["length_ratio_max"] = 1.9
        
        return adapted
    
    def _determine_quality_status(self, overall_score: float) -> str:
        """Determine overall quality status based on score"""
        if overall_score >= 0.85:
            return "EXCELLENT"
        elif overall_score >= 0.75:
            return "GOOD"
        elif overall_score >= 0.65:
            return "ACCEPTABLE"
        elif overall_score >= 0.5:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _should_flag_for_review(self, overall_score: float, component_scores: Dict[str, float],
                               length_analysis: Dict[str, Any], content_analysis: Dict[str, Any],
                               thresholds: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Determine if content should be flagged for manual review"""
        
        reasons = []
        
        # Check your original criteria
        if component_scores["coherence"] < thresholds["coherence_min"]:
            reasons.append(f"Low coherence score ({component_scores['coherence']:.2f} < {thresholds['coherence_min']})")
        
        if length_analysis["ratio"] > thresholds["length_ratio_max"]:
            reasons.append(f"Excessive length increase ({length_analysis['ratio']:.2f}x > {thresholds['length_ratio_max']}x)")
        
        # Additional quality checks
        if overall_score < thresholds["overall_quality_min"]:
            reasons.append(f"Overall quality below threshold ({overall_score:.2f} < {thresholds['overall_quality_min']})")
        
        if length_analysis["ratio"] < thresholds["length_ratio_min"]:
            reasons.append(f"Excessive length decrease ({length_analysis['ratio']:.2f}x < {thresholds['length_ratio_min']}x)")
        
        if length_analysis["status"] == "CRITICAL":
            reasons.append(f"Critical length issue: {length_analysis['message']}")
        
        # Content-specific checks
        sentiment_analysis = content_analysis.get("sentiment", {})
        if sentiment_analysis.get("drift", 0) > thresholds["sentiment_drift_max"]:
            reasons.append(f"Significant sentiment drift ({sentiment_analysis['drift']:.2f} > {thresholds['sentiment_drift_max']})")
        
        vocabulary_analysis = content_analysis.get("vocabulary", {})
        if vocabulary_analysis.get("overlap_ratio", 1) < thresholds["vocabulary_overlap_min"]:
            reasons.append(f"Low vocabulary overlap ({vocabulary_analysis['overlap_ratio']:.2f} < {thresholds['vocabulary_overlap_min']})")
        
        structure_analysis = content_analysis.get("structure", {})
        sentence_ratio = structure_analysis.get("sentence_ratio", 1)
        if (sentence_ratio < thresholds["structure_ratio_min"] or 
            sentence_ratio > thresholds["structure_ratio_max"]):
            reasons.append(f"Significant structural change (sentence ratio: {sentence_ratio:.2f})")
        
        # Critical component failures
        critical_failures = []
        if component_scores["coherence"] < 0.6:
            critical_failures.append("coherence")
        if component_scores["sentiment"] < 0.4:
            critical_failures.append("sentiment preservation")
        if component_scores["length"] < 0.3:
            critical_failures.append("length appropriateness")
        
        if critical_failures:
            reasons.append(f"Critical component failures: {', '.join(critical_failures)}")
        
        needs_review = len(reasons) > 0
        
        return needs_review, reasons
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update quality thresholds"""
        self.thresholds.update(new_thresholds)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update component weights"""
        # Normalize weights to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.weights.update({k: v/total_weight for k, v in new_weights.items()})

class QualityThresholdHandler:
    """Main handler for quality threshold management and UI integration"""
    
    def __init__(self, custom_thresholds: Optional[Dict[str, float]] = None):
        self.quality_scorer = ComprehensiveQualityScorer(custom_thresholds)
        self.quality_history = []
        self.review_queue = []
    
    def evaluate_content_quality(self, original: str, enhanced: str, 
                                content_type: Optional[str] = None) -> QualityScore:
        """
        Evaluate content quality and return comprehensive score
        
        Args:
            original: Original text
            enhanced: AI-enhanced text
            content_type: Optional content type for adaptive scoring
            
        Returns:
            QualityScore object with full analysis
        """
        quality_score = self.quality_scorer.calculate_quality_score(
            original, enhanced, content_type
        )
        
        # Store in history for analysis
        self._store_quality_result(original, enhanced, quality_score)
        
        # Add to review queue if flagged
        if quality_score.needs_review:
            self._add_to_review_queue(original, enhanced, quality_score)
        
        return quality_score
    
    def render_quality_analysis_ui(self, original: str, enhanced: str, 
                                  content_type: Optional[str] = None) -> Tuple[QualityScore, bool]:
        """
        Render comprehensive quality analysis UI
        
        Returns:
            Tuple of (QualityScore, user_override_applied)
        """
        # Calculate quality score
        quality_score = self.evaluate_content_quality(original, enhanced, content_type)
        
        # Render main quality display
        self._render_quality_header(quality_score)
        
        # Render component breakdown
        self._render_component_scores(quality_score)
        
        # Handle manual review flagging
        user_override = False
        if quality_score.needs_review:
            user_override = self._render_review_alert(quality_score)
        
        # Detailed analysis in expander
        self._render_detailed_analysis(quality_score)
        
        return quality_score, user_override
    
    def _render_quality_header(self, quality_score: QualityScore):
        """Render main quality score and status"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            score = quality_score.overall_score
            status = quality_score.quality_status
            
            # Status-based styling
            if status == "EXCELLENT":
                st.success(f"🌟 **Quality Score**: {score:.1%} - {status}")
            elif status == "GOOD":
                st.success(f"✅ **Quality Score**: {score:.1%} - {status}")
            elif status == "ACCEPTABLE":
                st.warning(f"⚠️ **Quality Score**: {score:.1%} - {status}")
            elif status == "POOR":
                st.error(f"❌ **Quality Score**: {score:.1%} - {status}")
            else:  # CRITICAL or ERROR
                st.error(f"🚨 **Quality Score**: {score:.1%} - {status}")
        
        with col2:
            # Review status
            if quality_score.needs_review:
                st.error("🔍 **NEEDS REVIEW**")
            else:
                st.success("✅ **APPROVED**")
        
        with col3:
            # Key metric
            coherence = quality_score.component_scores.get("coherence", 0)
            st.metric("Coherence", f"{coherence:.1%}")
    
    def _render_component_scores(self, quality_score: QualityScore):
        """Render breakdown of component scores"""
        st.subheader("📊 Quality Component Breakdown")
        
        scores = quality_score.component_scores
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Coherence", f"{scores.get('coherence', 0):.1%}")
            st.metric("📏 Length", f"{scores.get('length', 0):.1%}")
        
        with col2:
            st.metric("📖 Readability", f"{scores.get('readability', 0):.1%}")
            st.metric("💭 Sentiment", f"{scores.get('sentiment', 0):.1%}")
        
        with col3:
            st.metric("📚 Vocabulary", f"{scores.get('vocabulary', 0):.1%}")
            st.metric("🏗️ Structure", f"{scores.get('structure', 0):.1%}")
        
        # Visual progress bars
        st.write("**Component Score Visualization:**")
        for component, score in scores.items():
            if score > 0:  # Only show components with valid scores
                # Color coding based on score
                if score >= 0.8:
                    color = "🟢"
                elif score >= 0.6:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.write(f"{color} **{component.title()}**: {score:.1%}")
                st.progress(score)
    
    def _render_review_alert(self, quality_score: QualityScore) -> bool:
        """Render alert for content that needs manual review"""
        st.error("🚨 **AUTOMATIC QUALITY FLAG - MANUAL REVIEW REQUIRED**")
        
        # Show specific reasons
        st.write("**Quality issues detected:**")
        for reason in quality_score.review_reasons:
            st.write(f"• {reason}")
        
        # Explanation
        with st.expander("ℹ️ Why was this flagged?"):
            st.write(f"""
            This content was automatically flagged for manual review because it failed one or more quality thresholds:
            
            **Your Original Criteria:**
            - Coherence score below {quality_score.thresholds_used.get('coherence_min', 0.75):.1%}
            - Length ratio above {quality_score.thresholds_used.get('length_ratio_max', 1.8):.1f}x
            
            **Additional Quality Checks:**
            - Overall quality below {quality_score.thresholds_used.get('overall_quality_min', 0.7):.1%}
            - Significant sentiment drift
            - Poor vocabulary preservation
            - Structural issues
            
            Manual review ensures only high-quality enhanced content reaches your training dataset.
            """)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Send to Manual Review", type="primary"):
                st.success("✅ Added to manual review queue")
                return False
        
        with col2:
            if st.button("🔧 Override & Approve"):
                st.warning("⚠️ Quality check overridden - content approved despite issues")
                return True
        
        with col3:
            if st.button("❌ Reject Content"):
                st.error("🗑️ Content rejected - will not be included in dataset")
                return False
        
        return False
    
    def _render_detailed_analysis(self, quality_score: QualityScore):
        """Render detailed quality analysis"""
        with st.expander("🔍 Detailed Quality Analysis"):
            
            detailed = quality_score.detailed_analysis
            
            # Coherence analysis
            st.subheader("🎯 Coherence Analysis")
            coherence_score = detailed.get("coherence", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Semantic Similarity", f"{coherence_score:.3f}")
            with col2:
                if coherence_score >= 0.8:
                    st.success("Excellent semantic preservation")
                elif coherence_score >= 0.7:
                    st.info("Good semantic preservation")
                elif coherence_score >= 0.6:
                    st.warning("Moderate semantic drift")
                else:
                    st.error("Significant semantic drift detected")
            
            # Length analysis
            st.subheader("📏 Length Analysis")
            length_analysis = detailed.get("length_analysis", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Length", f"{length_analysis.get('original_length', 0):,} chars")
                st.metric("Enhanced Length", f"{length_analysis.get('enhanced_length', 0):,} chars")
            
            with col2:
                st.metric("Length Ratio", f"{length_analysis.get('ratio', 0):.2f}x")
                st.metric("Length Change", f"{length_analysis.get('length_change', 0):+,} chars")
            
            # Status indicator
            status = length_analysis.get("status", "UNKNOWN")
            message = length_analysis.get("message", "No message")
            
            if status == "OPTIMAL":
                st.success(f"✅ {message}")
            elif status == "ACCEPTABLE":
                st.info(f"ℹ️ {message}")
            elif status == "WARNING":
                st.warning(f"⚠️ {message}")
            else:  # CRITICAL
                st.error(f"🚨 {message}")
            
            # Content analysis details
            st.subheader("📝 Content Analysis")
            content_analysis = detailed.get("content_analysis", {})
            
            # Readability
            readability = content_analysis.get("readability", {})
            if readability.get("available", False):
                st.write(f"**Readability Change**: {readability.get('change', 0):+.1f} points")
                st.write(f"Original: {readability.get('original_score', 0):.1f}, Enhanced: {readability.get('enhanced_score', 0):.1f}")
            
            # Sentiment
            sentiment = content_analysis.get("sentiment", {})
            if sentiment.get("available", False):
                st.write(f"**Sentiment Drift**: {sentiment.get('drift', 0):.3f}")
                st.write(f"Original: {sentiment.get('original_sentiment', 0):+.3f}, Enhanced: {sentiment.get('enhanced_sentiment', 0):+.3f}")
            
            # Vocabulary
            vocabulary = content_analysis.get("vocabulary", {})
            if vocabulary.get("available", False):
                st.write(f"**Vocabulary Overlap**: {vocabulary.get('overlap_ratio', 0):.1%}")
                st.write(f"Unique words - Original: {vocabulary.get('original_unique_words', 0)}, Enhanced: {vocabulary.get('enhanced_unique_words', 0)}")
            
            # Structure
            structure = content_analysis.get("structure", {})
            if structure.get("available", False):
                st.write(f"**Sentence Ratio**: {structure.get('sentence_ratio', 0):.2f}")
                st.write(f"Sentences - Original: {structure.get('original_sentences', 0)}, Enhanced: {structure.get('enhanced_sentences', 0)}")
            
            # Thresholds used
            st.subheader("⚙️ Quality Thresholds")
            thresholds = quality_score.thresholds_used
            
            threshold_df = pd.DataFrame([
                {"Metric": "Coherence Minimum", "Threshold": f"{thresholds.get('coherence_min', 0):.1%}"},
                {"Metric": "Length Ratio Maximum", "Threshold": f"{thresholds.get('length_ratio_max', 0):.1f}x"},
                {"Metric": "Overall Quality Minimum", "Threshold": f"{thresholds.get('overall_quality_min', 0):.1%}"},
                {"Metric": "Sentiment Drift Maximum", "Threshold": f"{thresholds.get('sentiment_drift_max', 0):.2f}"},
                {"Metric": "Vocabulary Overlap Minimum", "Threshold": f"{thresholds.get('vocabulary_overlap_min', 0):.1%}"}
            ])
            
            st.dataframe(threshold_df, use_container_width=True)
    
    def _store_quality_result(self, original: str, enhanced: str, quality_score: QualityScore):
        """Store quality result for analysis and improvement"""
        quality_record = {
            'timestamp': quality_score.timestamp,
            'text_length': len(original),
            'overall_score': quality_score.overall_score,
            'quality_status': quality_score.quality_status,
            'needs_review': quality_score.needs_review,
            'component_scores': quality_score.component_scores,
            'review_reasons': quality_score.review_reasons,
            'text_sample': original[:100] + "..." if len(original) > 100 else original
        }
        
        self.quality_history.append(quality_record)
        
        # Keep only recent history to manage memory
        if len(self.quality_history) > 200:
            self.quality_history = self.quality_history[-100:]
    
    def _add_to_review_queue(self, original: str, enhanced: str, quality_score: QualityScore):
        """Add flagged content to manual review queue"""
        review_item = {
            'timestamp': quality_score.timestamp,
            'original': original,
            'enhanced': enhanced,
            'quality_score': quality_score,
            'status': 'PENDING'
        }
        
        self.review_queue.append(review_item)
        
        # Limit review queue size
        if len(self.review_queue) > 50:
            self.review_queue = self.review_queue[-25:]
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get comprehensive quality statistics"""
        if not self.quality_history:
            return {}
        
        df = pd.DataFrame(self.quality_history)
        
        stats = {
            'total_evaluations': len(df),
            'average_quality': df['overall_score'].mean(),
            'quality_distribution': df['quality_status'].value_counts().to_dict(),
            'review_rate': df['needs_review'].mean(),
            'component_averages': {
                component: df['component_scores'].apply(lambda x: x.get(component, 0)).mean()
                for component in ['coherence', 'length', 'readability', 'sentiment', 'vocabulary', 'structure']
            },
            'common_review_reasons': self._get_common_review_reasons(),
            'quality_trends': self._calculate_quality_trends(df),
            'review_queue_size': len(self.review_queue)
        }
        
        return stats
    
    def _get_common_review_reasons(self) -> Dict[str, int]:
        """Get most common reasons for manual review"""
        all_reasons = []
        for record in self.quality_history:
            all_reasons.extend(record.get('review_reasons', []))
        
        return dict(Counter(all_reasons).most_common(10))
    
    def _calculate_quality_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality trends over time"""
        if len(df) < 5:
            return {"insufficient_data": True}
        
        # Recent vs older quality comparison
        recent_data = df.tail(20)
        older_data = df.head(20) if len(df) > 20 else df.head(len(df)//2)
        
        trends = {
            "recent_avg_quality": recent_data['overall_score'].mean(),
            "older_avg_quality": older_data['overall_score'].mean(),
            "quality_improvement": recent_data['overall_score'].mean() - older_data['overall_score'].mean(),
            "recent_review_rate": recent_data['needs_review'].mean(),
            "older_review_rate": older_data['needs_review'].mean(),
            "review_rate_change": recent_data['needs_review'].mean() - older_data['needs_review'].mean()
        }
        
        return trends
    
    def render_quality_statistics_ui(self):
        """Render quality statistics dashboard"""
        stats = self.get_quality_statistics()
        
        if not stats:
            st.info("No quality statistics available yet.")
            return
        
        st.subheader("📊 Quality Analysis Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Evaluations", stats['total_evaluations'])
        
        with col2:
            st.metric("Average Quality", f"{stats['average_quality']:.1%}")
        
        with col3:
            st.metric("Review Rate", f"{stats['review_rate']:.1%}")
        
        with col4:
            st.metric("Review Queue", stats['review_queue_size'])
        
        # Quality distribution
        if stats['quality_distribution']:
            st.write("**Quality Status Distribution:**")
            quality_df = pd.DataFrame(list(stats['quality_distribution'].items()), 
                                    columns=['Status', 'Count'])
            st.bar_chart(quality_df.set_index('Status'))
        
        # Component averages
        st.write("**Average Component Scores:**")
        component_df = pd.DataFrame(list(stats['component_averages'].items()),
                                  columns=['Component', 'Average Score'])
        component_df['Average Score'] = component_df['Average Score'].round(3)
        st.dataframe(component_df, use_container_width=True)
        
        # Common review reasons
        if stats['common_review_reasons']:
            st.write("**Most Common Review Reasons:**")
            reasons_df = pd.DataFrame(list(stats['common_review_reasons'].items()),
                                    columns=['Reason', 'Frequency'])
            st.dataframe(reasons_df, use_container_width=True)
        
        # Quality trends
        trends = stats.get('quality_trends', {})
        if not trends.get('insufficient_data', False):
            st.write("**Quality Trends:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                quality_change = trends['quality_improvement']
                if quality_change > 0.05:
                    st.success(f"📈 Quality improving: +{quality_change:.1%}")
                elif quality_change < -0.05:
                    st.error(f"📉 Quality declining: {quality_change:.1%}")
                else:
                    st.info(f"📊 Quality stable: {quality_change:+.1%}")
            
            with col2:
                review_change = trends['review_rate_change']
                if review_change < -0.05:
                    st.success(f"📈 Fewer reviews needed: {review_change:.1%}")
                elif review_change > 0.05:
                    st.warning(f"📉 More reviews needed: +{review_change:.1%}")
                else:
                    st.info(f"📊 Review rate stable: {review_change:+.1%}")

# Example usage and testing
def main():
    """Example usage of the Quality Threshold Handler"""
    st.title("🎯 Quality Threshold Handler Demo")
    
    handler = QualityThresholdHandler()
    
    # Sample texts for testing
    sample_pairs = {
        "High Quality Enhancement": {
            "original": "What is consciousness? It's a deep question that philosophers have pondered for centuries.",
            "enhanced": "What is consciousness? This profound question has captivated philosophers, scientists, and spiritual seekers for centuries. Consciousness represents our fundamental awareness - the very capacity to experience, perceive, and know that we exist."
        },
        "Low Coherence (Drift)": {
            "original": "Meditation helps us find inner peace and clarity through mindful awareness.",
            "enhanced": "Quantum physics reveals that particles exist in multiple states simultaneously until observed, demonstrating the fundamental role of consciousness in reality."
        },
        "Excessive Length": {
            "original": "Be present in this moment.",
            "enhanced": "Being present in this moment requires a deep understanding of the nature of time, consciousness, and reality. We must cultivate awareness through various practices including meditation, mindfulness, contemplation, and self-inquiry. The present moment is not just a temporal concept but a gateway to understanding our true nature as pure awareness itself, beyond the limitations of the thinking mind and the illusions of past and future."
        },
        "Good Quality": {
            "original": "Q: How do I meditate? A: Sit quietly and observe your breath.",
            "enhanced": "Q: How do I meditate? A: Find a comfortable seated position, close your eyes gently, and bring your attention to the natural rhythm of your breath. Simply observe each inhalation and exhalation without trying to control or change anything."
        }
    }
    
    # Sample selection
    selected_sample = st.selectbox("Choose a sample pair:", list(sample_pairs.keys()))
    
    # Text inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Text")
        original = st.text_area(
            "Original:",
            value=sample_pairs[selected_sample]["original"],
            height=150
        )
    
    with col2:
        st.subheader("Enhanced Text")
        enhanced = st.text_area(
            "Enhanced:",
            value=sample_pairs[selected_sample]["enhanced"],
            height=150
        )
    
    # Content type selection
    content_type = st.selectbox(
        "Content Type:",
        ["auto_detect", "qa_pair", "dialogue", "monologue", "mixed"]
    )
    
    if content_type == "auto_detect":
        content_type = None
    
    if original and enhanced:
        st.subheader("Quality Analysis Results")
        
        # Perform quality analysis
        quality_score, user_override = handler.render_quality_analysis_ui(
            original, enhanced, content_type
        )
        
        # Show final decision
        if user_override:
            st.warning("⚠️ **Final Decision**: Approved with user override")
        elif quality_score.needs_review:
            st.error("🔍 **Final Decision**: Flagged for manual review")
        else:
            st.success("✅ **Final Decision**: Automatically approved")
    
    # Quality statistics
    with st.expander("📊 Quality Statistics Dashboard"):
        handler.render_quality_statistics_ui()

if __name__ == "__main__":
    main()

