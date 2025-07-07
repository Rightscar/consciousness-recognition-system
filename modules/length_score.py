"""
Length Score Module
==================

Penalize overly long outputs and reward appropriate brevity
to maintain optimal content length for training data.

Features:
- Ideal length range analysis
- Brevity reward system
- Length penalty calculation
- Content density assessment
- Efficiency scoring
- Length optimization recommendations
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from modules.logger import get_logger, log_event

class LengthScorer:
    """
    Comprehensive length analysis and scoring system
    
    Features:
    - Optimal length range detection
    - Brevity vs completeness balance
    - Content density analysis
    - Efficiency scoring
    - Length penalty calculation
    - Detailed length recommendations
    """
    
    def __init__(self):
        self.logger = get_logger("length_scorer")
        
        # Optimal length ranges for different content types
        self.optimal_ranges = {
            'qa_format': {
                'question': {'min': 5, 'ideal_min': 8, 'ideal_max': 25, 'max': 40},
                'answer': {'min': 10, 'ideal_min': 20, 'ideal_max': 100, 'max': 200},
                'total': {'min': 15, 'ideal_min': 30, 'ideal_max': 120, 'max': 250}
            },
            'chat_format': {
                'user_message': {'min': 3, 'ideal_min': 5, 'ideal_max': 30, 'max': 60},
                'assistant_message': {'min': 5, 'ideal_min': 10, 'ideal_max': 80, 'max': 150},
                'total': {'min': 10, 'ideal_min': 20, 'ideal_max': 100, 'max': 200}
            },
            'instruction_format': {
                'instruction': {'min': 5, 'ideal_min': 10, 'ideal_max': 50, 'max': 100},
                'output': {'min': 10, 'ideal_min': 20, 'ideal_max': 150, 'max': 300},
                'total': {'min': 15, 'ideal_min': 35, 'ideal_max': 180, 'max': 400}
            },
            'narrative_format': {
                'content': {'min': 20, 'ideal_min': 50, 'ideal_max': 200, 'max': 500},
                'total': {'min': 20, 'ideal_min': 50, 'ideal_max': 200, 'max': 500}
            },
            'general': {
                'content': {'min': 10, 'ideal_min': 25, 'ideal_max': 150, 'max': 300},
                'total': {'min': 10, 'ideal_min': 25, 'ideal_max': 150, 'max': 300}
            }
        }
        
        # Scoring thresholds
        self.thresholds = {
            'excellent': 0.95,
            'good': 0.85,
            'acceptable': 0.75,
            'concerning': 0.65,
            'poor': 0.50
        }
        
        # Initialize session state
        if 'length_analysis_cache' not in st.session_state:
            st.session_state['length_analysis_cache'] = {}
    
    def analyze_length(self, content: str, content_type: str = 'general', original_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive length analysis
        
        Args:
            content: Content to analyze
            content_type: Type of content ('qa_format', 'chat_format', etc.)
            original_content: Original content for comparison
        
        Returns:
            Detailed length analysis results
        """
        
        try:
            # Create cache key
            cache_key = f"{hash(content)}_{content_type}_{hash(original_content or '')}"
            
            # Check cache
            if cache_key in st.session_state['length_analysis_cache']:
                return st.session_state['length_analysis_cache'][cache_key]
            
            # Perform analysis
            results = {
                'content_type': content_type,
                'basic_metrics': self.calculate_basic_metrics(content),
                'length_scoring': self.calculate_length_scoring(content, content_type),
                'content_density': self.analyze_content_density(content),
                'efficiency_metrics': self.calculate_efficiency_metrics(content),
                'brevity_analysis': self.analyze_brevity(content, content_type),
                'verbosity_check': self.check_verbosity(content),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Compare with original if provided
            if original_content:
                results['expansion_analysis'] = self.analyze_length_expansion(content, original_content, content_type)
                results['length_drift'] = self.calculate_length_drift(content, original_content)
            
            # Calculate composite scores
            results['overall_length_score'] = self.calculate_overall_length_score(results)
            results['length_confidence'] = self.calculate_length_confidence(results)
            results['optimization_potential'] = self.calculate_optimization_potential(results)
            
            # Add quality assessment
            results['quality_assessment'] = self.assess_length_quality(results)
            
            # Cache results
            st.session_state['length_analysis_cache'][cache_key] = results
            
            # Log analysis
            log_event("length_analyzed", {
                "content_type": content_type,
                "overall_length_score": results['overall_length_score'],
                "length_confidence": results['length_confidence']
            }, "length_scorer")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Length analysis failed: {str(e)}")
            return self.get_fallback_length_results(content_type)
    
    def calculate_basic_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate basic length metrics"""
        
        try:
            # Character metrics
            char_count = len(content)
            char_count_no_spaces = len(content.replace(' ', ''))
            
            # Word metrics
            words = content.split()
            word_count = len(words)
            
            # Sentence metrics
            sentences = self.split_sentences(content)
            sentence_count = len(sentences)
            
            # Average metrics
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Paragraph metrics
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            paragraph_count = len(paragraphs)
            avg_paragraph_length = word_count / paragraph_count if paragraph_count > 0 else 0
            
            return {
                'char_count': char_count,
                'char_count_no_spaces': char_count_no_spaces,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'paragraph_count': paragraph_count,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'avg_paragraph_length': avg_paragraph_length,
                'words_per_char': word_count / char_count if char_count > 0 else 0
            }
        
        except Exception as e:
            self.logger.warning(f"Basic metrics calculation failed: {str(e)}")
            return {'word_count': 0, 'char_count': 0, 'sentence_count': 0}
    
    def calculate_length_scoring(self, content: str, content_type: str) -> Dict[str, Any]:
        """Calculate length scoring based on optimal ranges"""
        
        try:
            word_count = len(content.split())
            
            # Get optimal ranges for content type
            ranges = self.optimal_ranges.get(content_type, self.optimal_ranges['general'])
            total_range = ranges.get('total', ranges.get('content', ranges['general']['content']))
            
            # Calculate position within ranges
            min_length = total_range['min']
            ideal_min = total_range['ideal_min']
            ideal_max = total_range['ideal_max']
            max_length = total_range['max']
            
            # Calculate length score
            if word_count < min_length:
                # Too short
                score = max(0.0, word_count / min_length * 0.5)
                category = 'too_short'
                penalty = (min_length - word_count) / min_length
            elif word_count <= ideal_min:
                # Short but acceptable
                score = 0.5 + (word_count - min_length) / (ideal_min - min_length) * 0.3
                category = 'short'
                penalty = 0.0
            elif word_count <= ideal_max:
                # Ideal range
                score = 0.8 + (1.0 - abs(word_count - (ideal_min + ideal_max) / 2) / ((ideal_max - ideal_min) / 2)) * 0.2
                category = 'ideal'
                penalty = 0.0
            elif word_count <= max_length:
                # Long but acceptable
                score = 0.8 - (word_count - ideal_max) / (max_length - ideal_max) * 0.3
                category = 'long'
                penalty = (word_count - ideal_max) / (max_length - ideal_max) * 0.3
            else:
                # Too long
                score = max(0.0, 0.5 - (word_count - max_length) / max_length * 0.5)
                category = 'too_long'
                penalty = (word_count - max_length) / max_length
            
            # Calculate efficiency bonus
            if category == 'ideal' and word_count <= (ideal_min + ideal_max) / 2:
                efficiency_bonus = 0.1  # Reward conciseness within ideal range
            else:
                efficiency_bonus = 0.0
            
            final_score = min(1.0, score + efficiency_bonus)
            
            return {
                'word_count': word_count,
                'length_score': final_score,
                'length_category': category,
                'length_penalty': penalty,
                'efficiency_bonus': efficiency_bonus,
                'optimal_range': total_range,
                'position_in_range': self.calculate_range_position(word_count, total_range)
            }
        
        except Exception as e:
            self.logger.warning(f"Length scoring calculation failed: {str(e)}")
            return {'word_count': 0, 'length_score': 0.5, 'length_category': 'unknown'}
    
    def analyze_content_density(self, content: str) -> Dict[str, Any]:
        """Analyze content density and information richness"""
        
        try:
            words = content.split()
            word_count = len(words)
            
            if word_count == 0:
                return {'density_score': 0.0, 'information_ratio': 0.0}
            
            # Unique word ratio
            unique_words = set(word.lower() for word in words)
            unique_ratio = len(unique_words) / word_count
            
            # Content word ratio (excluding common stop words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
            content_words = [word for word in words if word.lower() not in stop_words]
            content_ratio = len(content_words) / word_count
            
            # Information density (combination of uniqueness and content)
            density_score = (unique_ratio * 0.6) + (content_ratio * 0.4)
            
            # Sentence density
            sentences = self.split_sentences(content)
            avg_sentence_length = word_count / len(sentences) if sentences else 0
            
            # Optimal sentence length is around 15-20 words
            sentence_density = 1.0 - abs(avg_sentence_length - 17.5) / 17.5 if avg_sentence_length > 0 else 0
            sentence_density = max(0.0, min(1.0, sentence_density))
            
            # Paragraph density
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            avg_paragraph_length = word_count / len(paragraphs) if paragraphs else 0
            
            return {
                'density_score': density_score,
                'unique_ratio': unique_ratio,
                'content_ratio': content_ratio,
                'information_ratio': density_score,
                'sentence_density': sentence_density,
                'avg_sentence_length': avg_sentence_length,
                'avg_paragraph_length': avg_paragraph_length,
                'total_sentences': len(sentences),
                'total_paragraphs': len(paragraphs)
            }
        
        except Exception as e:
            self.logger.warning(f"Content density analysis failed: {str(e)}")
            return {'density_score': 0.5, 'information_ratio': 0.5}
    
    def calculate_efficiency_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate content efficiency metrics"""
        
        try:
            words = content.split()
            word_count = len(words)
            char_count = len(content)
            
            if word_count == 0:
                return {'efficiency_score': 0.0, 'conciseness_ratio': 0.0}
            
            # Words per character (efficiency indicator)
            words_per_char = word_count / char_count if char_count > 0 else 0
            
            # Average word length (shorter words often indicate efficiency)
            avg_word_length = np.mean([len(word) for word in words])
            word_efficiency = max(0.0, 1.0 - (avg_word_length - 5) / 10)  # Optimal around 5 chars
            
            # Sentence efficiency
            sentences = self.split_sentences(content)
            avg_sentence_length = word_count / len(sentences) if sentences else 0
            sentence_efficiency = max(0.0, 1.0 - abs(avg_sentence_length - 15) / 15)  # Optimal around 15 words
            
            # Punctuation efficiency (not too much, not too little)
            punctuation_count = sum(1 for char in content if char in '.,!?;:')
            punctuation_ratio = punctuation_count / word_count if word_count > 0 else 0
            punctuation_efficiency = max(0.0, 1.0 - abs(punctuation_ratio - 0.1) / 0.1)  # Optimal around 10%
            
            # Overall efficiency score
            efficiency_score = (
                words_per_char * 0.2 +
                word_efficiency * 0.3 +
                sentence_efficiency * 0.3 +
                punctuation_efficiency * 0.2
            )
            
            # Conciseness ratio (information per word)
            unique_words = set(word.lower() for word in words)
            conciseness_ratio = len(unique_words) / word_count
            
            return {
                'efficiency_score': efficiency_score,
                'words_per_char': words_per_char,
                'word_efficiency': word_efficiency,
                'sentence_efficiency': sentence_efficiency,
                'punctuation_efficiency': punctuation_efficiency,
                'conciseness_ratio': conciseness_ratio,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'punctuation_ratio': punctuation_ratio
            }
        
        except Exception as e:
            self.logger.warning(f"Efficiency metrics calculation failed: {str(e)}")
            return {'efficiency_score': 0.5, 'conciseness_ratio': 0.5}
    
    def analyze_brevity(self, content: str, content_type: str) -> Dict[str, Any]:
        """Analyze brevity and conciseness"""
        
        try:
            word_count = len(content.split())
            
            # Get optimal ranges
            ranges = self.optimal_ranges.get(content_type, self.optimal_ranges['general'])
            total_range = ranges.get('total', ranges.get('content', ranges['general']['content']))
            
            ideal_min = total_range['ideal_min']
            ideal_max = total_range['ideal_max']
            ideal_midpoint = (ideal_min + ideal_max) / 2
            
            # Brevity score (reward being closer to ideal minimum)
            if word_count <= ideal_min:
                brevity_score = 1.0
            elif word_count <= ideal_midpoint:
                brevity_score = 1.0 - (word_count - ideal_min) / (ideal_midpoint - ideal_min) * 0.3
            elif word_count <= ideal_max:
                brevity_score = 0.7 - (word_count - ideal_midpoint) / (ideal_max - ideal_midpoint) * 0.3
            else:
                brevity_score = max(0.0, 0.4 - (word_count - ideal_max) / ideal_max * 0.4)
            
            # Conciseness indicators
            sentences = self.split_sentences(content)
            avg_sentence_length = word_count / len(sentences) if sentences else 0
            
            # Shorter sentences often indicate better conciseness
            sentence_conciseness = max(0.0, 1.0 - (avg_sentence_length - 12) / 20) if avg_sentence_length > 0 else 1.0
            
            # Word choice efficiency
            words = content.split()
            long_words = sum(1 for word in words if len(word) > 8)
            long_word_ratio = long_words / len(words) if words else 0
            word_conciseness = max(0.0, 1.0 - long_word_ratio * 2)  # Penalize excessive long words
            
            # Overall brevity assessment
            overall_brevity = (brevity_score * 0.5 + sentence_conciseness * 0.3 + word_conciseness * 0.2)
            
            return {
                'brevity_score': brevity_score,
                'overall_brevity': overall_brevity,
                'sentence_conciseness': sentence_conciseness,
                'word_conciseness': word_conciseness,
                'avg_sentence_length': avg_sentence_length,
                'long_word_ratio': long_word_ratio,
                'ideal_range_position': (word_count - ideal_min) / (ideal_max - ideal_min) if ideal_max > ideal_min else 0.5
            }
        
        except Exception as e:
            self.logger.warning(f"Brevity analysis failed: {str(e)}")
            return {'brevity_score': 0.5, 'overall_brevity': 0.5}
    
    def check_verbosity(self, content: str) -> Dict[str, Any]:
        """Check for verbosity and unnecessary length"""
        
        try:
            # Verbosity indicators
            verbosity_indicators = {
                'filler_phrases': [
                    'it is important to note that',
                    'it should be mentioned that',
                    'as we can see',
                    'in other words',
                    'to put it simply',
                    'basically',
                    'essentially'
                ],
                'redundant_qualifiers': [
                    'very', 'quite', 'rather', 'somewhat', 'fairly', 'pretty',
                    'really', 'truly', 'actually', 'literally'
                ],
                'unnecessary_connectors': [
                    'furthermore', 'moreover', 'additionally', 'in addition',
                    'also', 'likewise', 'similarly'
                ]
            }
            
            content_lower = content.lower()
            verbosity_count = 0
            found_indicators = {}
            
            for category, indicators in verbosity_indicators.items():
                category_count = 0
                found_in_category = []
                
                for indicator in indicators:
                    count = content_lower.count(indicator)
                    if count > 0:
                        category_count += count
                        found_in_category.append((indicator, count))
                
                found_indicators[category] = found_in_category
                verbosity_count += category_count
            
            # Calculate verbosity score
            word_count = len(content.split())
            verbosity_ratio = verbosity_count / word_count if word_count > 0 else 0
            verbosity_score = max(0.0, 1.0 - verbosity_ratio * 10)  # Scale verbosity impact
            
            # Repetition check
            words = content.split()
            word_counts = {}
            for word in words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
            repeated_words = {word: count for word, count in word_counts.items() if count > 3 and len(word) > 3}
            repetition_score = max(0.0, 1.0 - len(repeated_words) / len(word_counts) * 5) if word_counts else 1.0
            
            # Overall verbosity assessment
            overall_verbosity = (verbosity_score * 0.6 + repetition_score * 0.4)
            
            return {
                'verbosity_score': overall_verbosity,
                'verbosity_count': verbosity_count,
                'verbosity_ratio': verbosity_ratio,
                'found_indicators': found_indicators,
                'repeated_words': repeated_words,
                'repetition_score': repetition_score,
                'conciseness_level': self.classify_conciseness_level(overall_verbosity)
            }
        
        except Exception as e:
            self.logger.warning(f"Verbosity check failed: {str(e)}")
            return {'verbosity_score': 0.7, 'conciseness_level': 'moderate'}
    
    def analyze_length_expansion(self, enhanced_content: str, original_content: str, content_type: str) -> Dict[str, Any]:
        """Analyze length expansion from original to enhanced content"""
        
        try:
            original_words = len(original_content.split())
            enhanced_words = len(enhanced_content.split())
            
            expansion_ratio = enhanced_words / original_words if original_words > 0 else 1.0
            expansion_amount = enhanced_words - original_words
            
            # Get optimal ranges for content type
            ranges = self.optimal_ranges.get(content_type, self.optimal_ranges['general'])
            total_range = ranges.get('total', ranges.get('content', ranges['general']['content']))
            
            # Assess expansion quality
            if expansion_ratio <= 1.1:
                expansion_quality = 'minimal'
                expansion_score = 1.0
            elif expansion_ratio <= 1.5:
                expansion_quality = 'moderate'
                expansion_score = 0.9
            elif expansion_ratio <= 2.0:
                expansion_quality = 'significant'
                expansion_score = 0.7
            elif expansion_ratio <= 3.0:
                expansion_quality = 'excessive'
                expansion_score = 0.4
            else:
                expansion_quality = 'extreme'
                expansion_score = 0.1
            
            # Check if expansion brings content into optimal range
            original_in_range = self.is_in_optimal_range(original_words, total_range)
            enhanced_in_range = self.is_in_optimal_range(enhanced_words, total_range)
            
            if not original_in_range and enhanced_in_range:
                range_improvement = 'improved'
                range_bonus = 0.2
            elif original_in_range and not enhanced_in_range:
                range_improvement = 'degraded'
                range_bonus = -0.3
            elif original_in_range and enhanced_in_range:
                range_improvement = 'maintained'
                range_bonus = 0.1
            else:
                range_improvement = 'no_change'
                range_bonus = 0.0
            
            final_expansion_score = max(0.0, min(1.0, expansion_score + range_bonus))
            
            return {
                'expansion_ratio': expansion_ratio,
                'expansion_amount': expansion_amount,
                'original_words': original_words,
                'enhanced_words': enhanced_words,
                'expansion_quality': expansion_quality,
                'expansion_score': final_expansion_score,
                'range_improvement': range_improvement,
                'range_bonus': range_bonus,
                'original_in_range': original_in_range,
                'enhanced_in_range': enhanced_in_range
            }
        
        except Exception as e:
            self.logger.warning(f"Length expansion analysis failed: {str(e)}")
            return {'expansion_ratio': 1.0, 'expansion_quality': 'unknown'}
    
    def calculate_length_drift(self, enhanced_content: str, original_content: str) -> float:
        """Calculate length drift between original and enhanced content"""
        
        try:
            original_words = len(original_content.split())
            enhanced_words = len(enhanced_content.split())
            
            if original_words == 0:
                return 0.0
            
            # Calculate relative change
            relative_change = abs(enhanced_words - original_words) / original_words
            
            # Drift score (0 = no change, 1 = extreme change)
            drift_score = min(1.0, relative_change)
            
            return drift_score
        
        except Exception as e:
            self.logger.warning(f"Length drift calculation failed: {str(e)}")
            return 0.3
    
    def calculate_overall_length_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall length score"""
        
        # Collect component scores
        length_scoring = results.get('length_scoring', {})
        length_score = length_scoring.get('length_score', 0.5)
        
        density_analysis = results.get('content_density', {})
        density_score = density_analysis.get('density_score', 0.5)
        
        efficiency_metrics = results.get('efficiency_metrics', {})
        efficiency_score = efficiency_metrics.get('efficiency_score', 0.5)
        
        brevity_analysis = results.get('brevity_analysis', {})
        brevity_score = brevity_analysis.get('overall_brevity', 0.5)
        
        verbosity_check = results.get('verbosity_check', {})
        verbosity_score = verbosity_check.get('verbosity_score', 0.5)
        
        # Weighted combination
        overall_score = (
            length_score * 0.3 +
            density_score * 0.2 +
            efficiency_score * 0.2 +
            brevity_score * 0.15 +
            verbosity_score * 0.15
        )
        
        return overall_score
    
    def calculate_length_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in length analysis"""
        
        basic_metrics = results.get('basic_metrics', {})
        word_count = basic_metrics.get('word_count', 0)
        
        # Higher confidence with more content
        content_confidence = min(1.0, word_count / 50)  # Full confidence at 50+ words
        
        # Consistency across metrics
        scores = [
            results.get('overall_length_score', 0.5),
            results.get('content_density', {}).get('density_score', 0.5),
            results.get('efficiency_metrics', {}).get('efficiency_score', 0.5)
        ]
        
        # Calculate variance (lower variance = higher confidence)
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        consistency_confidence = max(0.0, 1.0 - (variance * 4))  # Scale variance
        
        # Combined confidence
        confidence = (content_confidence * 0.4 + consistency_confidence * 0.6)
        
        return confidence
    
    def calculate_optimization_potential(self, results: Dict[str, Any]) -> float:
        """Calculate potential for length optimization"""
        
        length_scoring = results.get('length_scoring', {})
        length_category = length_scoring.get('length_category', 'unknown')
        
        verbosity_check = results.get('verbosity_check', {})
        verbosity_score = verbosity_check.get('verbosity_score', 0.7)
        
        # Higher optimization potential for verbose or poorly sized content
        if length_category in ['too_long', 'long']:
            length_optimization = 0.8
        elif length_category in ['too_short', 'short']:
            length_optimization = 0.6
        else:
            length_optimization = 0.2
        
        verbosity_optimization = 1.0 - verbosity_score
        
        # Combined optimization potential
        optimization_potential = (length_optimization * 0.6 + verbosity_optimization * 0.4)
        
        return optimization_potential
    
    def assess_length_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall length quality"""
        
        overall_score = results.get('overall_length_score', 0.5)
        confidence = results.get('length_confidence', 0.5)
        
        # Determine quality level
        if overall_score >= self.thresholds['excellent'] and confidence >= 0.8:
            quality_level = 'excellent'
            status = 'pass'
            message = "Excellent length optimization"
        elif overall_score >= self.thresholds['good'] and confidence >= 0.6:
            quality_level = 'good'
            status = 'pass'
            message = "Good length balance"
        elif overall_score >= self.thresholds['acceptable']:
            quality_level = 'acceptable'
            status = 'pass'
            message = "Acceptable length"
        elif overall_score >= self.thresholds['concerning']:
            quality_level = 'concerning'
            status = 'review'
            message = "Concerning length issues"
        else:
            quality_level = 'poor'
            status = 'fail'
            message = "Poor length optimization"
        
        # Generate recommendations
        recommendations = self.generate_length_recommendations(results)
        
        return {
            'quality_level': quality_level,
            'status': status,
            'message': message,
            'recommendations': recommendations,
            'pass_threshold': overall_score >= self.thresholds['acceptable'],
            'confidence': confidence
        }
    
    def generate_length_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for length optimization"""
        
        recommendations = []
        
        # Length scoring recommendations
        length_scoring = results.get('length_scoring', {})
        length_category = length_scoring.get('length_category', 'unknown')
        
        if length_category == 'too_short':
            recommendations.append("Expand content to meet minimum length requirements")
        elif length_category == 'short':
            recommendations.append("Consider adding more detail or examples")
        elif length_category == 'long':
            recommendations.append("Consider condensing content for better readability")
        elif length_category == 'too_long':
            recommendations.append("Significantly reduce content length - focus on essential information")
        
        # Verbosity recommendations
        verbosity_check = results.get('verbosity_check', {})
        verbosity_score = verbosity_check.get('verbosity_score', 0.7)
        
        if verbosity_score < 0.6:
            recommendations.append("Remove filler phrases and unnecessary qualifiers")
            
            found_indicators = verbosity_check.get('found_indicators', {})
            for category, indicators in found_indicators.items():
                if indicators:
                    recommendations.append(f"Reduce {category.replace('_', ' ')}")
        
        # Efficiency recommendations
        efficiency_metrics = results.get('efficiency_metrics', {})
        efficiency_score = efficiency_metrics.get('efficiency_score', 0.5)
        
        if efficiency_score < 0.6:
            avg_sentence_length = efficiency_metrics.get('avg_sentence_length', 0)
            if avg_sentence_length > 20:
                recommendations.append("Break down long sentences for better readability")
            
            avg_word_length = efficiency_metrics.get('avg_word_length', 0)
            if avg_word_length > 6:
                recommendations.append("Use simpler, shorter words where possible")
        
        # Density recommendations
        density_analysis = results.get('content_density', {})
        density_score = density_analysis.get('density_score', 0.5)
        
        if density_score < 0.6:
            recommendations.append("Increase information density - add more meaningful content")
        
        # Expansion recommendations
        expansion_analysis = results.get('expansion_analysis', {})
        expansion_quality = expansion_analysis.get('expansion_quality', 'unknown')
        
        if expansion_quality == 'excessive':
            recommendations.append("Expansion is excessive - focus on quality over quantity")
        elif expansion_quality == 'extreme':
            recommendations.append("Extreme expansion detected - significantly reduce added content")
        
        return recommendations
    
    def calculate_range_position(self, word_count: int, range_dict: Dict[str, int]) -> float:
        """Calculate position within optimal range (0-1 scale)"""
        
        min_length = range_dict['min']
        max_length = range_dict['max']
        
        if word_count <= min_length:
            return 0.0
        elif word_count >= max_length:
            return 1.0
        else:
            return (word_count - min_length) / (max_length - min_length)
    
    def is_in_optimal_range(self, word_count: int, range_dict: Dict[str, int]) -> bool:
        """Check if word count is in optimal range"""
        
        ideal_min = range_dict['ideal_min']
        ideal_max = range_dict['ideal_max']
        
        return ideal_min <= word_count <= ideal_max
    
    def classify_conciseness_level(self, verbosity_score: float) -> str:
        """Classify conciseness level based on verbosity score"""
        
        if verbosity_score >= 0.9:
            return 'very_concise'
        elif verbosity_score >= 0.8:
            return 'concise'
        elif verbosity_score >= 0.6:
            return 'moderate'
        elif verbosity_score >= 0.4:
            return 'verbose'
        else:
            return 'very_verbose'
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def get_fallback_length_results(self, content_type: str) -> Dict[str, Any]:
        """Return fallback results when analysis fails"""
        
        return {
            'content_type': content_type,
            'basic_metrics': {'word_count': 0, 'char_count': 0, 'sentence_count': 0},
            'length_scoring': {'word_count': 0, 'length_score': 0.5, 'length_category': 'unknown'},
            'content_density': {'density_score': 0.5, 'information_ratio': 0.5},
            'efficiency_metrics': {'efficiency_score': 0.5, 'conciseness_ratio': 0.5},
            'brevity_analysis': {'brevity_score': 0.5, 'overall_brevity': 0.5},
            'verbosity_check': {'verbosity_score': 0.7, 'conciseness_level': 'moderate'},
            'overall_length_score': 0.5,
            'length_confidence': 0.3,
            'optimization_potential': 0.5,
            'quality_assessment': {
                'quality_level': 'unknown',
                'status': 'review',
                'message': 'Length analysis failed - manual review required',
                'recommendations': ['Manual length review required due to analysis error'],
                'pass_threshold': False,
                'confidence': 0.3
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def render_length_analysis(self, content: str, content_type: str = 'general', original_content: Optional[str] = None) -> Dict[str, Any]:
        """Render length analysis interface"""
        
        st.subheader("📏 Length Analysis")
        
        # Analyze length
        with st.spinner("Analyzing content length..."):
            results = self.analyze_length(content, content_type, original_content)
        
        # Display results
        self.render_length_metrics(results)
        self.render_length_assessment(results)
        self.render_length_recommendations(results)
        
        return results
    
    def render_length_metrics(self, results: Dict[str, Any]):
        """Render length analysis metrics"""
        
        st.markdown("**📏 Length Metrics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_score = results.get('overall_length_score', 0.5)
            st.metric(
                "Overall Length Score",
                f"{overall_score:.3f}",
                delta=f"{overall_score - 0.8:.3f}" if overall_score != 0.8 else None
            )
        
        with col2:
            basic_metrics = results.get('basic_metrics', {})
            word_count = basic_metrics.get('word_count', 0)
            st.metric(
                "Word Count",
                f"{word_count}",
                delta=None
            )
        
        with col3:
            length_scoring = results.get('length_scoring', {})
            length_score = length_scoring.get('length_score', 0.5)
            st.metric(
                "Length Score",
                f"{length_score:.3f}",
                delta=f"{length_score - 0.8:.3f}" if length_score != 0.8 else None
            )
        
        with col4:
            brevity_analysis = results.get('brevity_analysis', {})
            brevity_score = brevity_analysis.get('overall_brevity', 0.5)
            st.metric(
                "Brevity Score",
                f"{brevity_score:.3f}",
                delta=f"{brevity_score - 0.8:.3f}" if brevity_score != 0.8 else None
            )
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            density_analysis = results.get('content_density', {})
            density_score = density_analysis.get('density_score', 0.5)
            st.metric(
                "Content Density",
                f"{density_score:.3f}",
                delta=f"{density_score - 0.7:.3f}" if density_score != 0.7 else None
            )
        
        with col2:
            efficiency_metrics = results.get('efficiency_metrics', {})
            efficiency_score = efficiency_metrics.get('efficiency_score', 0.5)
            st.metric(
                "Efficiency Score",
                f"{efficiency_score:.3f}",
                delta=f"{efficiency_score - 0.7:.3f}" if efficiency_score != 0.7 else None
            )
        
        with col3:
            verbosity_check = results.get('verbosity_check', {})
            verbosity_score = verbosity_check.get('verbosity_score', 0.5)
            st.metric(
                "Conciseness Score",
                f"{verbosity_score:.3f}",
                delta=f"{verbosity_score - 0.8:.3f}" if verbosity_score != 0.8 else None
            )
    
    def render_length_assessment(self, results: Dict[str, Any]):
        """Render length quality assessment"""
        
        assessment = results.get('quality_assessment', {})
        quality_level = assessment.get('quality_level', 'unknown')
        status = assessment.get('status', 'review')
        message = assessment.get('message', 'Unknown status')
        confidence = assessment.get('confidence', 0.5)
        
        st.markdown("**📏 Length Quality Assessment:**")
        
        # Status indicator
        if status == 'pass':
            if quality_level == 'excellent':
                st.success(f"🎉 {message}")
            else:
                st.success(f"✅ {message}")
        elif status == 'review':
            st.warning(f"⚠️ {message}")
        else:
            st.error(f"❌ {message}")
        
        # Quality and confidence indicators
        col1, col2 = st.columns(2)
        
        with col1:
            quality_colors = {
                'excellent': '🟢',
                'good': '🟡',
                'acceptable': '🟠',
                'concerning': '🔴',
                'poor': '⚫',
                'unknown': '⚪'
            }
            st.write(f"**Quality Level:** {quality_colors.get(quality_level, '⚪')} {quality_level.title()}")
        
        with col2:
            confidence_color = '🟢' if confidence >= 0.8 else '🟡' if confidence >= 0.6 else '🔴'
            st.write(f"**Confidence:** {confidence_color} {confidence:.1%}")
        
        # Length category
        length_scoring = results.get('length_scoring', {})
        length_category = length_scoring.get('length_category', 'unknown')
        category_colors = {
            'ideal': '🟢',
            'short': '🟡',
            'long': '🟡',
            'too_short': '🔴',
            'too_long': '🔴',
            'unknown': '⚪'
        }
        st.write(f"**Length Category:** {category_colors.get(length_category, '⚪')} {length_category.replace('_', ' ').title()}")
    
    def render_length_recommendations(self, results: Dict[str, Any]):
        """Render length improvement recommendations"""
        
        assessment = results.get('quality_assessment', {})
        recommendations = assessment.get('recommendations', [])
        
        if recommendations:
            st.markdown("**💡 Length Optimization Recommendations:**")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations - length analysis looks good!")

# Integration function for main app
def analyze_content_length(content: str, content_type: str = 'general', original_content: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze content length and optimization
    
    Usage:
    from modules.length_score import analyze_content_length
    
    results = analyze_content_length(enhanced_content, 'qa_format', original_content)
    """
    
    scorer = LengthScorer()
    return scorer.analyze_length(content, content_type, original_content)

# Quick length check
def quick_length_check(content: str, content_type: str = 'general') -> float:
    """
    Quick length check returning overall score
    
    Usage:
    from modules.length_score import quick_length_check
    
    score = quick_length_check(enhanced_content, 'chat_format')
    """
    
    scorer = LengthScorer()
    results = scorer.analyze_length(content, content_type)
    return results.get('overall_length_score', 0.5)

if __name__ == "__main__":
    # Test the length scorer
    st.set_page_config(page_title="Length Scorer Test", layout="wide")
    
    st.title("Length Scorer Test")
    
    # Sample content
    content = st.text_area(
        "Content to Analyze",
        value="This is a sample text for length analysis. It contains multiple sentences to test various length metrics including word count, sentence structure, and overall content density.",
        height=100
    )
    
    # Content type selection
    content_type = st.selectbox(
        "Content Type",
        ['general', 'qa_format', 'chat_format', 'instruction_format', 'narrative_format']
    )
    
    if st.button("Analyze Length"):
        scorer = LengthScorer()
        results = scorer.render_length_analysis(content, content_type)

