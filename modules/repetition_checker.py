"""
Repetition Checker Module
========================

Detect filler or redundant phrases to prevent repetitive content
and overwriting with unnecessary expansion.

Features:
- Token uniqueness analysis
- N-gram repeat detection
- Filler phrase identification
- Redundancy scoring
- Content compression analysis
- Diversity metrics
"""

import streamlit as st
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime
import numpy as np
from modules.logger import get_logger, log_event

class RepetitionChecker:
    """
    Comprehensive repetition and redundancy analysis
    
    Features:
    - Token uniqueness measurement
    - N-gram repetition detection
    - Filler phrase identification
    - Content diversity analysis
    - Compression ratio calculation
    - Redundancy scoring
    """
    
    def __init__(self):
        self.logger = get_logger("repetition_checker")
        
        # Common filler phrases and words
        self.filler_phrases = {
            'redundant_starters': [
                'it is important to note that',
                'it should be mentioned that',
                'it is worth noting that',
                'as we can see',
                'as mentioned before',
                'as previously stated',
                'in other words',
                'to put it simply',
                'basically',
                'essentially',
                'fundamentally'
            ],
            'empty_connectors': [
                'furthermore',
                'moreover',
                'additionally',
                'in addition',
                'also',
                'likewise',
                'similarly',
                'correspondingly',
                'consequently',
                'therefore',
                'thus',
                'hence'
            ],
            'vague_qualifiers': [
                'very',
                'quite',
                'rather',
                'somewhat',
                'fairly',
                'pretty',
                'really',
                'truly',
                'actually',
                'literally',
                'basically',
                'essentially'
            ],
            'redundant_phrases': [
                'in my opinion',
                'i believe that',
                'it seems to me',
                'from my perspective',
                'in my view',
                'personally',
                'i think that',
                'i feel that',
                'it appears that',
                'it would seem'
            ]
        }
        
        # Repetition thresholds
        self.thresholds = {
            'excellent': 0.95,  # Very low repetition
            'good': 0.85,
            'acceptable': 0.75,
            'concerning': 0.65,
            'poor': 0.50
        }
        
        # Initialize session state
        if 'repetition_analysis_cache' not in st.session_state:
            st.session_state['repetition_analysis_cache'] = {}
    
    def analyze_repetition(self, text: str, original_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive repetition analysis
        
        Args:
            text: Text to analyze for repetition
            original_text: Original text for comparison
        
        Returns:
            Detailed repetition analysis results
        """
        
        try:
            # Create cache key
            cache_key = f"{hash(text)}_{hash(original_text or '')}"
            
            # Check cache
            if cache_key in st.session_state['repetition_analysis_cache']:
                return st.session_state['repetition_analysis_cache'][cache_key]
            
            # Perform analysis
            results = {
                'token_uniqueness': self.calculate_token_uniqueness(text),
                'ngram_repetition': self.analyze_ngram_repetition(text),
                'filler_analysis': self.analyze_filler_content(text),
                'content_diversity': self.calculate_content_diversity(text),
                'compression_ratio': self.calculate_compression_ratio(text),
                'redundancy_score': self.calculate_redundancy_score(text),
                'sentence_similarity': self.analyze_sentence_similarity(text),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Compare with original if provided
            if original_text:
                results['expansion_analysis'] = self.analyze_expansion_quality(text, original_text)
                results['content_drift'] = self.calculate_content_drift(text, original_text)
            
            # Calculate composite scores
            results['overall_uniqueness'] = self.calculate_overall_uniqueness(results)
            results['repetition_confidence'] = self.calculate_repetition_confidence(results)
            results['quality_degradation'] = self.calculate_quality_degradation(results)
            
            # Add quality assessment
            results['quality_assessment'] = self.assess_repetition_quality(results)
            
            # Cache results
            st.session_state['repetition_analysis_cache'][cache_key] = results
            
            # Log analysis
            log_event("repetition_analyzed", {
                "overall_uniqueness": results['overall_uniqueness'],
                "repetition_confidence": results['repetition_confidence']
            }, "repetition_checker")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Repetition analysis failed: {str(e)}")
            return self.get_fallback_repetition_results()
    
    def calculate_token_uniqueness(self, text: str) -> Dict[str, Any]:
        """Calculate token uniqueness metrics"""
        
        try:
            # Tokenize text
            tokens = self.tokenize_text(text)
            
            if not tokens:
                return {'uniqueness_ratio': 1.0, 'total_tokens': 0, 'unique_tokens': 0}
            
            # Count unique tokens
            unique_tokens = set(tokens)
            uniqueness_ratio = len(unique_tokens) / len(tokens)
            
            # Calculate token frequency distribution
            token_counts = Counter(tokens)
            most_common = token_counts.most_common(10)
            
            # Calculate repetition patterns
            repeated_tokens = {token: count for token, count in token_counts.items() if count > 1}
            repetition_density = len(repeated_tokens) / len(unique_tokens) if unique_tokens else 0
            
            return {
                'uniqueness_ratio': uniqueness_ratio,
                'total_tokens': len(tokens),
                'unique_tokens': len(unique_tokens),
                'repeated_tokens': len(repeated_tokens),
                'repetition_density': repetition_density,
                'most_common_tokens': most_common,
                'token_distribution': dict(token_counts)
            }
        
        except Exception as e:
            self.logger.warning(f"Token uniqueness calculation failed: {str(e)}")
            return {'uniqueness_ratio': 0.8, 'total_tokens': 0, 'unique_tokens': 0}
    
    def analyze_ngram_repetition(self, text: str) -> Dict[str, Any]:
        """Analyze n-gram repetition patterns"""
        
        try:
            tokens = self.tokenize_text(text)
            
            if len(tokens) < 2:
                return {'bigram_repetition': 0.0, 'trigram_repetition': 0.0, 'phrase_repetition': 0.0}
            
            # Analyze bigrams (2-grams)
            bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
            bigram_counts = Counter(bigrams)
            bigram_repetition = sum(1 for count in bigram_counts.values() if count > 1) / len(bigrams) if bigrams else 0
            
            # Analyze trigrams (3-grams)
            trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)] if len(tokens) >= 3 else []
            trigram_counts = Counter(trigrams)
            trigram_repetition = sum(1 for count in trigram_counts.values() if count > 1) / len(trigrams) if trigrams else 0
            
            # Analyze longer phrases (4-6 grams)
            phrase_repetitions = []
            for n in range(4, min(7, len(tokens)+1)):
                ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                ngram_counts = Counter(ngrams)
                repetition_rate = sum(1 for count in ngram_counts.values() if count > 1) / len(ngrams) if ngrams else 0
                phrase_repetitions.append(repetition_rate)
            
            phrase_repetition = max(phrase_repetitions) if phrase_repetitions else 0
            
            # Find most repeated n-grams
            repeated_bigrams = {ngram: count for ngram, count in bigram_counts.items() if count > 1}
            repeated_trigrams = {ngram: count for ngram, count in trigram_counts.items() if count > 1}
            
            return {
                'bigram_repetition': bigram_repetition,
                'trigram_repetition': trigram_repetition,
                'phrase_repetition': phrase_repetition,
                'repeated_bigrams': repeated_bigrams,
                'repeated_trigrams': repeated_trigrams,
                'total_bigrams': len(bigrams),
                'total_trigrams': len(trigrams)
            }
        
        except Exception as e:
            self.logger.warning(f"N-gram repetition analysis failed: {str(e)}")
            return {'bigram_repetition': 0.0, 'trigram_repetition': 0.0, 'phrase_repetition': 0.0}
    
    def analyze_filler_content(self, text: str) -> Dict[str, Any]:
        """Analyze filler phrases and unnecessary content"""
        
        try:
            text_lower = text.lower()
            filler_analysis = {}
            total_filler_count = 0
            
            # Analyze each category of filler phrases
            for category, phrases in self.filler_phrases.items():
                category_count = 0
                found_phrases = []
                
                for phrase in phrases:
                    count = text_lower.count(phrase.lower())
                    if count > 0:
                        category_count += count
                        found_phrases.append((phrase, count))
                
                filler_analysis[category] = {
                    'count': category_count,
                    'found_phrases': found_phrases
                }
                total_filler_count += category_count
            
            # Calculate filler density
            word_count = len(text.split())
            filler_density = total_filler_count / word_count if word_count > 0 else 0
            
            # Analyze other filler indicators
            excessive_adjectives = len(re.findall(r'\b(?:very|quite|extremely|incredibly|absolutely)\s+\w+', text_lower))
            excessive_adverbs = len(re.findall(r'\b\w+ly\b', text_lower))
            
            return {
                'total_filler_count': total_filler_count,
                'filler_density': filler_density,
                'category_analysis': filler_analysis,
                'excessive_adjectives': excessive_adjectives,
                'excessive_adverbs': excessive_adverbs,
                'word_count': word_count
            }
        
        except Exception as e:
            self.logger.warning(f"Filler content analysis failed: {str(e)}")
            return {'total_filler_count': 0, 'filler_density': 0.0}
    
    def calculate_content_diversity(self, text: str) -> Dict[str, Any]:
        """Calculate content diversity metrics"""
        
        try:
            tokens = self.tokenize_text(text)
            
            if not tokens:
                return {'lexical_diversity': 1.0, 'semantic_diversity': 1.0}
            
            # Lexical diversity (Type-Token Ratio)
            unique_tokens = set(tokens)
            lexical_diversity = len(unique_tokens) / len(tokens)
            
            # Sentence diversity
            sentences = self.split_sentences(text)
            sentence_lengths = [len(sentence.split()) for sentence in sentences]
            
            # Calculate sentence length diversity
            if len(sentence_lengths) > 1:
                length_variance = np.var(sentence_lengths)
                length_diversity = min(1.0, length_variance / 100)  # Normalize
            else:
                length_diversity = 1.0
            
            # Vocabulary richness
            word_length_diversity = np.var([len(token) for token in tokens]) if len(tokens) > 1 else 0
            vocab_richness = min(1.0, word_length_diversity / 10)  # Normalize
            
            # Semantic diversity (simplified)
            # Count different types of words (nouns, verbs, adjectives, etc.)
            pos_diversity = self.calculate_pos_diversity(tokens)
            
            return {
                'lexical_diversity': lexical_diversity,
                'sentence_length_diversity': length_diversity,
                'vocabulary_richness': vocab_richness,
                'pos_diversity': pos_diversity,
                'total_sentences': len(sentences),
                'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0
            }
        
        except Exception as e:
            self.logger.warning(f"Content diversity calculation failed: {str(e)}")
            return {'lexical_diversity': 0.8, 'semantic_diversity': 0.8}
    
    def calculate_compression_ratio(self, text: str) -> Dict[str, Any]:
        """Calculate content compression characteristics"""
        
        try:
            # Basic compression using simple algorithms
            import zlib
            
            original_size = len(text.encode('utf-8'))
            compressed_size = len(zlib.compress(text.encode('utf-8')))
            
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # Information density
            unique_chars = len(set(text))
            total_chars = len(text)
            char_diversity = unique_chars / total_chars if total_chars > 0 else 1.0
            
            # Repetitive pattern detection
            pattern_score = 1.0 - compression_ratio  # Lower compression = more repetitive
            
            return {
                'compression_ratio': compression_ratio,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'char_diversity': char_diversity,
                'pattern_score': pattern_score,
                'information_density': char_diversity * (1.0 - compression_ratio)
            }
        
        except Exception as e:
            self.logger.warning(f"Compression ratio calculation failed: {str(e)}")
            return {'compression_ratio': 0.7, 'pattern_score': 0.3}
    
    def calculate_redundancy_score(self, text: str) -> Dict[str, Any]:
        """Calculate overall redundancy score"""
        
        try:
            # Combine multiple redundancy indicators
            token_analysis = self.calculate_token_uniqueness(text)
            ngram_analysis = self.analyze_ngram_repetition(text)
            filler_analysis = self.analyze_filler_content(text)
            
            # Calculate component scores
            token_redundancy = 1.0 - token_analysis.get('uniqueness_ratio', 0.8)
            ngram_redundancy = max(
                ngram_analysis.get('bigram_repetition', 0.0),
                ngram_analysis.get('trigram_repetition', 0.0),
                ngram_analysis.get('phrase_repetition', 0.0)
            )
            filler_redundancy = min(1.0, filler_analysis.get('filler_density', 0.0) * 5)  # Scale up
            
            # Weighted redundancy score
            redundancy_score = (
                token_redundancy * 0.4 +
                ngram_redundancy * 0.4 +
                filler_redundancy * 0.2
            )
            
            return {
                'redundancy_score': redundancy_score,
                'token_redundancy': token_redundancy,
                'ngram_redundancy': ngram_redundancy,
                'filler_redundancy': filler_redundancy,
                'redundancy_level': self.classify_redundancy_level(redundancy_score)
            }
        
        except Exception as e:
            self.logger.warning(f"Redundancy score calculation failed: {str(e)}")
            return {'redundancy_score': 0.3, 'redundancy_level': 'moderate'}
    
    def analyze_sentence_similarity(self, text: str) -> Dict[str, Any]:
        """Analyze similarity between sentences"""
        
        try:
            sentences = self.split_sentences(text)
            
            if len(sentences) < 2:
                return {'avg_similarity': 0.0, 'max_similarity': 0.0, 'similar_pairs': []}
            
            # Calculate pairwise sentence similarities
            similarities = []
            similar_pairs = []
            
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)):
                    similarity = self.calculate_sentence_similarity(sentences[i], sentences[j])
                    similarities.append(similarity)
                    
                    if similarity > 0.7:  # High similarity threshold
                        similar_pairs.append((i, j, similarity))
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            max_similarity = max(similarities) if similarities else 0.0
            
            return {
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'similar_pairs': similar_pairs,
                'total_comparisons': len(similarities),
                'high_similarity_count': len(similar_pairs)
            }
        
        except Exception as e:
            self.logger.warning(f"Sentence similarity analysis failed: {str(e)}")
            return {'avg_similarity': 0.0, 'max_similarity': 0.0, 'similar_pairs': []}
    
    def analyze_expansion_quality(self, enhanced_text: str, original_text: str) -> Dict[str, Any]:
        """Analyze quality of text expansion"""
        
        try:
            # Length analysis
            original_length = len(original_text.split())
            enhanced_length = len(enhanced_text.split())
            expansion_ratio = enhanced_length / original_length if original_length > 0 else 1.0
            
            # Content preservation analysis
            original_tokens = set(self.tokenize_text(original_text.lower()))
            enhanced_tokens = set(self.tokenize_text(enhanced_text.lower()))
            
            preserved_content = len(original_tokens.intersection(enhanced_tokens))
            content_preservation = preserved_content / len(original_tokens) if original_tokens else 1.0
            
            # New content analysis
            new_content = enhanced_tokens - original_tokens
            new_content_ratio = len(new_content) / len(enhanced_tokens) if enhanced_tokens else 0.0
            
            # Quality assessment
            if expansion_ratio > 2.0:
                expansion_quality = 'excessive'
            elif expansion_ratio > 1.5:
                expansion_quality = 'significant'
            elif expansion_ratio > 1.2:
                expansion_quality = 'moderate'
            elif expansion_ratio > 1.0:
                expansion_quality = 'minimal'
            else:
                expansion_quality = 'compression'
            
            return {
                'expansion_ratio': expansion_ratio,
                'original_length': original_length,
                'enhanced_length': enhanced_length,
                'content_preservation': content_preservation,
                'new_content_ratio': new_content_ratio,
                'expansion_quality': expansion_quality,
                'preserved_tokens': preserved_content,
                'new_tokens': len(new_content)
            }
        
        except Exception as e:
            self.logger.warning(f"Expansion quality analysis failed: {str(e)}")
            return {'expansion_ratio': 1.0, 'expansion_quality': 'unknown'}
    
    def calculate_content_drift(self, enhanced_text: str, original_text: str) -> float:
        """Calculate content drift between original and enhanced text"""
        
        try:
            # Simple content drift based on token overlap
            original_tokens = set(self.tokenize_text(original_text.lower()))
            enhanced_tokens = set(self.tokenize_text(enhanced_text.lower()))
            
            if not original_tokens:
                return 0.0
            
            overlap = len(original_tokens.intersection(enhanced_tokens))
            drift = 1.0 - (overlap / len(original_tokens))
            
            return drift
        
        except Exception as e:
            self.logger.warning(f"Content drift calculation failed: {str(e)}")
            return 0.3
    
    def calculate_overall_uniqueness(self, results: Dict[str, Any]) -> float:
        """Calculate overall uniqueness score"""
        
        # Collect uniqueness indicators
        token_uniqueness = results.get('token_uniqueness', {}).get('uniqueness_ratio', 0.8)
        
        # Invert repetition scores (lower repetition = higher uniqueness)
        ngram_analysis = results.get('ngram_repetition', {})
        ngram_uniqueness = 1.0 - max(
            ngram_analysis.get('bigram_repetition', 0.0),
            ngram_analysis.get('trigram_repetition', 0.0),
            ngram_analysis.get('phrase_repetition', 0.0)
        )
        
        # Filler analysis (lower filler = higher uniqueness)
        filler_density = results.get('filler_analysis', {}).get('filler_density', 0.0)
        filler_uniqueness = max(0.0, 1.0 - (filler_density * 5))  # Scale filler impact
        
        # Content diversity
        diversity_analysis = results.get('content_diversity', {})
        content_uniqueness = diversity_analysis.get('lexical_diversity', 0.8)
        
        # Weighted combination
        overall_uniqueness = (
            token_uniqueness * 0.3 +
            ngram_uniqueness * 0.3 +
            filler_uniqueness * 0.2 +
            content_uniqueness * 0.2
        )
        
        return overall_uniqueness
    
    def calculate_repetition_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in repetition analysis"""
        
        # Collect confidence indicators
        token_analysis = results.get('token_uniqueness', {})
        total_tokens = token_analysis.get('total_tokens', 0)
        
        # Higher confidence with more tokens
        token_confidence = min(1.0, total_tokens / 100)  # Full confidence at 100+ tokens
        
        # Consistency across different metrics
        uniqueness_scores = [
            results.get('overall_uniqueness', 0.5),
            1.0 - results.get('redundancy_score', {}).get('redundancy_score', 0.5),
            results.get('content_diversity', {}).get('lexical_diversity', 0.5)
        ]
        
        # Calculate variance (lower variance = higher confidence)
        mean_score = sum(uniqueness_scores) / len(uniqueness_scores)
        variance = sum((score - mean_score) ** 2 for score in uniqueness_scores) / len(uniqueness_scores)
        
        consistency_confidence = max(0.0, 1.0 - (variance * 4))  # Scale variance
        
        # Combined confidence
        confidence = (token_confidence * 0.4 + consistency_confidence * 0.6)
        
        return confidence
    
    def calculate_quality_degradation(self, results: Dict[str, Any]) -> float:
        """Calculate quality degradation due to repetition"""
        
        redundancy_score = results.get('redundancy_score', {}).get('redundancy_score', 0.3)
        filler_density = results.get('filler_analysis', {}).get('filler_density', 0.0)
        
        # Quality degradation increases with redundancy and filler
        degradation = (redundancy_score * 0.7) + (min(filler_density * 5, 1.0) * 0.3)
        
        return degradation
    
    def assess_repetition_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall repetition quality"""
        
        overall_score = results.get('overall_uniqueness', 0.5)
        confidence = results.get('repetition_confidence', 0.5)
        
        # Determine quality level
        if overall_score >= self.thresholds['excellent'] and confidence >= 0.8:
            quality_level = 'excellent'
            status = 'pass'
            message = "Excellent content uniqueness"
        elif overall_score >= self.thresholds['good'] and confidence >= 0.6:
            quality_level = 'good'
            status = 'pass'
            message = "Good content uniqueness"
        elif overall_score >= self.thresholds['acceptable']:
            quality_level = 'acceptable'
            status = 'pass'
            message = "Acceptable content uniqueness"
        elif overall_score >= self.thresholds['concerning']:
            quality_level = 'concerning'
            status = 'review'
            message = "Concerning repetition detected"
        else:
            quality_level = 'poor'
            status = 'fail'
            message = "Excessive repetition detected"
        
        # Generate recommendations
        recommendations = self.generate_repetition_recommendations(results)
        
        return {
            'quality_level': quality_level,
            'status': status,
            'message': message,
            'recommendations': recommendations,
            'pass_threshold': overall_score >= self.thresholds['acceptable'],
            'confidence': confidence
        }
    
    def generate_repetition_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for reducing repetition"""
        
        recommendations = []
        
        # Token uniqueness recommendations
        token_analysis = results.get('token_uniqueness', {})
        if token_analysis.get('uniqueness_ratio', 0.8) < 0.7:
            recommendations.append("Increase vocabulary diversity - use more varied word choices")
        
        # N-gram repetition recommendations
        ngram_analysis = results.get('ngram_repetition', {})
        if ngram_analysis.get('phrase_repetition', 0.0) > 0.3:
            recommendations.append("Reduce repeated phrases and sentence structures")
        
        # Filler content recommendations
        filler_analysis = results.get('filler_analysis', {})
        if filler_analysis.get('filler_density', 0.0) > 0.1:
            recommendations.append("Remove filler phrases and unnecessary qualifiers")
            
            # Specific filler recommendations
            category_analysis = filler_analysis.get('category_analysis', {})
            for category, data in category_analysis.items():
                if data.get('count', 0) > 2:
                    recommendations.append(f"Reduce {category.replace('_', ' ')}")
        
        # Redundancy recommendations
        redundancy_analysis = results.get('redundancy_score', {})
        redundancy_level = redundancy_analysis.get('redundancy_level', 'low')
        if redundancy_level in ['high', 'very_high']:
            recommendations.append("Significantly reduce redundant content and repetitive patterns")
        
        # Expansion quality recommendations
        expansion_analysis = results.get('expansion_analysis', {})
        expansion_quality = expansion_analysis.get('expansion_quality', 'unknown')
        if expansion_quality == 'excessive':
            recommendations.append("Content expansion is excessive - focus on quality over quantity")
        
        return recommendations
    
    def classify_redundancy_level(self, redundancy_score: float) -> str:
        """Classify redundancy level based on score"""
        
        if redundancy_score >= 0.8:
            return 'very_high'
        elif redundancy_score >= 0.6:
            return 'high'
        elif redundancy_score >= 0.4:
            return 'moderate'
        elif redundancy_score >= 0.2:
            return 'low'
        else:
            return 'very_low'
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences"""
        
        try:
            # Simple token-based similarity
            tokens1 = set(self.tokenize_text(sent1))
            tokens2 = set(self.tokenize_text(sent2))
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            similarity = intersection / union if union > 0 else 0.0
            return similarity
        
        except Exception as e:
            self.logger.warning(f"Sentence similarity calculation failed: {str(e)}")
            return 0.0
    
    def calculate_pos_diversity(self, tokens: List[str]) -> float:
        """Calculate part-of-speech diversity (simplified)"""
        
        try:
            # Simple heuristic-based POS classification
            pos_counts = defaultdict(int)
            
            for token in tokens:
                if token.endswith('ing'):
                    pos_counts['verb'] += 1
                elif token.endswith('ed'):
                    pos_counts['verb'] += 1
                elif token.endswith('ly'):
                    pos_counts['adverb'] += 1
                elif token.endswith('tion') or token.endswith('sion'):
                    pos_counts['noun'] += 1
                elif len(token) > 6:
                    pos_counts['noun'] += 1
                else:
                    pos_counts['other'] += 1
            
            # Calculate diversity
            total_tokens = sum(pos_counts.values())
            if total_tokens == 0:
                return 1.0
            
            # Shannon diversity index (simplified)
            diversity = 0.0
            for count in pos_counts.values():
                if count > 0:
                    p = count / total_tokens
                    diversity -= p * np.log2(p)
            
            # Normalize to 0-1 scale
            max_diversity = np.log2(len(pos_counts)) if pos_counts else 1.0
            normalized_diversity = diversity / max_diversity if max_diversity > 0 else 1.0
            
            return normalized_diversity
        
        except Exception as e:
            self.logger.warning(f"POS diversity calculation failed: {str(e)}")
            return 0.8
    
    def get_fallback_repetition_results(self) -> Dict[str, Any]:
        """Return fallback results when analysis fails"""
        
        return {
            'token_uniqueness': {'uniqueness_ratio': 0.8, 'total_tokens': 0, 'unique_tokens': 0},
            'ngram_repetition': {'bigram_repetition': 0.0, 'trigram_repetition': 0.0, 'phrase_repetition': 0.0},
            'filler_analysis': {'total_filler_count': 0, 'filler_density': 0.0},
            'content_diversity': {'lexical_diversity': 0.8, 'semantic_diversity': 0.8},
            'compression_ratio': {'compression_ratio': 0.7, 'pattern_score': 0.3},
            'redundancy_score': {'redundancy_score': 0.3, 'redundancy_level': 'moderate'},
            'sentence_similarity': {'avg_similarity': 0.0, 'max_similarity': 0.0, 'similar_pairs': []},
            'overall_uniqueness': 0.7,
            'repetition_confidence': 0.3,
            'quality_degradation': 0.3,
            'quality_assessment': {
                'quality_level': 'unknown',
                'status': 'review',
                'message': 'Repetition analysis failed - manual review required',
                'recommendations': ['Manual repetition review required due to analysis error'],
                'pass_threshold': False,
                'confidence': 0.3
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def render_repetition_analysis(self, text: str, original_text: Optional[str] = None) -> Dict[str, Any]:
        """Render repetition analysis interface"""
        
        st.subheader("🔄 Repetition Analysis")
        
        # Analyze repetition
        with st.spinner("Analyzing repetition and redundancy..."):
            results = self.analyze_repetition(text, original_text)
        
        # Display results
        self.render_repetition_metrics(results)
        self.render_repetition_assessment(results)
        self.render_repetition_recommendations(results)
        
        return results
    
    def render_repetition_metrics(self, results: Dict[str, Any]):
        """Render repetition analysis metrics"""
        
        st.markdown("**🔄 Repetition Metrics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            uniqueness = results.get('overall_uniqueness', 0.5)
            st.metric(
                "Overall Uniqueness",
                f"{uniqueness:.3f}",
                delta=f"{uniqueness - 0.8:.3f}" if uniqueness != 0.8 else None
            )
        
        with col2:
            token_analysis = results.get('token_uniqueness', {})
            token_uniqueness = token_analysis.get('uniqueness_ratio', 0.5)
            st.metric(
                "Token Uniqueness",
                f"{token_uniqueness:.3f}",
                delta=f"{token_uniqueness - 0.8:.3f}" if token_uniqueness != 0.8 else None
            )
        
        with col3:
            redundancy_analysis = results.get('redundancy_score', {})
            redundancy = redundancy_analysis.get('redundancy_score', 0.5)
            st.metric(
                "Redundancy Score",
                f"{redundancy:.3f}",
                delta=f"{redundancy - 0.3:.3f}" if redundancy != 0.3 else None,
                delta_color="inverse"
            )
        
        with col4:
            confidence = results.get('repetition_confidence', 0.5)
            st.metric(
                "Analysis Confidence",
                f"{confidence:.3f}",
                delta=f"{confidence - 0.8:.3f}" if confidence != 0.8 else None
            )
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filler_analysis = results.get('filler_analysis', {})
            filler_density = filler_analysis.get('filler_density', 0.0)
            st.metric(
                "Filler Density",
                f"{filler_density:.3f}",
                delta=f"{filler_density - 0.05:.3f}" if filler_density != 0.05 else None,
                delta_color="inverse"
            )
        
        with col2:
            diversity_analysis = results.get('content_diversity', {})
            lexical_diversity = diversity_analysis.get('lexical_diversity', 0.5)
            st.metric(
                "Lexical Diversity",
                f"{lexical_diversity:.3f}",
                delta=f"{lexical_diversity - 0.7:.3f}" if lexical_diversity != 0.7 else None
            )
        
        with col3:
            compression_analysis = results.get('compression_ratio', {})
            pattern_score = compression_analysis.get('pattern_score', 0.5)
            st.metric(
                "Pattern Uniqueness",
                f"{pattern_score:.3f}",
                delta=f"{pattern_score - 0.7:.3f}" if pattern_score != 0.7 else None
            )
    
    def render_repetition_assessment(self, results: Dict[str, Any]):
        """Render repetition quality assessment"""
        
        assessment = results.get('quality_assessment', {})
        quality_level = assessment.get('quality_level', 'unknown')
        status = assessment.get('status', 'review')
        message = assessment.get('message', 'Unknown status')
        confidence = assessment.get('confidence', 0.5)
        
        st.markdown("**🔄 Repetition Quality Assessment:**")
        
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
    
    def render_repetition_recommendations(self, results: Dict[str, Any]):
        """Render repetition improvement recommendations"""
        
        assessment = results.get('quality_assessment', {})
        recommendations = assessment.get('recommendations', [])
        
        if recommendations:
            st.markdown("**💡 Repetition Improvement Recommendations:**")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations - repetition analysis looks good!")

# Integration function for main app
def analyze_content_repetition(text: str, original_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze content repetition and redundancy
    
    Usage:
    from modules.repetition_checker import analyze_content_repetition
    
    results = analyze_content_repetition(enhanced_text, original_text)
    """
    
    checker = RepetitionChecker()
    return checker.analyze_repetition(text, original_text)

# Quick repetition check
def quick_repetition_check(text: str) -> float:
    """
    Quick repetition check returning uniqueness score
    
    Usage:
    from modules.repetition_checker import quick_repetition_check
    
    score = quick_repetition_check(enhanced_text)
    """
    
    checker = RepetitionChecker()
    results = checker.analyze_repetition(text)
    return results.get('overall_uniqueness', 0.5)

if __name__ == "__main__":
    # Test the repetition checker
    st.set_page_config(page_title="Repetition Checker Test", layout="wide")
    
    st.title("Repetition Checker Test")
    
    # Sample text
    text = st.text_area(
        "Text to Analyze",
        value="This is a sample text. This text contains some repetitive elements. The text is designed to test repetition detection. Some phrases may repeat in this text.",
        height=100
    )
    
    if st.button("Analyze Repetition"):
        checker = RepetitionChecker()
        results = checker.render_repetition_analysis(text)

