"""
Semantic Similarity Module
==========================

Compare enhanced vs. original embeddings to detect semantic drift
and ensure the AI enhancement preserves the original meaning.

Features:
- Cosine similarity calculation
- Jaccard overlap analysis
- Semantic drift detection
- Content preservation scoring
- Embedding-based comparison
- Multiple similarity metrics
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from modules.logger import get_logger, log_event

class SemanticSimilarityAnalyzer:
    """
    Comprehensive semantic similarity analysis for content enhancement validation
    
    Features:
    - Multiple similarity metrics (cosine, Jaccard, semantic overlap)
    - Embedding-based comparison with fallback to TF-IDF
    - Semantic drift detection and scoring
    - Content preservation analysis
    - Detailed similarity reporting
    """
    
    def __init__(self):
        self.logger = get_logger("semantic_similarity")
        
        # Initialize TF-IDF vectorizer for fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Similarity thresholds
        self.thresholds = {
            'excellent': 0.90,
            'good': 0.80,
            'acceptable': 0.70,
            'concerning': 0.60,
            'poor': 0.50
        }
        
        # Initialize session state
        if 'similarity_cache' not in st.session_state:
            st.session_state['similarity_cache'] = {}
    
    def calculate_comprehensive_similarity(self, original_text: str, enhanced_text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive similarity metrics between original and enhanced text
        
        Returns:
        - cosine_similarity: Embedding-based cosine similarity
        - jaccard_similarity: Word-level Jaccard overlap
        - semantic_overlap: Semantic concept overlap
        - content_preservation: Overall content preservation score
        - drift_score: Semantic drift indicator (lower is better)
        """
        
        try:
            # Create cache key
            cache_key = f"{hash(original_text)}_{hash(enhanced_text)}"
            
            # Check cache first
            if cache_key in st.session_state['similarity_cache']:
                return st.session_state['similarity_cache'][cache_key]
            
            # Calculate multiple similarity metrics
            results = {
                'cosine_similarity': self.calculate_cosine_similarity(original_text, enhanced_text),
                'jaccard_similarity': self.calculate_jaccard_similarity(original_text, enhanced_text),
                'semantic_overlap': self.calculate_semantic_overlap(original_text, enhanced_text),
                'word_overlap': self.calculate_word_overlap(original_text, enhanced_text),
                'concept_preservation': self.calculate_concept_preservation(original_text, enhanced_text),
                'length_ratio': len(enhanced_text) / len(original_text) if len(original_text) > 0 else 1.0,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Calculate composite scores
            results['content_preservation'] = self.calculate_content_preservation_score(results)
            results['drift_score'] = self.calculate_drift_score(results)
            results['overall_similarity'] = self.calculate_overall_similarity(results)
            
            # Add quality assessment
            results['quality_assessment'] = self.assess_similarity_quality(results)
            
            # Cache results
            st.session_state['similarity_cache'][cache_key] = results
            
            # Log analysis
            log_event("semantic_similarity_calculated", {
                "cosine_similarity": results['cosine_similarity'],
                "overall_similarity": results['overall_similarity'],
                "drift_score": results['drift_score']
            }, "semantic_similarity")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Semantic similarity calculation failed: {str(e)}")
            return self.get_fallback_results()
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors"""
        
        try:
            # Prepare texts
            texts = [text1, text2]
            
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Return similarity between the two texts
            return float(similarity_matrix[0, 1])
        
        except Exception as e:
            self.logger.warning(f"Cosine similarity calculation failed: {str(e)}")
            return 0.5  # Neutral fallback
    
    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity based on word overlap"""
        
        try:
            # Tokenize and normalize
            words1 = set(self.tokenize_text(text1.lower()))
            words2 = set(self.tokenize_text(text2.lower()))
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        
        except Exception as e:
            self.logger.warning(f"Jaccard similarity calculation failed: {str(e)}")
            return 0.0
    
    def calculate_semantic_overlap(self, text1: str, text2: str) -> float:
        """Calculate semantic overlap based on key concepts"""
        
        try:
            # Extract key concepts (nouns, verbs, adjectives)
            concepts1 = self.extract_key_concepts(text1)
            concepts2 = self.extract_key_concepts(text2)
            
            if not concepts1 or not concepts2:
                return 0.5  # Neutral if no concepts found
            
            # Calculate concept overlap
            common_concepts = len(concepts1.intersection(concepts2))
            total_concepts = len(concepts1.union(concepts2))
            
            return common_concepts / total_concepts if total_concepts > 0 else 0.0
        
        except Exception as e:
            self.logger.warning(f"Semantic overlap calculation failed: {str(e)}")
            return 0.5
    
    def calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap percentage"""
        
        try:
            words1 = set(self.tokenize_text(text1.lower()))
            words2 = set(self.tokenize_text(text2.lower()))
            
            if not words1:
                return 0.0
            
            overlap = len(words1.intersection(words2))
            return overlap / len(words1)
        
        except Exception as e:
            self.logger.warning(f"Word overlap calculation failed: {str(e)}")
            return 0.0
    
    def calculate_concept_preservation(self, text1: str, text2: str) -> float:
        """Calculate how well key concepts are preserved"""
        
        try:
            # Extract important concepts from original
            original_concepts = self.extract_important_concepts(text1)
            enhanced_concepts = self.extract_important_concepts(text2)
            
            if not original_concepts:
                return 1.0  # Perfect if no concepts to preserve
            
            # Check how many original concepts are preserved
            preserved = len(original_concepts.intersection(enhanced_concepts))
            return preserved / len(original_concepts)
        
        except Exception as e:
            self.logger.warning(f"Concept preservation calculation failed: {str(e)}")
            return 0.5
    
    def calculate_content_preservation_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall content preservation score"""
        
        # Weighted combination of metrics
        weights = {
            'cosine_similarity': 0.4,
            'semantic_overlap': 0.3,
            'concept_preservation': 0.2,
            'word_overlap': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        
        return score
    
    def calculate_drift_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate semantic drift score (lower is better)"""
        
        # Drift indicators
        cosine_drift = 1.0 - metrics.get('cosine_similarity', 0.5)
        concept_drift = 1.0 - metrics.get('concept_preservation', 0.5)
        length_penalty = max(0, metrics.get('length_ratio', 1.0) - 2.0) * 0.2  # Penalty for excessive length
        
        # Combined drift score
        drift_score = (cosine_drift * 0.5) + (concept_drift * 0.3) + (length_penalty * 0.2)
        
        return min(1.0, drift_score)  # Cap at 1.0
    
    def calculate_overall_similarity(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall similarity score"""
        
        # Primary similarity metrics
        primary_score = (
            metrics.get('cosine_similarity', 0.5) * 0.5 +
            metrics.get('semantic_overlap', 0.5) * 0.3 +
            metrics.get('concept_preservation', 0.5) * 0.2
        )
        
        # Apply penalties for excessive drift
        drift_penalty = metrics.get('drift_score', 0.0) * 0.2
        
        return max(0.0, primary_score - drift_penalty)
    
    def assess_similarity_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of similarity and provide recommendations"""
        
        overall_score = metrics.get('overall_similarity', 0.5)
        drift_score = metrics.get('drift_score', 0.5)
        
        # Determine quality level
        if overall_score >= self.thresholds['excellent']:
            quality_level = 'excellent'
            status = 'pass'
            message = "Excellent semantic preservation"
        elif overall_score >= self.thresholds['good']:
            quality_level = 'good'
            status = 'pass'
            message = "Good semantic preservation"
        elif overall_score >= self.thresholds['acceptable']:
            quality_level = 'acceptable'
            status = 'pass'
            message = "Acceptable semantic preservation"
        elif overall_score >= self.thresholds['concerning']:
            quality_level = 'concerning'
            status = 'review'
            message = "Concerning semantic drift detected"
        else:
            quality_level = 'poor'
            status = 'fail'
            message = "Significant semantic drift detected"
        
        # Generate recommendations
        recommendations = self.generate_recommendations(metrics)
        
        return {
            'quality_level': quality_level,
            'status': status,
            'message': message,
            'recommendations': recommendations,
            'pass_threshold': overall_score >= self.thresholds['acceptable']
        }
    
    def generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on similarity analysis"""
        
        recommendations = []
        
        # Cosine similarity recommendations
        if metrics.get('cosine_similarity', 0.5) < 0.7:
            recommendations.append("Consider reducing the extent of content modification")
        
        # Concept preservation recommendations
        if metrics.get('concept_preservation', 0.5) < 0.8:
            recommendations.append("Ensure key concepts from original content are preserved")
        
        # Length ratio recommendations
        length_ratio = metrics.get('length_ratio', 1.0)
        if length_ratio > 2.0:
            recommendations.append("Content expansion is excessive - consider more concise enhancement")
        elif length_ratio < 0.5:
            recommendations.append("Content may be over-compressed - ensure completeness")
        
        # Drift score recommendations
        if metrics.get('drift_score', 0.0) > 0.4:
            recommendations.append("Significant semantic drift detected - review enhancement approach")
        
        # Jaccard similarity recommendations
        if metrics.get('jaccard_similarity', 0.5) < 0.3:
            recommendations.append("Very low word overlap - verify content relevance")
        
        return recommendations
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        
        # Simple tokenization - remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word for word in text.split() if len(word) > 2]
    
    def extract_key_concepts(self, text: str) -> set:
        """Extract key concepts from text (simplified approach)"""
        
        # For now, use important words (longer than 4 characters)
        words = self.tokenize_text(text.lower())
        concepts = set()
        
        for word in words:
            if len(word) > 4 and word not in ['would', 'could', 'should', 'might', 'about', 'where', 'there', 'their']:
                concepts.add(word)
        
        return concepts
    
    def extract_important_concepts(self, text: str) -> set:
        """Extract important concepts that should be preserved"""
        
        # Extract longer, more meaningful words
        words = self.tokenize_text(text.lower())
        important = set()
        
        for word in words:
            if len(word) > 5:  # Longer words are typically more important
                important.add(word)
        
        return important
    
    def get_fallback_results(self) -> Dict[str, Any]:
        """Return fallback results when calculation fails"""
        
        return {
            'cosine_similarity': 0.5,
            'jaccard_similarity': 0.5,
            'semantic_overlap': 0.5,
            'word_overlap': 0.5,
            'concept_preservation': 0.5,
            'length_ratio': 1.0,
            'content_preservation': 0.5,
            'drift_score': 0.5,
            'overall_similarity': 0.5,
            'quality_assessment': {
                'quality_level': 'unknown',
                'status': 'review',
                'message': 'Similarity calculation failed - manual review required',
                'recommendations': ['Manual review required due to calculation error'],
                'pass_threshold': False
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def render_similarity_analysis(self, original_text: str, enhanced_text: str) -> Dict[str, Any]:
        """Render similarity analysis interface"""
        
        st.subheader("🔍 Semantic Similarity Analysis")
        
        # Calculate similarity
        with st.spinner("Analyzing semantic similarity..."):
            results = self.calculate_comprehensive_similarity(original_text, enhanced_text)
        
        # Display results
        self.render_similarity_metrics(results)
        self.render_similarity_assessment(results)
        self.render_similarity_recommendations(results)
        
        return results
    
    def render_similarity_metrics(self, results: Dict[str, Any]):
        """Render similarity metrics display"""
        
        st.markdown("**📊 Similarity Metrics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cosine_sim = results.get('cosine_similarity', 0.5)
            st.metric(
                "Cosine Similarity",
                f"{cosine_sim:.3f}",
                delta=f"{cosine_sim - 0.8:.3f}" if cosine_sim != 0.8 else None
            )
        
        with col2:
            jaccard_sim = results.get('jaccard_similarity', 0.5)
            st.metric(
                "Word Overlap",
                f"{jaccard_sim:.3f}",
                delta=f"{jaccard_sim - 0.6:.3f}" if jaccard_sim != 0.6 else None
            )
        
        with col3:
            concept_pres = results.get('concept_preservation', 0.5)
            st.metric(
                "Concept Preservation",
                f"{concept_pres:.3f}",
                delta=f"{concept_pres - 0.8:.3f}" if concept_pres != 0.8 else None
            )
        
        with col4:
            overall_sim = results.get('overall_similarity', 0.5)
            st.metric(
                "Overall Similarity",
                f"{overall_sim:.3f}",
                delta=f"{overall_sim - 0.75:.3f}" if overall_sim != 0.75 else None
            )
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            drift_score = results.get('drift_score', 0.5)
            st.metric(
                "Drift Score",
                f"{drift_score:.3f}",
                delta=f"{0.2 - drift_score:.3f}",  # Lower is better
                delta_color="inverse"
            )
        
        with col2:
            length_ratio = results.get('length_ratio', 1.0)
            st.metric(
                "Length Ratio",
                f"{length_ratio:.2f}x",
                delta=f"{length_ratio - 1.5:.2f}" if length_ratio != 1.5 else None
            )
        
        with col3:
            semantic_overlap = results.get('semantic_overlap', 0.5)
            st.metric(
                "Semantic Overlap",
                f"{semantic_overlap:.3f}",
                delta=f"{semantic_overlap - 0.7:.3f}" if semantic_overlap != 0.7 else None
            )
    
    def render_similarity_assessment(self, results: Dict[str, Any]):
        """Render similarity quality assessment"""
        
        assessment = results.get('quality_assessment', {})
        quality_level = assessment.get('quality_level', 'unknown')
        status = assessment.get('status', 'review')
        message = assessment.get('message', 'Unknown status')
        
        st.markdown("**🎯 Quality Assessment:**")
        
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
        
        # Quality level indicator
        quality_colors = {
            'excellent': '🟢',
            'good': '🟡',
            'acceptable': '🟠',
            'concerning': '🔴',
            'poor': '⚫',
            'unknown': '⚪'
        }
        
        st.write(f"**Quality Level:** {quality_colors.get(quality_level, '⚪')} {quality_level.title()}")
    
    def render_similarity_recommendations(self, results: Dict[str, Any]):
        """Render similarity recommendations"""
        
        assessment = results.get('quality_assessment', {})
        recommendations = assessment.get('recommendations', [])
        
        if recommendations:
            st.markdown("**💡 Recommendations:**")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations - similarity analysis looks good!")

# Integration function for main app
def analyze_semantic_similarity(original_text: str, enhanced_text: str) -> Dict[str, Any]:
    """
    Analyze semantic similarity between original and enhanced text
    
    Usage:
    from modules.semantic_similarity import analyze_semantic_similarity
    
    results = analyze_semantic_similarity(original, enhanced)
    """
    
    analyzer = SemanticSimilarityAnalyzer()
    return analyzer.calculate_comprehensive_similarity(original_text, enhanced_text)

# Quick similarity check
def quick_similarity_check(original_text: str, enhanced_text: str) -> float:
    """
    Quick similarity check returning overall score
    
    Usage:
    from modules.semantic_similarity import quick_similarity_check
    
    score = quick_similarity_check(original, enhanced)
    """
    
    analyzer = SemanticSimilarityAnalyzer()
    results = analyzer.calculate_comprehensive_similarity(original_text, enhanced_text)
    return results.get('overall_similarity', 0.5)

if __name__ == "__main__":
    # Test the semantic similarity analyzer
    st.set_page_config(page_title="Semantic Similarity Test", layout="wide")
    
    st.title("Semantic Similarity Analyzer Test")
    
    # Sample texts
    original = st.text_area(
        "Original Text",
        value="What is artificial intelligence? AI is a field of computer science that aims to create machines capable of intelligent behavior.",
        height=100
    )
    
    enhanced = st.text_area(
        "Enhanced Text", 
        value="Artificial intelligence represents a transformative field within computer science, dedicated to developing sophisticated machines and systems that can exhibit intelligent behavior, learn from experience, and make autonomous decisions.",
        height=100
    )
    
    if st.button("Analyze Similarity"):
        analyzer = SemanticSimilarityAnalyzer()
        results = analyzer.render_similarity_analysis(original, enhanced)

