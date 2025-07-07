"""
Tone Alignment Module
====================

Check tone vs. style reference to ensure enhanced content matches
the selected spiritual school or writing style.

Features:
- Tone reference comparison
- Embedding distance calculation
- Style consistency analysis
- Spiritual tone validation
- Multi-dimensional tone scoring
- Reference-based alignment
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from modules.logger import get_logger, log_event

class ToneAlignmentAnalyzer:
    """
    Comprehensive tone alignment analysis for content enhancement validation
    
    Features:
    - Reference tone comparison using embeddings
    - Multi-dimensional tone analysis
    - Spiritual style validation
    - Tone consistency scoring
    - Style drift detection
    - Detailed alignment reporting
    """
    
    def __init__(self):
        self.logger = get_logger("tone_alignment")
        
        # Initialize TF-IDF vectorizer for tone analysis
        self.tone_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True,
            analyzer='word'
        )
        
        # Tone alignment thresholds
        self.thresholds = {
            'excellent': 0.85,
            'good': 0.75,
            'acceptable': 0.65,
            'concerning': 0.55,
            'poor': 0.45
        }
        
        # Initialize session state
        if 'tone_references' not in st.session_state:
            st.session_state['tone_references'] = {}
        
        if 'tone_analysis_cache' not in st.session_state:
            st.session_state['tone_analysis_cache'] = {}
        
        # Load tone references
        self.load_tone_references()
        
        # Tone characteristics for different spiritual schools
        self.tone_characteristics = {
            'advaita_vedanta': {
                'keywords': ['awareness', 'consciousness', 'self', 'reality', 'truth', 'being', 'witness', 'absolute', 'brahman', 'atman'],
                'style_markers': ['inquiry', 'questioning', 'direct', 'simple', 'profound'],
                'avoid_words': ['believe', 'faith', 'worship', 'ritual', 'ceremony'],
                'tone_qualities': ['contemplative', 'direct', 'philosophical', 'non-dual']
            },
            'zen_buddhism': {
                'keywords': ['mindfulness', 'present', 'moment', 'emptiness', 'compassion', 'meditation', 'zazen', 'koan'],
                'style_markers': ['simple', 'direct', 'paradoxical', 'minimal', 'practical'],
                'avoid_words': ['complex', 'elaborate', 'theoretical', 'academic'],
                'tone_qualities': ['simple', 'direct', 'paradoxical', 'peaceful']
            },
            'sufi_mysticism': {
                'keywords': ['love', 'heart', 'divine', 'beloved', 'soul', 'ecstasy', 'union', 'whirling', 'poetry'],
                'style_markers': ['poetic', 'metaphorical', 'passionate', 'devotional', 'mystical'],
                'avoid_words': ['dry', 'academic', 'cold', 'analytical'],
                'tone_qualities': ['passionate', 'poetic', 'devotional', 'mystical']
            },
            'christian_mysticism': {
                'keywords': ['prayer', 'contemplation', 'divine', 'grace', 'spirit', 'communion', 'sacred', 'holy'],
                'style_markers': ['reverent', 'humble', 'devotional', 'contemplative', 'sacred'],
                'avoid_words': ['casual', 'irreverent', 'secular', 'mundane'],
                'tone_qualities': ['reverent', 'humble', 'contemplative', 'sacred']
            },
            'mindfulness_meditation': {
                'keywords': ['awareness', 'present', 'breath', 'body', 'thoughts', 'feelings', 'observation', 'acceptance'],
                'style_markers': ['gentle', 'non-judgmental', 'practical', 'accessible', 'grounded'],
                'avoid_words': ['harsh', 'judgmental', 'complex', 'esoteric'],
                'tone_qualities': ['gentle', 'accepting', 'practical', 'grounded']
            },
            'universal_wisdom': {
                'keywords': ['wisdom', 'truth', 'understanding', 'compassion', 'love', 'peace', 'harmony', 'unity'],
                'style_markers': ['inclusive', 'universal', 'balanced', 'respectful', 'integrative'],
                'avoid_words': ['exclusive', 'dogmatic', 'sectarian', 'divisive'],
                'tone_qualities': ['inclusive', 'balanced', 'respectful', 'universal']
            }
        }
    
    def load_tone_references(self):
        """Load tone reference texts from prompts directory"""
        
        try:
            prompts_dir = "prompts"
            if os.path.exists(prompts_dir):
                for filename in os.listdir(prompts_dir):
                    if filename.endswith('.txt'):
                        tone_name = filename.replace('.txt', '')
                        file_path = os.path.join(prompts_dir, filename)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                st.session_state['tone_references'][tone_name] = content
                        except Exception as e:
                            self.logger.warning(f"Failed to load tone reference {filename}: {str(e)}")
        
        except Exception as e:
            self.logger.warning(f"Failed to load tone references: {str(e)}")
    
    def analyze_tone_alignment(self, text: str, target_tone: str, reference_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze tone alignment between text and target tone
        
        Args:
            text: Text to analyze
            target_tone: Target tone/style (e.g., 'zen_buddhism', 'sufi_mysticism')
            reference_text: Optional reference text for comparison
        
        Returns:
            Comprehensive tone alignment analysis
        """
        
        try:
            # Create cache key
            cache_key = f"{hash(text)}_{target_tone}_{hash(reference_text or '')}"
            
            # Check cache
            if cache_key in st.session_state['tone_analysis_cache']:
                return st.session_state['tone_analysis_cache'][cache_key]
            
            # Get reference text
            if not reference_text:
                reference_text = st.session_state['tone_references'].get(target_tone, '')
            
            # Calculate alignment metrics
            results = {
                'target_tone': target_tone,
                'reference_similarity': self.calculate_reference_similarity(text, reference_text),
                'keyword_alignment': self.calculate_keyword_alignment(text, target_tone),
                'style_consistency': self.calculate_style_consistency(text, target_tone),
                'tone_quality_match': self.calculate_tone_quality_match(text, target_tone),
                'avoid_words_check': self.check_avoid_words(text, target_tone),
                'length_appropriateness': self.assess_length_appropriateness(text, target_tone),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Calculate composite scores
            results['overall_alignment'] = self.calculate_overall_alignment(results)
            results['alignment_confidence'] = self.calculate_alignment_confidence(results)
            results['tone_drift'] = self.calculate_tone_drift(results)
            
            # Add quality assessment
            results['quality_assessment'] = self.assess_tone_quality(results)
            
            # Cache results
            st.session_state['tone_analysis_cache'][cache_key] = results
            
            # Log analysis
            log_event("tone_alignment_analyzed", {
                "target_tone": target_tone,
                "overall_alignment": results['overall_alignment'],
                "alignment_confidence": results['alignment_confidence']
            }, "tone_alignment")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Tone alignment analysis failed: {str(e)}")
            return self.get_fallback_tone_results(target_tone)
    
    def calculate_reference_similarity(self, text: str, reference_text: str) -> float:
        """Calculate similarity to reference tone text"""
        
        if not reference_text or not text:
            return 0.5
        
        try:
            # Use TF-IDF to compare texts
            texts = [text, reference_text]
            tfidf_matrix = self.tone_vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return float(similarity_matrix[0, 1])
        
        except Exception as e:
            self.logger.warning(f"Reference similarity calculation failed: {str(e)}")
            return 0.5
    
    def calculate_keyword_alignment(self, text: str, target_tone: str) -> float:
        """Calculate alignment based on tone-specific keywords"""
        
        if target_tone not in self.tone_characteristics:
            return 0.5
        
        try:
            characteristics = self.tone_characteristics[target_tone]
            keywords = characteristics.get('keywords', [])
            
            if not keywords:
                return 0.5
            
            # Normalize text
            text_lower = text.lower()
            
            # Count keyword matches
            matches = 0
            for keyword in keywords:
                if keyword in text_lower:
                    matches += 1
            
            # Calculate alignment score
            alignment = matches / len(keywords)
            
            # Boost score if multiple keywords appear
            if matches > len(keywords) * 0.3:  # More than 30% of keywords
                alignment = min(1.0, alignment * 1.2)
            
            return alignment
        
        except Exception as e:
            self.logger.warning(f"Keyword alignment calculation failed: {str(e)}")
            return 0.5
    
    def calculate_style_consistency(self, text: str, target_tone: str) -> float:
        """Calculate style consistency with target tone"""
        
        if target_tone not in self.tone_characteristics:
            return 0.5
        
        try:
            characteristics = self.tone_characteristics[target_tone]
            style_markers = characteristics.get('style_markers', [])
            
            if not style_markers:
                return 0.5
            
            # Analyze text for style markers
            text_lower = text.lower()
            style_score = 0.0
            
            # Check for style indicators
            for marker in style_markers:
                if marker == 'simple':
                    # Check for simple language (shorter sentences, common words)
                    sentences = text.split('.')
                    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
                    if avg_sentence_length < 20:  # Simple sentences
                        style_score += 0.2
                
                elif marker == 'direct':
                    # Check for direct language (active voice, clear statements)
                    if any(word in text_lower for word in ['is', 'are', 'this', 'that', 'simply', 'clearly']):
                        style_score += 0.2
                
                elif marker == 'poetic':
                    # Check for poetic language (metaphors, imagery)
                    if any(word in text_lower for word in ['like', 'as', 'flows', 'dance', 'song', 'beauty']):
                        style_score += 0.2
                
                elif marker == 'contemplative':
                    # Check for contemplative language (questions, reflection)
                    if '?' in text or any(word in text_lower for word in ['consider', 'reflect', 'ponder', 'contemplate']):
                        style_score += 0.2
                
                elif marker == 'practical':
                    # Check for practical language (instructions, steps)
                    if any(word in text_lower for word in ['practice', 'try', 'begin', 'start', 'step', 'method']):
                        style_score += 0.2
            
            return min(1.0, style_score)
        
        except Exception as e:
            self.logger.warning(f"Style consistency calculation failed: {str(e)}")
            return 0.5
    
    def calculate_tone_quality_match(self, text: str, target_tone: str) -> float:
        """Calculate how well text matches tone qualities"""
        
        if target_tone not in self.tone_characteristics:
            return 0.5
        
        try:
            characteristics = self.tone_characteristics[target_tone]
            tone_qualities = characteristics.get('tone_qualities', [])
            
            if not tone_qualities:
                return 0.5
            
            text_lower = text.lower()
            quality_score = 0.0
            
            # Analyze text for tone qualities
            for quality in tone_qualities:
                if quality == 'gentle':
                    # Check for gentle language
                    if any(word in text_lower for word in ['gently', 'softly', 'kindly', 'peacefully', 'calmly']):
                        quality_score += 0.25
                
                elif quality == 'direct':
                    # Check for direct communication
                    if not any(word in text_lower for word in ['perhaps', 'maybe', 'might', 'possibly']):
                        quality_score += 0.25
                
                elif quality == 'passionate':
                    # Check for passionate language
                    if any(word in text_lower for word in ['love', 'heart', 'soul', 'deeply', 'profound']):
                        quality_score += 0.25
                
                elif quality == 'humble':
                    # Check for humble language
                    if any(word in text_lower for word in ['humbly', 'simply', 'may', 'perhaps', 'invitation']):
                        quality_score += 0.25
                
                elif quality == 'inclusive':
                    # Check for inclusive language
                    if any(word in text_lower for word in ['all', 'everyone', 'together', 'unity', 'universal']):
                        quality_score += 0.25
            
            return min(1.0, quality_score)
        
        except Exception as e:
            self.logger.warning(f"Tone quality match calculation failed: {str(e)}")
            return 0.5
    
    def check_avoid_words(self, text: str, target_tone: str) -> float:
        """Check for words that should be avoided for the target tone"""
        
        if target_tone not in self.tone_characteristics:
            return 1.0  # No penalty if no avoid words defined
        
        try:
            characteristics = self.tone_characteristics[target_tone]
            avoid_words = characteristics.get('avoid_words', [])
            
            if not avoid_words:
                return 1.0
            
            text_lower = text.lower()
            violations = 0
            
            for word in avoid_words:
                if word in text_lower:
                    violations += 1
            
            # Calculate penalty (1.0 = no violations, 0.0 = many violations)
            penalty = max(0.0, 1.0 - (violations / len(avoid_words)))
            
            return penalty
        
        except Exception as e:
            self.logger.warning(f"Avoid words check failed: {str(e)}")
            return 1.0
    
    def assess_length_appropriateness(self, text: str, target_tone: str) -> float:
        """Assess if text length is appropriate for the tone"""
        
        try:
            word_count = len(text.split())
            
            # Different tones have different ideal lengths
            if target_tone == 'zen_buddhism':
                # Zen prefers brevity
                if word_count < 50:
                    return 1.0
                elif word_count < 100:
                    return 0.8
                else:
                    return 0.6
            
            elif target_tone == 'sufi_mysticism':
                # Sufi can be more elaborate
                if 50 <= word_count <= 200:
                    return 1.0
                elif word_count < 300:
                    return 0.8
                else:
                    return 0.6
            
            else:
                # General appropriateness
                if 30 <= word_count <= 150:
                    return 1.0
                elif word_count < 250:
                    return 0.8
                else:
                    return 0.6
        
        except Exception as e:
            self.logger.warning(f"Length appropriateness assessment failed: {str(e)}")
            return 0.8
    
    def calculate_overall_alignment(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall tone alignment score"""
        
        # Weighted combination of metrics
        weights = {
            'reference_similarity': 0.3,
            'keyword_alignment': 0.25,
            'style_consistency': 0.2,
            'tone_quality_match': 0.15,
            'avoid_words_check': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        
        return score
    
    def calculate_alignment_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate confidence in the alignment assessment"""
        
        # Higher confidence when multiple metrics agree
        scores = [
            metrics.get('reference_similarity', 0.5),
            metrics.get('keyword_alignment', 0.5),
            metrics.get('style_consistency', 0.5),
            metrics.get('tone_quality_match', 0.5)
        ]
        
        # Calculate variance (lower variance = higher confidence)
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Convert variance to confidence (0-1 scale)
        confidence = max(0.0, 1.0 - (variance * 4))  # Scale variance
        
        return confidence
    
    def calculate_tone_drift(self, metrics: Dict[str, Any]) -> float:
        """Calculate tone drift score (lower is better)"""
        
        # Drift indicators
        alignment_drift = 1.0 - metrics.get('overall_alignment', 0.5)
        avoid_words_penalty = 1.0 - metrics.get('avoid_words_check', 1.0)
        
        # Combined drift score
        drift_score = (alignment_drift * 0.7) + (avoid_words_penalty * 0.3)
        
        return drift_score
    
    def assess_tone_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of tone alignment"""
        
        overall_score = metrics.get('overall_alignment', 0.5)
        confidence = metrics.get('alignment_confidence', 0.5)
        
        # Determine quality level
        if overall_score >= self.thresholds['excellent'] and confidence >= 0.8:
            quality_level = 'excellent'
            status = 'pass'
            message = "Excellent tone alignment"
        elif overall_score >= self.thresholds['good'] and confidence >= 0.6:
            quality_level = 'good'
            status = 'pass'
            message = "Good tone alignment"
        elif overall_score >= self.thresholds['acceptable']:
            quality_level = 'acceptable'
            status = 'pass'
            message = "Acceptable tone alignment"
        elif overall_score >= self.thresholds['concerning']:
            quality_level = 'concerning'
            status = 'review'
            message = "Concerning tone misalignment"
        else:
            quality_level = 'poor'
            status = 'fail'
            message = "Poor tone alignment"
        
        # Generate recommendations
        recommendations = self.generate_tone_recommendations(metrics)
        
        return {
            'quality_level': quality_level,
            'status': status,
            'message': message,
            'recommendations': recommendations,
            'pass_threshold': overall_score >= self.thresholds['acceptable'],
            'confidence': confidence
        }
    
    def generate_tone_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for tone improvement"""
        
        recommendations = []
        target_tone = metrics.get('target_tone', 'unknown')
        
        # Reference similarity recommendations
        if metrics.get('reference_similarity', 0.5) < 0.6:
            recommendations.append(f"Study reference examples of {target_tone.replace('_', ' ')} style")
        
        # Keyword alignment recommendations
        if metrics.get('keyword_alignment', 0.5) < 0.4:
            if target_tone in self.tone_characteristics:
                keywords = self.tone_characteristics[target_tone].get('keywords', [])
                recommendations.append(f"Include more {target_tone.replace('_', ' ')} keywords: {', '.join(keywords[:3])}")
        
        # Style consistency recommendations
        if metrics.get('style_consistency', 0.5) < 0.6:
            if target_tone in self.tone_characteristics:
                style_markers = self.tone_characteristics[target_tone].get('style_markers', [])
                recommendations.append(f"Adopt {target_tone.replace('_', ' ')} style: {', '.join(style_markers[:3])}")
        
        # Avoid words recommendations
        if metrics.get('avoid_words_check', 1.0) < 0.8:
            if target_tone in self.tone_characteristics:
                avoid_words = self.tone_characteristics[target_tone].get('avoid_words', [])
                recommendations.append(f"Avoid words that conflict with {target_tone.replace('_', ' ')}: {', '.join(avoid_words[:3])}")
        
        # Length recommendations
        if metrics.get('length_appropriateness', 0.8) < 0.7:
            if target_tone == 'zen_buddhism':
                recommendations.append("Consider more concise, direct expression (Zen style)")
            else:
                recommendations.append("Adjust content length to match tone expectations")
        
        return recommendations
    
    def get_fallback_tone_results(self, target_tone: str) -> Dict[str, Any]:
        """Return fallback results when analysis fails"""
        
        return {
            'target_tone': target_tone,
            'reference_similarity': 0.5,
            'keyword_alignment': 0.5,
            'style_consistency': 0.5,
            'tone_quality_match': 0.5,
            'avoid_words_check': 1.0,
            'length_appropriateness': 0.8,
            'overall_alignment': 0.5,
            'alignment_confidence': 0.3,
            'tone_drift': 0.5,
            'quality_assessment': {
                'quality_level': 'unknown',
                'status': 'review',
                'message': 'Tone analysis failed - manual review required',
                'recommendations': ['Manual tone review required due to analysis error'],
                'pass_threshold': False,
                'confidence': 0.3
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def render_tone_analysis(self, text: str, target_tone: str, reference_text: Optional[str] = None) -> Dict[str, Any]:
        """Render tone alignment analysis interface"""
        
        st.subheader("🎭 Tone Alignment Analysis")
        
        # Calculate alignment
        with st.spinner("Analyzing tone alignment..."):
            results = self.analyze_tone_alignment(text, target_tone, reference_text)
        
        # Display results
        self.render_tone_metrics(results)
        self.render_tone_assessment(results)
        self.render_tone_recommendations(results)
        
        return results
    
    def render_tone_metrics(self, results: Dict[str, Any]):
        """Render tone alignment metrics"""
        
        st.markdown("**🎯 Tone Alignment Metrics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ref_sim = results.get('reference_similarity', 0.5)
            st.metric(
                "Reference Similarity",
                f"{ref_sim:.3f}",
                delta=f"{ref_sim - 0.7:.3f}" if ref_sim != 0.7 else None
            )
        
        with col2:
            keyword_align = results.get('keyword_alignment', 0.5)
            st.metric(
                "Keyword Alignment",
                f"{keyword_align:.3f}",
                delta=f"{keyword_align - 0.6:.3f}" if keyword_align != 0.6 else None
            )
        
        with col3:
            style_consist = results.get('style_consistency', 0.5)
            st.metric(
                "Style Consistency",
                f"{style_consist:.3f}",
                delta=f"{style_consist - 0.7:.3f}" if style_consist != 0.7 else None
            )
        
        with col4:
            overall_align = results.get('overall_alignment', 0.5)
            st.metric(
                "Overall Alignment",
                f"{overall_align:.3f}",
                delta=f"{overall_align - 0.75:.3f}" if overall_align != 0.75 else None
            )
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tone_quality = results.get('tone_quality_match', 0.5)
            st.metric(
                "Tone Quality Match",
                f"{tone_quality:.3f}",
                delta=f"{tone_quality - 0.7:.3f}" if tone_quality != 0.7 else None
            )
        
        with col2:
            avoid_check = results.get('avoid_words_check', 1.0)
            st.metric(
                "Avoid Words Check",
                f"{avoid_check:.3f}",
                delta=f"{avoid_check - 1.0:.3f}" if avoid_check != 1.0 else None
            )
        
        with col3:
            confidence = results.get('alignment_confidence', 0.5)
            st.metric(
                "Confidence",
                f"{confidence:.3f}",
                delta=f"{confidence - 0.8:.3f}" if confidence != 0.8 else None
            )
    
    def render_tone_assessment(self, results: Dict[str, Any]):
        """Render tone quality assessment"""
        
        assessment = results.get('quality_assessment', {})
        quality_level = assessment.get('quality_level', 'unknown')
        status = assessment.get('status', 'review')
        message = assessment.get('message', 'Unknown status')
        confidence = assessment.get('confidence', 0.5)
        
        st.markdown("**🎭 Tone Quality Assessment:**")
        
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
    
    def render_tone_recommendations(self, results: Dict[str, Any]):
        """Render tone improvement recommendations"""
        
        assessment = results.get('quality_assessment', {})
        recommendations = assessment.get('recommendations', [])
        
        if recommendations:
            st.markdown("**💡 Tone Improvement Recommendations:**")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations - tone alignment looks good!")

# Integration function for main app
def analyze_tone_alignment(text: str, target_tone: str, reference_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze tone alignment for enhanced text
    
    Usage:
    from modules.tone_alignment import analyze_tone_alignment
    
    results = analyze_tone_alignment(enhanced_text, 'zen_buddhism')
    """
    
    analyzer = ToneAlignmentAnalyzer()
    return analyzer.analyze_tone_alignment(text, target_tone, reference_text)

# Quick tone check
def quick_tone_check(text: str, target_tone: str) -> float:
    """
    Quick tone alignment check returning overall score
    
    Usage:
    from modules.tone_alignment import quick_tone_check
    
    score = quick_tone_check(enhanced_text, 'sufi_mysticism')
    """
    
    analyzer = ToneAlignmentAnalyzer()
    results = analyzer.analyze_tone_alignment(text, target_tone)
    return results.get('overall_alignment', 0.5)

if __name__ == "__main__":
    # Test the tone alignment analyzer
    st.set_page_config(page_title="Tone Alignment Test", layout="wide")
    
    st.title("Tone Alignment Analyzer Test")
    
    # Sample text
    text = st.text_area(
        "Text to Analyze",
        value="In the stillness of meditation, we discover the profound simplicity of being. Each breath becomes a teacher, each moment an invitation to deeper awareness.",
        height=100
    )
    
    # Target tone selection
    target_tone = st.selectbox(
        "Target Tone",
        ['zen_buddhism', 'sufi_mysticism', 'advaita_vedanta', 'christian_mysticism', 'mindfulness_meditation', 'universal_wisdom']
    )
    
    if st.button("Analyze Tone Alignment"):
        analyzer = ToneAlignmentAnalyzer()
        results = analyzer.render_tone_analysis(text, target_tone)

