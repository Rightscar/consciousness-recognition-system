"""
Tone Consistency Monitor Module
==============================

Monitors and ensures tone consistency in AI-enhanced content.
Prevents tone drift with semantic similarity checking and tone anchors.

Features:
- Tone anchor examples for each spiritual tradition
- Semantic similarity scoring vs tone base examples
- Automatic tone consistency flagging
- Reviewer tone consistency alerts
- Tone drift detection and correction
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False


@dataclass
class ToneAnchor:
    """Tone anchor example for consistency checking"""
    text: str
    tone: str
    embedding: Optional[List[float]] = None
    source: str = "manual"
    confidence: float = 1.0


@dataclass
class ToneConsistencyResult:
    """Result of tone consistency analysis"""
    similarity_score: float
    consistency_flag: str  # "excellent", "good", "warning", "poor"
    closest_anchor: str
    drift_detected: bool
    recommendations: List[str]
    confidence: float


class ToneConsistencyMonitor:
    """Monitor tone consistency in enhanced content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentence transformer model (cached)
        self.model = None
        if SIMILARITY_AVAILABLE:
            try:
                self.model = self._load_similarity_model()
            except Exception as e:
                self.logger.warning(f"Failed to load similarity model: {e}")
        
        # Tone anchors for each spiritual tradition
        self.tone_anchors = self._initialize_tone_anchors()
        
        # Consistency thresholds
        self.thresholds = {
            "excellent": 0.85,
            "good": 0.75,
            "warning": 0.65,
            "poor": 0.0
        }
    
    @st.cache_resource
    def _load_similarity_model(_self):
        """Load sentence transformer model with caching"""
        try:
            # Use a lightweight model for better performance
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except Exception as e:
            _self.logger.error(f"Failed to load similarity model: {e}")
            return None
    
    def _initialize_tone_anchors(self) -> Dict[str, List[ToneAnchor]]:
        """Initialize tone anchor examples for each spiritual tradition"""
        
        anchors = {
            "advaita_vedanta": [
                ToneAnchor(
                    text="The Self is beyond all attributes and descriptions. It is the pure awareness in which all experiences arise and dissolve. Through inquiry into 'Who am I?', one discovers this unchanging reality that was never born and never dies.",
                    tone="advaita_vedanta",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="What you seek is what you are. The seeker and the sought are one. This understanding comes not through accumulation of knowledge but through the removal of ignorance about your true nature.",
                    tone="advaita_vedanta",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="Consciousness is not produced by the brain; rather, the brain appears in consciousness. You are that pure consciousness, untouched by the modifications of the mind.",
                    tone="advaita_vedanta",
                    source="traditional",
                    confidence=1.0
                )
            ],
            
            "zen_buddhism": [
                ToneAnchor(
                    text="Sitting quietly, doing nothing, spring comes, and the grass grows by itself. This is the way of Zen - effortless effort, natural awakening.",
                    tone="zen_buddhism",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="Before enlightenment, chop wood, carry water. After enlightenment, chop wood, carry water. The ordinary mind is the Way.",
                    tone="zen_buddhism",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="What is Buddha? Three pounds of flax. What is the Way? Ordinary mind. These simple words point to the extraordinary within the ordinary.",
                    tone="zen_buddhism",
                    source="traditional",
                    confidence=1.0
                )
            ],
            
            "christian_mysticism": [
                ToneAnchor(
                    text="In the depths of silence, the soul meets its Beloved. Here, in the secret chamber of the heart, divine love transforms all seeking into pure being.",
                    tone="christian_mysticism",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="God is closer to you than you are to yourself. In surrendering the false self, we discover our true identity as beloved children of the Divine.",
                    tone="christian_mysticism",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="The dark night of the soul is not punishment but purification. In this sacred emptiness, divine grace works its mysterious transformation.",
                    tone="christian_mysticism",
                    source="traditional",
                    confidence=1.0
                )
            ],
            
            "sufi_mysticism": [
                ToneAnchor(
                    text="The Beloved is all in all, the lover merely veils Him; the Beloved is all that lives, the lover a dead thing. When love has no object, then love is perfect.",
                    tone="sufi_mysticism",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="In your light I learn how to love. In your beauty, how to make poems. You dance inside my chest where no one sees you, but sometimes I do, and that sight becomes this art, this music, this form.",
                    tone="sufi_mysticism",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="The breezes at dawn have secrets to tell you. Don't go back to sleep! You must ask for what you really want. Don't go back to sleep!",
                    tone="sufi_mysticism",
                    source="traditional",
                    confidence=1.0
                )
            ],
            
            "mindfulness_meditation": [
                ToneAnchor(
                    text="In this moment, there is only this moment. Breathing in, I know I am breathing in. Breathing out, I know I am breathing out. This simple awareness is the foundation of all wisdom.",
                    tone="mindfulness_meditation",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="Observe your thoughts like clouds passing in the sky. They arise, they change, they pass away. You are the vast sky, not the temporary weather.",
                    tone="mindfulness_meditation",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="Each step can be a meditation. Each breath can be a prayer. In mindful awareness, ordinary activities become gateways to presence.",
                    tone="mindfulness_meditation",
                    source="traditional",
                    confidence=1.0
                )
            ],
            
            "universal_wisdom": [
                ToneAnchor(
                    text="Truth is one, paths are many. Whether through devotion, inquiry, service, or meditation, all sincere seekers arrive at the same destination - the recognition of our fundamental unity.",
                    tone="universal_wisdom",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="The light that shines in the depths of your being is the same light that illuminates all existence. This recognition dissolves the illusion of separation.",
                    tone="universal_wisdom",
                    source="traditional",
                    confidence=1.0
                ),
                ToneAnchor(
                    text="Wisdom traditions across cultures point to the same essential truth: love is the fabric of reality, consciousness is its ground, and peace is our natural state.",
                    tone="universal_wisdom",
                    source="traditional",
                    confidence=1.0
                )
            ]
        }
        
        # Generate embeddings for anchors if model is available
        if self.model:
            for tone, anchor_list in anchors.items():
                for anchor in anchor_list:
                    try:
                        embedding = self.model.encode(anchor.text)
                        anchor.embedding = embedding.tolist()
                    except Exception as e:
                        self.logger.warning(f"Failed to generate embedding for {tone} anchor: {e}")
        
        return anchors
    
    def check_tone_consistency(self, 
                             enhanced_text: str, 
                             target_tone: str, 
                             original_text: Optional[str] = None) -> ToneConsistencyResult:
        """Check tone consistency of enhanced text against tone anchors"""
        
        if not self.model or not SIMILARITY_AVAILABLE:
            # Fallback to simple keyword-based checking
            return self._fallback_tone_check(enhanced_text, target_tone)
        
        try:
            # Generate embedding for enhanced text
            enhanced_embedding = self.model.encode(enhanced_text)
            
            # Get tone anchors for target tone
            anchors = self.tone_anchors.get(target_tone, [])
            if not anchors:
                return ToneConsistencyResult(
                    similarity_score=0.0,
                    consistency_flag="warning",
                    closest_anchor="No anchors available",
                    drift_detected=True,
                    recommendations=["Add tone anchors for this tradition"],
                    confidence=0.0
                )
            
            # Calculate similarities to all anchors
            similarities = []
            anchor_texts = []
            
            for anchor in anchors:
                if anchor.embedding:
                    anchor_embedding = np.array(anchor.embedding).reshape(1, -1)
                    enhanced_embedding_reshaped = enhanced_embedding.reshape(1, -1)
                    
                    similarity = cosine_similarity(enhanced_embedding_reshaped, anchor_embedding)[0][0]
                    similarities.append(similarity)
                    anchor_texts.append(anchor.text[:100] + "...")
            
            if not similarities:
                return self._fallback_tone_check(enhanced_text, target_tone)
            
            # Find best match
            max_similarity = max(similarities)
            best_anchor_idx = similarities.index(max_similarity)
            closest_anchor = anchor_texts[best_anchor_idx]
            
            # Determine consistency flag
            consistency_flag = "poor"
            for flag, threshold in sorted(self.thresholds.items(), key=lambda x: x[1], reverse=True):
                if max_similarity >= threshold:
                    consistency_flag = flag
                    break
            
            # Detect drift
            drift_detected = max_similarity < self.thresholds["warning"]
            
            # Generate recommendations
            recommendations = self._generate_tone_recommendations(
                max_similarity, consistency_flag, target_tone, enhanced_text
            )
            
            return ToneConsistencyResult(
                similarity_score=max_similarity,
                consistency_flag=consistency_flag,
                closest_anchor=closest_anchor,
                drift_detected=drift_detected,
                recommendations=recommendations,
                confidence=max_similarity
            )
            
        except Exception as e:
            self.logger.error(f"Error in tone consistency check: {e}")
            return self._fallback_tone_check(enhanced_text, target_tone)
    
    def _fallback_tone_check(self, text: str, target_tone: str) -> ToneConsistencyResult:
        """Fallback tone checking using keyword analysis"""
        
        # Define tone-specific keywords
        tone_keywords = {
            "advaita_vedanta": ["self", "awareness", "consciousness", "inquiry", "being", "reality", "atman", "brahman"],
            "zen_buddhism": ["mind", "ordinary", "sitting", "meditation", "present", "moment", "way", "buddha"],
            "christian_mysticism": ["divine", "god", "soul", "beloved", "grace", "prayer", "sacred", "spirit"],
            "sufi_mysticism": ["beloved", "love", "heart", "divine", "dance", "poetry", "wine", "union"],
            "mindfulness_meditation": ["breath", "awareness", "present", "moment", "mindful", "observation", "meditation"],
            "universal_wisdom": ["truth", "unity", "wisdom", "light", "consciousness", "love", "peace", "universal"]
        }
        
        keywords = tone_keywords.get(target_tone, [])
        text_lower = text.lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        score = min(matches / len(keywords), 1.0) if keywords else 0.5
        
        # Determine flag based on score
        if score >= 0.7:
            flag = "good"
        elif score >= 0.4:
            flag = "warning"
        else:
            flag = "poor"
        
        return ToneConsistencyResult(
            similarity_score=score,
            consistency_flag=flag,
            closest_anchor="Keyword-based analysis",
            drift_detected=score < 0.4,
            recommendations=[f"Consider including more {target_tone.replace('_', ' ')} terminology"],
            confidence=score
        )
    
    def _generate_tone_recommendations(self, 
                                     similarity_score: float, 
                                     consistency_flag: str, 
                                     target_tone: str, 
                                     text: str) -> List[str]:
        """Generate recommendations for improving tone consistency"""
        
        recommendations = []
        
        if consistency_flag == "poor":
            recommendations.extend([
                f"Text shows significant drift from {target_tone.replace('_', ' ')} tradition",
                "Consider revising to better align with traditional language and concepts",
                "Review tone anchor examples for guidance"
            ])
        
        elif consistency_flag == "warning":
            recommendations.extend([
                f"Text partially aligns with {target_tone.replace('_', ' ')} tradition",
                "Consider strengthening traditional terminology and concepts",
                "Review specific phrases that may be causing drift"
            ])
        
        elif consistency_flag == "good":
            recommendations.append("Text shows good alignment with traditional tone")
        
        elif consistency_flag == "excellent":
            recommendations.append("Text excellently maintains traditional tone and language")
        
        # Tone-specific recommendations
        tone_specific = {
            "advaita_vedanta": [
                "Use non-dual language (Self, awareness, being)",
                "Avoid dualistic concepts (seeker/sought separation)",
                "Emphasize direct inquiry and recognition"
            ],
            "zen_buddhism": [
                "Embrace simplicity and directness",
                "Use paradoxical or pointing language",
                "Emphasize ordinary mind and natural way"
            ],
            "christian_mysticism": [
                "Use devotional and relational language",
                "Emphasize divine love and grace",
                "Include concepts of surrender and transformation"
            ],
            "sufi_mysticism": [
                "Use poetic and metaphorical language",
                "Emphasize love, longing, and union",
                "Include imagery of dance, wine, and beloved"
            ],
            "mindfulness_meditation": [
                "Focus on present moment awareness",
                "Use clear, practical language",
                "Emphasize observation and acceptance"
            ],
            "universal_wisdom": [
                "Bridge different traditions respectfully",
                "Emphasize common ground and unity",
                "Use inclusive, non-sectarian language"
            ]
        }
        
        if similarity_score < 0.75 and target_tone in tone_specific:
            recommendations.extend(tone_specific[target_tone])
        
        return recommendations
    
    def batch_tone_analysis(self, enhanced_items: List[Dict[str, Any]], target_tone: str) -> Dict[str, Any]:
        """Analyze tone consistency for a batch of enhanced items"""
        
        results = []
        total_score = 0
        drift_count = 0
        flag_counts = {"excellent": 0, "good": 0, "warning": 0, "poor": 0}
        
        for item in enhanced_items:
            enhanced_text = item.get('enhanced', '')
            original_text = item.get('original', {}).get('content', '')
            
            result = self.check_tone_consistency(enhanced_text, target_tone, original_text)
            results.append({
                "item_id": item.get('id', 'unknown'),
                "result": result
            })
            
            total_score += result.similarity_score
            if result.drift_detected:
                drift_count += 1
            flag_counts[result.consistency_flag] += 1
        
        avg_score = total_score / len(enhanced_items) if enhanced_items else 0
        drift_percentage = (drift_count / len(enhanced_items)) * 100 if enhanced_items else 0
        
        return {
            "total_items": len(enhanced_items),
            "average_similarity": avg_score,
            "drift_percentage": drift_percentage,
            "flag_distribution": flag_counts,
            "items_needing_review": [r for r in results if r["result"].consistency_flag in ["warning", "poor"]],
            "overall_assessment": self._get_overall_assessment(avg_score, drift_percentage),
            "batch_recommendations": self._get_batch_recommendations(flag_counts, drift_percentage)
        }
    
    def _get_overall_assessment(self, avg_score: float, drift_percentage: float) -> str:
        """Get overall assessment for batch analysis"""
        
        if avg_score >= 0.85 and drift_percentage < 10:
            return "Excellent tone consistency across batch"
        elif avg_score >= 0.75 and drift_percentage < 20:
            return "Good tone consistency with minor variations"
        elif avg_score >= 0.65 and drift_percentage < 40:
            return "Moderate tone consistency - review recommended"
        else:
            return "Poor tone consistency - significant review needed"
    
    def _get_batch_recommendations(self, flag_counts: Dict[str, int], drift_percentage: float) -> List[str]:
        """Get recommendations for batch processing"""
        
        recommendations = []
        total_items = sum(flag_counts.values())
        
        if flag_counts["poor"] > total_items * 0.2:
            recommendations.append("High number of poor consistency items - review prompt template")
        
        if flag_counts["warning"] > total_items * 0.3:
            recommendations.append("Many items need tone adjustment - consider batch reprocessing")
        
        if drift_percentage > 30:
            recommendations.append("Significant tone drift detected - strengthen tone anchors in prompts")
        
        if flag_counts["excellent"] > total_items * 0.8:
            recommendations.append("Excellent consistency - current approach is working well")
        
        return recommendations
    
    def render_tone_consistency_dashboard(self, enhanced_items: List[Dict[str, Any]], target_tone: str):
        """Render tone consistency dashboard in Streamlit"""
        
        if not enhanced_items:
            st.info("No enhanced items to analyze")
            return
        
        with st.expander("🎭 Tone Consistency Analysis", expanded=True):
            
            # Run batch analysis
            analysis = self.batch_tone_analysis(enhanced_items, target_tone)
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Similarity", f"{analysis['average_similarity']:.3f}")
            with col2:
                st.metric("Drift %", f"{analysis['drift_percentage']:.1f}%")
            with col3:
                excellent_pct = (analysis['flag_distribution']['excellent'] / analysis['total_items']) * 100
                st.metric("Excellent %", f"{excellent_pct:.1f}%")
            with col4:
                poor_pct = (analysis['flag_distribution']['poor'] / analysis['total_items']) * 100
                st.metric("Poor %", f"{poor_pct:.1f}%")
            
            # Overall assessment
            assessment = analysis['overall_assessment']
            if "Excellent" in assessment:
                st.success(f"✅ {assessment}")
            elif "Good" in assessment:
                st.info(f"ℹ️ {assessment}")
            elif "Moderate" in assessment:
                st.warning(f"⚠️ {assessment}")
            else:
                st.error(f"❌ {assessment}")
            
            # Flag distribution
            st.write("**Consistency Distribution:**")
            flag_data = analysis['flag_distribution']
            for flag, count in flag_data.items():
                percentage = (count / analysis['total_items']) * 100
                st.write(f"- {flag.title()}: {count} items ({percentage:.1f}%)")
            
            # Items needing review
            items_needing_review = analysis['items_needing_review']
            if items_needing_review:
                st.write(f"**Items Needing Review ({len(items_needing_review)}):**")
                for item in items_needing_review[:5]:  # Show first 5
                    result = item['result']
                    st.write(f"- Item {item['item_id']}: {result.consistency_flag} (similarity: {result.similarity_score:.3f})")
                
                if len(items_needing_review) > 5:
                    st.write(f"... and {len(items_needing_review) - 5} more")
            
            # Recommendations
            if analysis['batch_recommendations']:
                st.write("**Recommendations:**")
                for rec in analysis['batch_recommendations']:
                    st.write(f"- {rec}")
            
            # Tone anchor examples
            if st.checkbox("Show Tone Anchor Examples"):
                anchors = self.tone_anchors.get(target_tone, [])
                if anchors:
                    st.write(f"**{target_tone.replace('_', ' ').title()} Tone Anchors:**")
                    for i, anchor in enumerate(anchors, 1):
                        st.write(f"{i}. {anchor.text}")
                else:
                    st.write("No tone anchors available for this tradition")


# Global tone consistency monitor instance
tone_monitor = ToneConsistencyMonitor()


def get_tone_monitor() -> ToneConsistencyMonitor:
    """Get the global tone consistency monitor instance"""
    return tone_monitor

