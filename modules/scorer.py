"""
Consciousness Recognition Scorer

Scores spiritual dialogues based on consciousness recognition criteria.
"""

import re
from enum import Enum
from typing import Dict, List, Any, Optional


class DialogueMode(Enum):
    """Dialogue mode classification."""
    CONSCIOUSNESS = "consciousness"
    INQUIRY = "inquiry"
    TEACHING = "teaching"
    MIXED = "mixed"


class ConsciousnessScorer:
    """Scores dialogues for consciousness recognition quality."""
    
    def __init__(self):
        """Initialize the consciousness scorer."""
        self.direct_pointing_phrases = [
            "you are awareness", "remain as you are", "you are that",
            "be as you are", "you are consciousness", "what you are",
            "your true nature", "pure awareness", "just be",
            "stay as you are", "abide as you are"
        ]
        
        self.non_dual_phrases = [
            "no seeker", "not two", "one consciousness", "no separation",
            "no individual", "no person", "no doer", "no experiencer",
            "only consciousness", "pure being", "undivided"
        ]
        
        self.disidentification_phrases = [
            "not the body", "not the mind", "not thoughts", "not emotions",
            "ego is illusion", "false identity", "not the person",
            "beyond the body", "prior to thoughts"
        ]
        
        self.presence_phrases = [
            "right now", "this moment", "here and now", "present moment",
            "be still", "just sit", "simply be", "in this moment",
            "immediate experience", "direct experience"
        ]
        
        self.seeking_phrases = [
            "practice more", "develop", "achieve", "attain", "reach",
            "become enlightened", "get realization", "work on yourself",
            "improve", "progress", "advance"
        ]
        
        self.conceptual_phrases = [
            "understand that", "you must know", "the teaching says",
            "according to", "the philosophy", "the concept",
            "intellectually", "mentally grasp"
        ]
    
    def score_dialogue(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Score a dialogue for consciousness recognition quality.
        
        Args:
            question: The question text
            answer: The answer text
            
        Returns:
            Scoring results with breakdown
        """
        # Combine question and answer for analysis
        full_text = f"{question} {answer}".lower()
        
        # Calculate component scores
        direct_pointing = self._score_phrases(full_text, self.direct_pointing_phrases)
        non_dual = self._score_phrases(full_text, self.non_dual_phrases)
        disidentification = self._score_phrases(full_text, self.disidentification_phrases)
        presence = self._score_phrases(full_text, self.presence_phrases)
        
        # Negative scoring (subtract from total)
        seeking_penalty = self._score_phrases(full_text, self.seeking_phrases) * -1
        conceptual_penalty = self._score_phrases(full_text, self.conceptual_phrases) * -1
        
        # Weighted scoring
        weighted_score = (
            direct_pointing * 0.25 +
            non_dual * 0.20 +
            disidentification * 0.20 +
            presence * 0.15 +
            seeking_penalty * 0.10 +
            conceptual_penalty * 0.10
        )
        
        # Normalize to 0-1 range
        overall_score = max(0, min(1, weighted_score))
        
        # Determine mode
        mode = self._determine_mode(question, answer, overall_score)
        
        return {
            'overall_score': overall_score,
            'direct_pointing': direct_pointing,
            'non_dual': non_dual,
            'disidentification': disidentification,
            'presence': presence,
            'seeking_penalty': abs(seeking_penalty),
            'conceptual_penalty': abs(conceptual_penalty),
            'mode': mode.value,
            'weighted_breakdown': {
                'direct_pointing': direct_pointing * 0.25,
                'non_dual': non_dual * 0.20,
                'disidentification': disidentification * 0.20,
                'presence': presence * 0.15,
                'avoid_seeking': seeking_penalty * 0.10,
                'non_conceptual': conceptual_penalty * 0.10
            }
        }
    
    def _score_phrases(self, text: str, phrases: List[str]) -> float:
        """Score text based on phrase matches."""
        matches = 0
        total_phrases = len(phrases)
        
        for phrase in phrases:
            if phrase in text:
                matches += 1
        
        return matches / total_phrases if total_phrases > 0 else 0
    
    def _determine_mode(self, question: str, answer: str, score: float) -> DialogueMode:
        """Determine the dialogue mode."""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Check for inquiry indicators
        inquiry_indicators = [
            "what is", "who am i", "how to", "what should",
            "feeling", "emotion", "fear", "doubt", "suffering"
        ]
        
        # Check for consciousness indicators
        consciousness_indicators = [
            "you are", "awareness", "consciousness", "being",
            "remain", "abide", "just be"
        ]
        
        # Check for teaching indicators
        teaching_indicators = [
            "understand", "know that", "realize", "see that",
            "the truth is", "what happens"
        ]
        
        inquiry_count = sum(1 for indicator in inquiry_indicators 
                          if indicator in question_lower or indicator in answer_lower)
        
        consciousness_count = sum(1 for indicator in consciousness_indicators 
                                if indicator in answer_lower)
        
        teaching_count = sum(1 for indicator in teaching_indicators 
                           if indicator in answer_lower)
        
        # Determine primary mode
        if score >= 0.7 and consciousness_count >= 2:
            return DialogueMode.CONSCIOUSNESS
        elif inquiry_count >= 2:
            return DialogueMode.INQUIRY
        elif teaching_count >= 2:
            return DialogueMode.TEACHING
        else:
            return DialogueMode.MIXED

