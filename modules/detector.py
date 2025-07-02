"""
Dialogue Detector for Consciousness Recognition System

Detects spiritual dialogues using regex patterns and semantic similarity.
"""

import re
from typing import List, Dict, Any, Optional


class DialogueDetector:
    """Basic dialogue detector for spiritual texts."""
    
    def __init__(self):
        """Initialize the dialogue detector."""
        self.qa_patterns = [
            r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)',
            r'Question:\s*(.*?)\s*Answer:\s*(.*?)(?=Question:|$)',
            r'Questioner:\s*(.*?)\s*Maharaj:\s*(.*?)(?=Questioner:|$)',
            r'Visitor:\s*(.*?)\s*Ramana:\s*(.*?)(?=Visitor:|$)',
        ]
    
    def detect_dialogues(
        self, 
        text: str, 
        mode: str = "regex",
        semantic_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Detect dialogues in text.
        
        Args:
            text: Input text to analyze
            mode: Detection mode ("regex", "semantic", "auto")
            semantic_threshold: Threshold for semantic similarity
            
        Returns:
            List of detected dialogues
        """
        dialogues = []
        
        if mode in ["regex", "auto"]:
            dialogues.extend(self._detect_regex_dialogues(text))
        
        if mode in ["semantic", "auto"]:
            dialogues.extend(self._detect_semantic_dialogues(text, semantic_threshold))
        
        # Remove duplicates
        unique_dialogues = []
        seen = set()
        
        for dialogue in dialogues:
            key = (dialogue['question'][:50], dialogue['answer'][:50])
            if key not in seen:
                seen.add(key)
                unique_dialogues.append(dialogue)
        
        return unique_dialogues
    
    def _detect_regex_dialogues(self, text: str) -> List[Dict[str, Any]]:
        """Detect dialogues using regex patterns."""
        dialogues = []
        
        for pattern in self.qa_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                if len(match) == 2:
                    question, answer = match
                    
                    # Clean up text
                    question = re.sub(r'\s+', ' ', question.strip())
                    answer = re.sub(r'\s+', ' ', answer.strip())
                    
                    if len(question) > 10 and len(answer) > 10:
                        dialogues.append({
                            'question': question,
                            'answer': answer,
                            'source': 'regex',
                            'confidence': 0.8
                        })
        
        return dialogues
    
    def _detect_semantic_dialogues(self, text: str, threshold: float) -> List[Dict[str, Any]]:
        """Detect dialogues using semantic similarity."""
        # This is a simplified version - the enhanced detector has full implementation
        dialogues = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Look for question-like patterns
        question_indicators = ['?', 'how', 'what', 'why', 'when', 'where', 'who']
        
        for i, para in enumerate(paragraphs):
            if any(indicator in para.lower() for indicator in question_indicators):
                if i + 1 < len(paragraphs):
                    question = para
                    answer = paragraphs[i + 1]
                    
                    if len(question) > 20 and len(answer) > 20:
                        dialogues.append({
                            'question': question,
                            'answer': answer,
                            'source': 'semantic',
                            'confidence': 0.6
                        })
        
        return dialogues

