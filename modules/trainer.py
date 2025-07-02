"""
OpenAI Training Data Generator

Formats consciousness recognition dialogues for OpenAI fine-tuning.
"""

import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class OpenAITrainer:
    """Generates OpenAI fine-tuning training data."""
    
    def __init__(self):
        """Initialize the trainer."""
        self.dialogues = []
        self.statistics = {
            'total_dialogues': 0,
            'mode_distribution': {},
            'score_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'sources': set()
        }
    
    def add_dialogue(
        self,
        question: str,
        answer: str,
        score: float,
        mode: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a dialogue to the training dataset.
        
        Args:
            question: The question text
            answer: The answer text
            score: Consciousness recognition score (0-1)
            mode: Dialogue mode (consciousness, inquiry, teaching, mixed)
            source: Source file or book name
            metadata: Additional metadata
        """
        dialogue = {
            'question': question,
            'answer': answer,
            'score': score,
            'mode': mode,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.dialogues.append(dialogue)
        self._update_statistics(dialogue)
    
    def export_to_jsonl(self, output_path: str) -> Dict[str, Any]:
        """
        Export dialogues to JSONL format for OpenAI fine-tuning.
        
        Args:
            output_path: Path to save the JSONL file
            
        Returns:
            Export results
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for dialogue in self.dialogues:
                    # Format for OpenAI fine-tuning
                    training_example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a consciousness recognition guide. Point directly to awareness without concepts or seeking. Respond from non-dual understanding."
                            },
                            {
                                "role": "user",
                                "content": dialogue['question']
                            },
                            {
                                "role": "assistant",
                                "content": dialogue['answer']
                            }
                        ],
                        "metadata": {
                            "score": dialogue['score'],
                            "mode": dialogue['mode'],
                            "source": dialogue['source'],
                            "timestamp": dialogue['timestamp']
                        }
                    }
                    
                    f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
            
            return {
                'success': True,
                'count': len(self.dialogues),
                'path': output_path,
                'statistics': self.statistics
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'count': 0
            }
    
    def export_to_csv(self, output_path: str) -> Dict[str, Any]:
        """
        Export dialogues to CSV format for analysis.
        
        Args:
            output_path: Path to save the CSV file
            
        Returns:
            Export results
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'Question', 'Answer', 'Score', 'Mode', 'Source', 
                    'Timestamp', 'Question_Length', 'Answer_Length'
                ])
                
                # Write data
                for dialogue in self.dialogues:
                    writer.writerow([
                        dialogue['question'],
                        dialogue['answer'],
                        dialogue['score'],
                        dialogue['mode'],
                        dialogue['source'],
                        dialogue['timestamp'],
                        len(dialogue['question']),
                        len(dialogue['answer'])
                    ])
            
            return {
                'success': True,
                'count': len(self.dialogues),
                'path': output_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'count': 0
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training dataset statistics."""
        return {
            'total_dialogues': len(self.dialogues),
            'mode_distribution': dict(self.statistics['mode_distribution']),
            'score_distribution': dict(self.statistics['score_distribution']),
            'sources': list(self.statistics['sources']),
            'average_score': sum(d['score'] for d in self.dialogues) / max(1, len(self.dialogues)),
            'score_range': {
                'min': min((d['score'] for d in self.dialogues), default=0),
                'max': max((d['score'] for d in self.dialogues), default=0)
            }
        }
    
    def filter_dialogues(
        self,
        min_score: float = 0.0,
        max_score: float = 1.0,
        modes: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter dialogues based on criteria.
        
        Args:
            min_score: Minimum score threshold
            max_score: Maximum score threshold
            modes: List of modes to include
            sources: List of sources to include
            
        Returns:
            Filtered dialogues
        """
        filtered = []
        
        for dialogue in self.dialogues:
            # Score filter
            if not (min_score <= dialogue['score'] <= max_score):
                continue
            
            # Mode filter
            if modes and dialogue['mode'] not in modes:
                continue
            
            # Source filter
            if sources and dialogue['source'] not in sources:
                continue
            
            filtered.append(dialogue)
        
        return filtered
    
    def _update_statistics(self, dialogue: Dict[str, Any]):
        """Update internal statistics."""
        self.statistics['total_dialogues'] = len(self.dialogues)
        
        # Mode distribution
        mode = dialogue['mode']
        self.statistics['mode_distribution'][mode] = \
            self.statistics['mode_distribution'].get(mode, 0) + 1
        
        # Score distribution
        score = dialogue['score']
        if score >= 0.7:
            self.statistics['score_distribution']['high'] += 1
        elif score >= 0.4:
            self.statistics['score_distribution']['medium'] += 1
        else:
            self.statistics['score_distribution']['low'] += 1
        
        # Sources
        self.statistics['sources'].add(dialogue['source'])

