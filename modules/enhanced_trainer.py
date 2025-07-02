"""
Enhanced OpenAI Training Data Generator with Comprehensive Validation

Includes input validation, output verification, and quality control
for consciousness recognition training data.
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from .output_validator import OutputValidator
    from .trainer import OpenAITrainer
except ImportError:
    from output_validator import OutputValidator
    from trainer import OpenAITrainer


class EnhancedOpenAITrainer(OpenAITrainer):
    """Enhanced trainer with comprehensive validation and quality control."""
    
    def __init__(self):
        """Initialize the enhanced trainer."""
        super().__init__()
        self.output_validator = OutputValidator()
        self.validation_enabled = True
        self.auto_fix_enabled = True
        self.rejected_dialogues = []
        
        # Quality thresholds
        self.min_quality_score = 0.3
        self.max_warnings_per_item = 3
        
    def add_dialogue(
        self,
        question: str,
        answer: str,
        score: float,
        mode: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a dialogue with pre-validation.
        
        Args:
            question: The question text
            answer: The answer text
            score: Consciousness recognition score (0-1)
            mode: Dialogue mode (consciousness, inquiry, teaching, mixed)
            source: Source file or book name
            metadata: Additional metadata
            
        Returns:
            Dict with addition result and any issues
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
        
        # Validate before adding if validation is enabled
        if self.validation_enabled:
            validation_result = self.output_validator.validate_dialogue_before_export(dialogue)
            
            if not validation_result['is_valid']:
                # Try auto-fix if enabled
                if self.auto_fix_enabled:
                    fixed_dialogue = self._attempt_auto_fix(dialogue, validation_result)
                    if fixed_dialogue:
                        dialogue = fixed_dialogue
                        validation_result = self.output_validator.validate_dialogue_before_export(dialogue)
                
                # If still invalid, reject
                if not validation_result['is_valid']:
                    self.rejected_dialogues.append({
                        'dialogue': dialogue,
                        'issues': validation_result['issues'],
                        'warnings': validation_result['warnings']
                    })
                    
                    return {
                        'success': False,
                        'issues': validation_result['issues'],
                        'warnings': validation_result['warnings'],
                        'rejected': True
                    }
            
            # Check warning threshold
            if len(validation_result['warnings']) > self.max_warnings_per_item:
                self.rejected_dialogues.append({
                    'dialogue': dialogue,
                    'issues': ['Too many warnings'],
                    'warnings': validation_result['warnings']
                })
                
                return {
                    'success': False,
                    'issues': ['Dialogue rejected due to excessive warnings'],
                    'warnings': validation_result['warnings'],
                    'rejected': True
                }
        
        # Add to collection
        self.dialogues.append(dialogue)
        self._update_statistics(dialogue)
        
        return {
            'success': True,
            'issues': [],
            'warnings': validation_result.get('warnings', []) if self.validation_enabled else [],
            'rejected': False
        }
    
    def export_to_jsonl(self, output_path: str, validate_output: bool = True) -> Dict[str, Any]:
        """
        Export dialogues to JSONL with comprehensive validation.
        
        Args:
            output_path: Path to save the JSONL file
            validate_output: Whether to validate the output file
            
        Returns:
            Export results with validation details
        """
        try:
            # Pre-export validation
            if len(self.dialogues) == 0:
                return {
                    'success': False,
                    'error': 'No dialogues to export',
                    'count': 0,
                    'validation': None
                }
            
            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export with enhanced error handling
            export_result = self._export_with_validation(output_path)
            
            if not export_result['success']:
                return export_result
            
            # Post-export validation
            validation_result = None
            if validate_output:
                validation_result = self.output_validator.validate_jsonl_output(output_path)
                
                # If validation fails, attempt to fix and re-export
                if not validation_result.is_valid and self.auto_fix_enabled:
                    fix_result = self._attempt_output_fix(output_path, validation_result)
                    if fix_result['success']:
                        validation_result = self.output_validator.validate_jsonl_output(output_path)
            
            return {
                'success': True,
                'count': len(self.dialogues),
                'path': output_path,
                'statistics': self.statistics,
                'validation': validation_result,
                'rejected_count': len(self.rejected_dialogues)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'count': 0,
                'validation': None
            }
    
    def _export_with_validation(self, output_path: str) -> Dict[str, Any]:
        """Export with item-by-item validation."""
        try:
            exported_count = 0
            skipped_count = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for dialogue in self.dialogues:
                    try:
                        # Create training example
                        training_example = self._create_training_example(dialogue)
                        
                        # Validate training example structure
                        if self._validate_training_example(training_example):
                            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                            exported_count += 1
                        else:
                            skipped_count += 1
                            
                    except Exception as e:
                        skipped_count += 1
                        # Log the error but continue processing
                        continue
            
            return {
                'success': True,
                'exported_count': exported_count,
                'skipped_count': skipped_count
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Export failed: {str(e)}"
            }
    
    def _create_training_example(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """Create a training example with enhanced metadata."""
        # Determine appropriate system prompt based on content type
        system_prompt = self._get_system_prompt(dialogue)
        
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
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
                "timestamp": dialogue['timestamp'],
                "processing_method": dialogue.get('metadata', {}).get('detection_method', 'unknown'),
                "original_type": dialogue.get('metadata', {}).get('original_type', 'dialogue'),
                "quality_validated": True
            }
        }
        
        return training_example
    
    def _get_system_prompt(self, dialogue: Dict[str, Any]) -> str:
        """Get appropriate system prompt based on dialogue characteristics."""
        mode = dialogue.get('mode', 'unknown')
        detection_method = dialogue.get('metadata', {}).get('detection_method', 'unknown')
        
        if mode == 'consciousness':
            return "You are a consciousness recognition guide. Point directly to awareness without concepts or seeking. Respond from non-dual understanding."
        elif mode == 'inquiry':
            return "You are a spiritual inquiry guide. Help questioners investigate their true nature through direct pointing and self-inquiry."
        elif mode == 'teaching':
            return "You are a non-dual teacher. Share wisdom that points to the truth of what the questioner already is."
        elif detection_method == 'passage_extraction':
            return "You are a consciousness guide. Respond with the wisdom and understanding that points to the questioner's true nature."
        else:
            return "You are a consciousness recognition guide. Point directly to awareness without concepts or seeking. Respond from non-dual understanding."
    
    def _validate_training_example(self, training_example: Dict[str, Any]) -> bool:
        """Validate a single training example."""
        try:
            # Check required structure
            if 'messages' not in training_example or 'metadata' not in training_example:
                return False
            
            messages = training_example['messages']
            if len(messages) != 3:
                return False
            
            # Check message roles
            expected_roles = ['system', 'user', 'assistant']
            for i, message in enumerate(messages):
                if message.get('role') != expected_roles[i]:
                    return False
                if not message.get('content', '').strip():
                    return False
            
            # Check content lengths
            user_content = messages[1]['content']
            assistant_content = messages[2]['content']
            
            if (len(user_content) < 10 or len(user_content) > 2000 or
                len(assistant_content) < 20 or len(assistant_content) > 4000):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _attempt_auto_fix(self, dialogue: Dict[str, Any], validation_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to automatically fix dialogue issues."""
        fixed_dialogue = dialogue.copy()
        
        for issue in validation_result['issues']:
            if 'Question too short' in issue:
                # Pad short questions
                question = fixed_dialogue['question']
                if len(question) < 10:
                    fixed_dialogue['question'] = f"Can you explain: {question}?"
            
            elif 'Answer too short' in issue:
                # Skip answers that are too short - can't reliably fix
                return None
            
            elif 'Invalid score' in issue:
                # Set default score
                fixed_dialogue['score'] = 0.5
            
            elif 'Missing or empty required field' in issue:
                # Set defaults for missing fields
                if 'question' not in fixed_dialogue or not fixed_dialogue['question']:
                    return None  # Can't fix missing question
                if 'answer' not in fixed_dialogue or not fixed_dialogue['answer']:
                    return None  # Can't fix missing answer
                if 'score' not in fixed_dialogue:
                    fixed_dialogue['score'] = 0.5
                if 'mode' not in fixed_dialogue:
                    fixed_dialogue['mode'] = 'unknown'
                if 'source' not in fixed_dialogue:
                    fixed_dialogue['source'] = 'auto-fixed'
        
        return fixed_dialogue
    
    def _attempt_output_fix(self, output_path: str, validation_result) -> Dict[str, Any]:
        """Attempt to fix output file issues."""
        try:
            # For now, just re-export with stricter validation
            # In the future, could implement more sophisticated fixes
            backup_path = output_path + '.backup'
            os.rename(output_path, backup_path)
            
            # Re-export with stricter validation
            self.validation_enabled = True
            export_result = self._export_with_validation(output_path)
            
            if export_result['success']:
                os.remove(backup_path)
                return {'success': True, 'method': 're-export'}
            else:
                os.rename(backup_path, output_path)
                return {'success': False, 'error': 'Fix attempt failed'}
                
        except Exception as e:
            return {'success': False, 'error': f'Fix attempt failed: {str(e)}'}
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report."""
        total_processed = len(self.dialogues) + len(self.rejected_dialogues)
        
        return {
            'total_processed': total_processed,
            'accepted': len(self.dialogues),
            'rejected': len(self.rejected_dialogues),
            'acceptance_rate': len(self.dialogues) / max(total_processed, 1),
            'average_score': sum(d['score'] for d in self.dialogues) / max(len(self.dialogues), 1),
            'rejection_reasons': self._analyze_rejection_reasons(),
            'quality_distribution': self._get_quality_distribution()
        }
    
    def _analyze_rejection_reasons(self) -> Dict[str, int]:
        """Analyze reasons for dialogue rejection."""
        reasons = {}
        
        for rejected in self.rejected_dialogues:
            for issue in rejected['issues']:
                reason_key = issue.split(':')[0]  # Get the main reason
                reasons[reason_key] = reasons.get(reason_key, 0) + 1
        
        return reasons
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of quality scores."""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for dialogue in self.dialogues:
            score = dialogue['score']
            if score >= 0.7:
                distribution['high'] += 1
            elif score >= 0.4:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def export_rejected_dialogues(self, output_path: str) -> Dict[str, Any]:
        """Export rejected dialogues for analysis."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for rejected in self.rejected_dialogues:
                    f.write(json.dumps(rejected, ensure_ascii=False) + '\n')
            
            return {
                'success': True,
                'count': len(self.rejected_dialogues),
                'path': output_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'count': 0
            }

