"""
Comprehensive Output Validation for Consciousness Recognition System

Validates and quality-checks all system outputs including JSONL files,
dialogue quality, and training data integrity.
"""

import json
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import hashlib
from datetime import datetime


@dataclass
class OutputValidationResult:
    """Result of output validation."""
    is_valid: bool
    total_items: int
    valid_items: int
    invalid_items: int
    quality_score: float
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    file_integrity: bool
    openai_compliance: bool


class OutputValidator:
    """Validates system outputs for quality and compliance."""
    
    def __init__(self):
        """Initialize the output validator."""
        self.min_question_length = 10
        self.max_question_length = 2000
        self.min_answer_length = 20
        self.max_answer_length = 4000
        self.min_consciousness_score = 0.1
        self.required_fields = ['messages', 'metadata']
        self.required_message_fields = ['role', 'content']
        self.required_metadata_fields = ['score', 'mode', 'source']
        
        # Quality patterns for consciousness content
        self.quality_indicators = [
            r'\b(awareness|consciousness|being|presence)\b',
            r'\b(you are|what you are|true nature)\b',
            r'\b(remain as|be as you are|just be)\b',
            r'\b(here and now|this moment|present)\b'
        ]
        
        # Anti-patterns that indicate poor quality
        self.anti_patterns = [
            r'\b(practice more|develop|achieve|attain)\b',
            r'\b(become enlightened|get realization)\b',
            r'\b(work on yourself|improve|progress)\b',
            r'\b(the teaching says|according to)\b'
        ]
    
    def validate_jsonl_output(self, file_path: str) -> OutputValidationResult:
        """
        Comprehensive validation of JSONL output file.
        
        Args:
            file_path: Path to JSONL file to validate
            
        Returns:
            OutputValidationResult with detailed analysis
        """
        issues = []
        warnings = []
        recommendations = []
        
        try:
            # Check file existence and accessibility
            if not os.path.exists(file_path):
                return OutputValidationResult(
                    is_valid=False,
                    total_items=0,
                    valid_items=0,
                    invalid_items=0,
                    quality_score=0.0,
                    issues=[f"Output file does not exist: {file_path}"],
                    warnings=[],
                    recommendations=["Ensure export process completed successfully"],
                    file_integrity=False,
                    openai_compliance=False
                )
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return OutputValidationResult(
                    is_valid=False,
                    total_items=0,
                    valid_items=0,
                    invalid_items=0,
                    quality_score=0.0,
                    issues=["Output file is empty"],
                    warnings=[],
                    recommendations=["Check if any dialogues were processed"],
                    file_integrity=False,
                    openai_compliance=False
                )
            
            # Validate file integrity
            file_integrity = self._validate_file_integrity(file_path)
            if not file_integrity['is_valid']:
                issues.extend(file_integrity['issues'])
            
            # Parse and validate JSONL content
            validation_results = self._validate_jsonl_content(file_path)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(validation_results['items'])
            
            # Check OpenAI compliance
            openai_compliance = self._check_openai_compliance(validation_results['items'])
            
            # Generate recommendations
            recommendations.extend(self._generate_recommendations(validation_results, quality_score))
            
            # Determine overall validity
            is_valid = (
                file_integrity['is_valid'] and
                validation_results['valid_count'] > 0 and
                len(validation_results['critical_issues']) == 0 and
                quality_score >= 0.3
            )
            
            return OutputValidationResult(
                is_valid=is_valid,
                total_items=validation_results['total_count'],
                valid_items=validation_results['valid_count'],
                invalid_items=validation_results['invalid_count'],
                quality_score=quality_score,
                issues=issues + validation_results['critical_issues'],
                warnings=warnings + validation_results['warnings'],
                recommendations=recommendations,
                file_integrity=file_integrity['is_valid'],
                openai_compliance=openai_compliance['is_compliant']
            )
            
        except Exception as e:
            return OutputValidationResult(
                is_valid=False,
                total_items=0,
                valid_items=0,
                invalid_items=0,
                quality_score=0.0,
                issues=[f"Validation failed: {str(e)}"],
                warnings=[],
                recommendations=["Check file format and content"],
                file_integrity=False,
                openai_compliance=False
            )
    
    def _validate_file_integrity(self, file_path: str) -> Dict[str, Any]:
        """Validate file integrity and format."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Check if file is readable
                content = f.read()
                
                # Check for encoding issues
                try:
                    content.encode('utf-8')
                except UnicodeEncodeError:
                    issues.append("File contains invalid UTF-8 characters")
                
                # Check for basic JSONL structure
                lines = content.strip().split('\n')
                if not lines or lines == ['']:
                    issues.append("File appears to be empty or contains only whitespace")
                
                # Check for malformed JSON lines
                malformed_lines = []
                for i, line in enumerate(lines):
                    if line.strip():  # Skip empty lines
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            malformed_lines.append(i + 1)
                
                if malformed_lines:
                    issues.append(f"Malformed JSON on lines: {malformed_lines[:5]}")  # Show first 5
                
            return {
                'is_valid': len(issues) == 0,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'issues': [f"File integrity check failed: {str(e)}"]
            }
    
    def _validate_jsonl_content(self, file_path: str) -> Dict[str, Any]:
        """Validate JSONL content structure and quality."""
        items = []
        critical_issues = []
        warnings = []
        valid_count = 0
        invalid_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        validation_result = self._validate_single_item(item, line_num)
                        
                        items.append({
                            'item': item,
                            'line_number': line_num,
                            'validation': validation_result
                        })
                        
                        if validation_result['is_valid']:
                            valid_count += 1
                        else:
                            invalid_count += 1
                            critical_issues.extend([
                                f"Line {line_num}: {issue}" 
                                for issue in validation_result['issues']
                            ])
                        
                        warnings.extend([
                            f"Line {line_num}: {warning}" 
                            for warning in validation_result['warnings']
                        ])
                        
                    except json.JSONDecodeError as e:
                        invalid_count += 1
                        critical_issues.append(f"Line {line_num}: Invalid JSON - {str(e)}")
            
            return {
                'items': items,
                'total_count': valid_count + invalid_count,
                'valid_count': valid_count,
                'invalid_count': invalid_count,
                'critical_issues': critical_issues,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'items': [],
                'total_count': 0,
                'valid_count': 0,
                'invalid_count': 0,
                'critical_issues': [f"Content validation failed: {str(e)}"],
                'warnings': []
            }
    
    def _validate_single_item(self, item: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """Validate a single JSONL item."""
        issues = []
        warnings = []
        
        # Check required top-level fields
        for field in self.required_fields:
            if field not in item:
                issues.append(f"Missing required field: {field}")
        
        # Validate messages structure
        if 'messages' in item:
            messages_validation = self._validate_messages(item['messages'])
            issues.extend(messages_validation['issues'])
            warnings.extend(messages_validation['warnings'])
        
        # Validate metadata
        if 'metadata' in item:
            metadata_validation = self._validate_metadata(item['metadata'])
            issues.extend(metadata_validation['issues'])
            warnings.extend(metadata_validation['warnings'])
        
        # Content quality validation
        if 'messages' in item and len(item['messages']) >= 3:
            quality_validation = self._validate_content_quality(item['messages'])
            warnings.extend(quality_validation['warnings'])
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def _validate_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate messages structure for OpenAI format."""
        issues = []
        warnings = []
        
        if not isinstance(messages, list):
            issues.append("Messages must be a list")
            return {'issues': issues, 'warnings': warnings}
        
        if len(messages) != 3:
            issues.append(f"Expected 3 messages (system, user, assistant), got {len(messages)}")
        
        expected_roles = ['system', 'user', 'assistant']
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                issues.append(f"Message {i} must be a dictionary")
                continue
            
            # Check required fields
            for field in self.required_message_fields:
                if field not in message:
                    issues.append(f"Message {i} missing required field: {field}")
            
            # Check role sequence
            if i < len(expected_roles) and message.get('role') != expected_roles[i]:
                issues.append(f"Message {i} has role '{message.get('role')}', expected '{expected_roles[i]}'")
            
            # Check content length
            content = message.get('content', '')
            if not content or not content.strip():
                issues.append(f"Message {i} has empty content")
            elif i == 1:  # User message (question)
                if len(content) < self.min_question_length:
                    warnings.append(f"Question too short: {len(content)} chars")
                elif len(content) > self.max_question_length:
                    warnings.append(f"Question too long: {len(content)} chars")
            elif i == 2:  # Assistant message (answer)
                if len(content) < self.min_answer_length:
                    warnings.append(f"Answer too short: {len(content)} chars")
                elif len(content) > self.max_answer_length:
                    warnings.append(f"Answer too long: {len(content)} chars")
        
        return {'issues': issues, 'warnings': warnings}
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata structure."""
        issues = []
        warnings = []
        
        if not isinstance(metadata, dict):
            issues.append("Metadata must be a dictionary")
            return {'issues': issues, 'warnings': warnings}
        
        # Check required metadata fields
        for field in self.required_metadata_fields:
            if field not in metadata:
                issues.append(f"Metadata missing required field: {field}")
        
        # Validate score
        if 'score' in metadata:
            score = metadata['score']
            if not isinstance(score, (int, float)):
                issues.append("Score must be a number")
            elif score < 0 or score > 1:
                issues.append(f"Score must be between 0 and 1, got {score}")
            elif score < self.min_consciousness_score:
                warnings.append(f"Low consciousness score: {score}")
        
        # Validate mode
        if 'mode' in metadata:
            mode = metadata['mode']
            valid_modes = ['consciousness', 'inquiry', 'teaching', 'mixed', 'unknown']
            if mode not in valid_modes:
                warnings.append(f"Unexpected mode: {mode}")
        
        return {'issues': issues, 'warnings': warnings}
    
    def _validate_content_quality(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate content quality for consciousness recognition."""
        warnings = []
        
        if len(messages) < 3:
            return {'warnings': warnings}
        
        question = messages[1].get('content', '')
        answer = messages[2].get('content', '')
        combined_text = f"{question} {answer}".lower()
        
        # Check for quality indicators
        quality_matches = sum(1 for pattern in self.quality_indicators 
                            if re.search(pattern, combined_text))
        
        # Check for anti-patterns
        anti_matches = sum(1 for pattern in self.anti_patterns 
                         if re.search(pattern, combined_text))
        
        if quality_matches == 0:
            warnings.append("No consciousness-related keywords found")
        
        if anti_matches > 0:
            warnings.append(f"Found {anti_matches} seeking/conceptual patterns")
        
        # Check for repetitive content
        if len(set(answer.split())) / max(len(answer.split()), 1) < 0.5:
            warnings.append("Answer appears repetitive")
        
        return {'warnings': warnings}
    
    def _calculate_quality_score(self, items: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score for the dataset."""
        if not items:
            return 0.0
        
        total_score = 0.0
        valid_items = 0
        
        for item_data in items:
            if item_data['validation']['is_valid']:
                item = item_data['item']
                
                # Base score from metadata
                metadata_score = item.get('metadata', {}).get('score', 0.0)
                
                # Quality adjustments
                quality_adjustment = 0.0
                
                # Bonus for good structure
                if len(item.get('messages', [])) == 3:
                    quality_adjustment += 0.1
                
                # Penalty for warnings
                warning_count = len(item_data['validation']['warnings'])
                quality_adjustment -= warning_count * 0.05
                
                final_score = max(0.0, min(1.0, metadata_score + quality_adjustment))
                total_score += final_score
                valid_items += 1
        
        return total_score / max(valid_items, 1)
    
    def _check_openai_compliance(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check compliance with OpenAI fine-tuning requirements."""
        issues = []
        
        if not items:
            return {'is_compliant': False, 'issues': ['No items to validate']}
        
        valid_items = [item for item in items if item['validation']['is_valid']]
        
        if len(valid_items) < 10:
            issues.append(f"Insufficient training examples: {len(valid_items)} (minimum 10 recommended)")
        
        # Check for diversity in content
        unique_questions = set()
        for item_data in valid_items:
            messages = item_data['item'].get('messages', [])
            if len(messages) >= 2:
                question = messages[1].get('content', '')
                unique_questions.add(question[:100])  # First 100 chars for uniqueness
        
        uniqueness_ratio = len(unique_questions) / max(len(valid_items), 1)
        if uniqueness_ratio < 0.8:
            issues.append(f"Low content diversity: {uniqueness_ratio:.2f} (many similar questions)")
        
        return {
            'is_compliant': len(issues) == 0,
            'issues': issues
        }
    
    def _generate_recommendations(self, validation_results: Dict[str, Any], quality_score: float) -> List[str]:
        """Generate recommendations for improving output quality."""
        recommendations = []
        
        if validation_results['invalid_count'] > 0:
            recommendations.append(f"Fix {validation_results['invalid_count']} invalid items before using for training")
        
        if quality_score < 0.5:
            recommendations.append("Quality score is low - consider raising consciousness score threshold")
        
        if validation_results['total_count'] < 50:
            recommendations.append("Small dataset - consider processing more source material")
        
        warning_count = len(validation_results['warnings'])
        if warning_count > validation_results['total_count'] * 0.5:
            recommendations.append("High warning rate - review source material quality")
        
        return recommendations
    
    def validate_dialogue_before_export(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual dialogue before adding to export."""
        issues = []
        warnings = []
        
        # Check required fields
        required = ['question', 'answer', 'score']
        for field in required:
            if field not in dialogue or not dialogue[field]:
                issues.append(f"Missing or empty required field: {field}")
        
        # Validate content lengths
        question = dialogue.get('question', '')
        answer = dialogue.get('answer', '')
        
        if len(question) < self.min_question_length:
            issues.append(f"Question too short: {len(question)} chars")
        elif len(question) > self.max_question_length:
            warnings.append(f"Question very long: {len(question)} chars")
        
        if len(answer) < self.min_answer_length:
            issues.append(f"Answer too short: {len(answer)} chars")
        elif len(answer) > self.max_answer_length:
            warnings.append(f"Answer very long: {len(answer)} chars")
        
        # Validate score
        score = dialogue.get('score', 0)
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            issues.append(f"Invalid score: {score}")
        elif score < self.min_consciousness_score:
            warnings.append(f"Low consciousness score: {score}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

