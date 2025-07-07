"""
Structure Validator Module
=========================

Enforce format compliance to prevent Q&A being converted into
vague summaries or broken formats.

Features:
- Q&A format validation
- Chat format compliance
- Structure preservation
- Format-specific validation
- Schema enforcement
- Pattern matching
"""

import streamlit as st
import re
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
from modules.logger import get_logger, log_event

class StructureValidator:
    """
    Comprehensive structure validation for content format compliance
    
    Features:
    - Multiple format validation (Q&A, Chat, Instruction, etc.)
    - Structure preservation analysis
    - Format-specific pattern matching
    - Schema compliance checking
    - Detailed validation reporting
    """
    
    def __init__(self):
        self.logger = get_logger("structure_validator")
        
        # Format patterns and schemas
        self.format_patterns = {
            'qa_format': {
                'patterns': [
                    r'(?i)(?:question|q):\s*(.+?)(?:answer|a):\s*(.+)',
                    r'(?i)(.+?)\?\s*(.+)',
                    r'(?i)q:\s*(.+?)\s*a:\s*(.+)',
                ],
                'required_elements': ['question', 'answer'],
                'structure_indicators': ['?', 'Q:', 'A:', 'Question:', 'Answer:']
            },
            'chat_format': {
                'patterns': [
                    r'(?i)(?:user|human):\s*(.+?)(?:assistant|ai|bot):\s*(.+)',
                    r'(?i)(?:system):\s*(.+?)(?:user):\s*(.+?)(?:assistant):\s*(.+)',
                ],
                'required_elements': ['messages'],
                'structure_indicators': ['User:', 'Assistant:', 'System:', 'Human:', 'AI:']
            },
            'instruction_format': {
                'patterns': [
                    r'(?i)(?:instruction|task):\s*(.+?)(?:input|context):\s*(.+?)(?:output|response):\s*(.+)',
                    r'(?i)(?:instruction):\s*(.+?)(?:output):\s*(.+)',
                ],
                'required_elements': ['instruction', 'output'],
                'structure_indicators': ['Instruction:', 'Input:', 'Output:', 'Task:']
            },
            'narrative_format': {
                'patterns': [
                    r'(.+)',  # Any continuous text
                ],
                'required_elements': ['content'],
                'structure_indicators': []
            }
        }
        
        # Validation thresholds
        self.thresholds = {
            'excellent': 0.95,
            'good': 0.85,
            'acceptable': 0.75,
            'concerning': 0.65,
            'poor': 0.50
        }
        
        # Initialize session state
        if 'structure_validation_cache' not in st.session_state:
            st.session_state['structure_validation_cache'] = {}
    
    def validate_structure(self, content: Union[str, Dict], expected_format: str, original_content: Optional[Union[str, Dict]] = None) -> Dict[str, Any]:
        """
        Validate content structure against expected format
        
        Args:
            content: Content to validate (string or dict)
            expected_format: Expected format ('qa_format', 'chat_format', etc.)
            original_content: Original content for comparison
        
        Returns:
            Comprehensive structure validation results
        """
        
        try:
            # Create cache key
            cache_key = f"{hash(str(content))}_{expected_format}_{hash(str(original_content or ''))}"
            
            # Check cache
            if cache_key in st.session_state['structure_validation_cache']:
                return st.session_state['structure_validation_cache'][cache_key]
            
            # Perform validation based on content type
            if isinstance(content, dict):
                results = self.validate_dict_structure(content, expected_format, original_content)
            else:
                results = self.validate_text_structure(content, expected_format, original_content)
            
            # Add metadata
            results.update({
                'expected_format': expected_format,
                'content_type': type(content).__name__,
                'validation_timestamp': datetime.now().isoformat()
            })
            
            # Calculate composite scores
            results['overall_compliance'] = self.calculate_overall_compliance(results)
            results['structure_preservation'] = self.calculate_structure_preservation(results, original_content)
            results['format_confidence'] = self.calculate_format_confidence(results)
            
            # Add quality assessment
            results['quality_assessment'] = self.assess_structure_quality(results)
            
            # Cache results
            st.session_state['structure_validation_cache'][cache_key] = results
            
            # Log validation
            log_event("structure_validated", {
                "expected_format": expected_format,
                "overall_compliance": results['overall_compliance'],
                "format_confidence": results['format_confidence']
            }, "structure_validator")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Structure validation failed: {str(e)}")
            return self.get_fallback_structure_results(expected_format)
    
    def validate_text_structure(self, text: str, expected_format: str, original_text: Optional[str] = None) -> Dict[str, Any]:
        """Validate text-based content structure"""
        
        results = {
            'format_detected': self.detect_format(text),
            'pattern_matches': self.check_format_patterns(text, expected_format),
            'structure_indicators': self.check_structure_indicators(text, expected_format),
            'element_completeness': self.check_element_completeness(text, expected_format),
            'format_consistency': self.check_format_consistency(text, expected_format),
            'structure_clarity': self.assess_structure_clarity(text, expected_format)
        }
        
        # Compare with original if provided
        if original_text:
            results['structure_preservation_score'] = self.compare_structure_preservation(text, original_text, expected_format)
            results['format_drift'] = self.calculate_format_drift(text, original_text, expected_format)
        
        return results
    
    def validate_dict_structure(self, data: Dict, expected_format: str, original_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate dictionary-based content structure"""
        
        results = {
            'schema_compliance': self.check_schema_compliance(data, expected_format),
            'required_fields': self.check_required_fields(data, expected_format),
            'field_types': self.validate_field_types(data, expected_format),
            'nested_structure': self.validate_nested_structure(data, expected_format),
            'data_completeness': self.assess_data_completeness(data, expected_format)
        }
        
        # Compare with original if provided
        if original_data:
            results['structure_preservation_score'] = self.compare_dict_structure_preservation(data, original_data, expected_format)
            results['schema_drift'] = self.calculate_schema_drift(data, original_data, expected_format)
        
        return results
    
    def detect_format(self, text: str) -> str:
        """Detect the most likely format of the text"""
        
        format_scores = {}
        
        for format_name, format_info in self.format_patterns.items():
            score = 0
            
            # Check patterns
            for pattern in format_info['patterns']:
                if re.search(pattern, text, re.DOTALL):
                    score += 2
            
            # Check structure indicators
            for indicator in format_info['structure_indicators']:
                if indicator.lower() in text.lower():
                    score += 1
            
            format_scores[format_name] = score
        
        # Return format with highest score
        if format_scores:
            return max(format_scores, key=format_scores.get)
        else:
            return 'unknown'
    
    def check_format_patterns(self, text: str, expected_format: str) -> Dict[str, Any]:
        """Check if text matches expected format patterns"""
        
        if expected_format not in self.format_patterns:
            return {'matches': [], 'score': 0.0}
        
        format_info = self.format_patterns[expected_format]
        matches = []
        
        for pattern in format_info['patterns']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                matches.append({
                    'pattern': pattern,
                    'groups': match.groups(),
                    'span': match.span()
                })
        
        # Calculate pattern match score
        score = len(matches) / len(format_info['patterns']) if format_info['patterns'] else 0.0
        
        return {
            'matches': matches,
            'score': score,
            'total_patterns': len(format_info['patterns']),
            'matched_patterns': len(matches)
        }
    
    def check_structure_indicators(self, text: str, expected_format: str) -> Dict[str, Any]:
        """Check for structure indicators in text"""
        
        if expected_format not in self.format_patterns:
            return {'indicators_found': [], 'score': 0.0}
        
        format_info = self.format_patterns[expected_format]
        indicators_found = []
        
        for indicator in format_info['structure_indicators']:
            if indicator.lower() in text.lower():
                indicators_found.append(indicator)
        
        # Calculate indicator score
        score = len(indicators_found) / len(format_info['structure_indicators']) if format_info['structure_indicators'] else 1.0
        
        return {
            'indicators_found': indicators_found,
            'score': score,
            'total_indicators': len(format_info['structure_indicators']),
            'found_indicators': len(indicators_found)
        }
    
    def check_element_completeness(self, text: str, expected_format: str) -> Dict[str, Any]:
        """Check if all required elements are present"""
        
        if expected_format not in self.format_patterns:
            return {'elements_found': [], 'score': 0.0}
        
        format_info = self.format_patterns[expected_format]
        required_elements = format_info['required_elements']
        elements_found = []
        
        # Check for each required element
        for element in required_elements:
            if expected_format == 'qa_format':
                if element == 'question':
                    if '?' in text or any(word in text.lower() for word in ['question', 'q:', 'what', 'how', 'why', 'when', 'where']):
                        elements_found.append(element)
                elif element == 'answer':
                    if any(word in text.lower() for word in ['answer', 'a:', 'response', 'reply']):
                        elements_found.append(element)
            
            elif expected_format == 'chat_format':
                if element == 'messages':
                    if any(word in text.lower() for word in ['user:', 'assistant:', 'system:', 'human:', 'ai:']):
                        elements_found.append(element)
            
            elif expected_format == 'instruction_format':
                if element == 'instruction':
                    if any(word in text.lower() for word in ['instruction:', 'task:', 'please', 'create', 'write']):
                        elements_found.append(element)
                elif element == 'output':
                    if any(word in text.lower() for word in ['output:', 'response:', 'result:']):
                        elements_found.append(element)
        
        # Calculate completeness score
        score = len(elements_found) / len(required_elements) if required_elements else 1.0
        
        return {
            'elements_found': elements_found,
            'score': score,
            'required_elements': required_elements,
            'missing_elements': [elem for elem in required_elements if elem not in elements_found]
        }
    
    def check_format_consistency(self, text: str, expected_format: str) -> Dict[str, Any]:
        """Check internal format consistency"""
        
        consistency_issues = []
        score = 1.0
        
        if expected_format == 'qa_format':
            # Check for multiple question/answer pairs consistency
            question_count = len(re.findall(r'(?i)(?:question|q):', text))
            answer_count = len(re.findall(r'(?i)(?:answer|a):', text))
            
            if question_count != answer_count and question_count > 0:
                consistency_issues.append(f"Mismatched Q&A pairs: {question_count} questions, {answer_count} answers")
                score -= 0.3
            
            # Check for proper question format
            if '?' not in text and question_count == 0:
                consistency_issues.append("No clear question format detected")
                score -= 0.2
        
        elif expected_format == 'chat_format':
            # Check for proper conversation flow
            user_messages = len(re.findall(r'(?i)(?:user|human):', text))
            assistant_messages = len(re.findall(r'(?i)(?:assistant|ai|bot):', text))
            
            if abs(user_messages - assistant_messages) > 1:
                consistency_issues.append(f"Unbalanced conversation: {user_messages} user, {assistant_messages} assistant")
                score -= 0.3
        
        elif expected_format == 'instruction_format':
            # Check for clear instruction-output structure
            if 'instruction' not in text.lower() and 'task' not in text.lower():
                consistency_issues.append("No clear instruction detected")
                score -= 0.4
        
        score = max(0.0, score)
        
        return {
            'consistency_issues': consistency_issues,
            'score': score,
            'issue_count': len(consistency_issues)
        }
    
    def assess_structure_clarity(self, text: str, expected_format: str) -> Dict[str, Any]:
        """Assess how clear and well-structured the content is"""
        
        clarity_score = 1.0
        clarity_factors = []
        
        # General clarity factors
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_length > 30:
            clarity_score -= 0.1
            clarity_factors.append("Long sentences may reduce clarity")
        
        # Format-specific clarity
        if expected_format == 'qa_format':
            # Check for clear question-answer separation
            if not any(sep in text for sep in ['Q:', 'A:', 'Question:', 'Answer:']):
                clarity_score -= 0.2
                clarity_factors.append("Unclear Q&A separation")
        
        elif expected_format == 'chat_format':
            # Check for clear speaker identification
            if not any(speaker in text for speaker in ['User:', 'Assistant:', 'System:']):
                clarity_score -= 0.2
                clarity_factors.append("Unclear speaker identification")
        
        # Check for excessive complexity
        complex_words = len([word for word in text.split() if len(word) > 10])
        total_words = len(text.split())
        
        if total_words > 0 and complex_words / total_words > 0.2:
            clarity_score -= 0.1
            clarity_factors.append("High complexity may reduce clarity")
        
        clarity_score = max(0.0, clarity_score)
        
        return {
            'clarity_score': clarity_score,
            'clarity_factors': clarity_factors,
            'avg_sentence_length': avg_sentence_length,
            'complexity_ratio': complex_words / total_words if total_words > 0 else 0
        }
    
    def check_schema_compliance(self, data: Dict, expected_format: str) -> Dict[str, Any]:
        """Check dictionary data against expected schema"""
        
        compliance_issues = []
        score = 1.0
        
        if expected_format == 'chat_format':
            # Check for messages array
            if 'messages' not in data:
                compliance_issues.append("Missing 'messages' field")
                score -= 0.5
            elif not isinstance(data['messages'], list):
                compliance_issues.append("'messages' should be a list")
                score -= 0.3
            else:
                # Check message structure
                for i, message in enumerate(data['messages']):
                    if not isinstance(message, dict):
                        compliance_issues.append(f"Message {i} should be a dictionary")
                        score -= 0.1
                    elif 'role' not in message or 'content' not in message:
                        compliance_issues.append(f"Message {i} missing 'role' or 'content'")
                        score -= 0.1
        
        elif expected_format == 'qa_format':
            # Check for question and answer fields
            if 'question' not in data:
                compliance_issues.append("Missing 'question' field")
                score -= 0.4
            if 'answer' not in data:
                compliance_issues.append("Missing 'answer' field")
                score -= 0.4
        
        elif expected_format == 'instruction_format':
            # Check for instruction and output fields
            if 'instruction' not in data:
                compliance_issues.append("Missing 'instruction' field")
                score -= 0.4
            if 'output' not in data:
                compliance_issues.append("Missing 'output' field")
                score -= 0.4
        
        score = max(0.0, score)
        
        return {
            'compliance_issues': compliance_issues,
            'score': score,
            'issue_count': len(compliance_issues)
        }
    
    def check_required_fields(self, data: Dict, expected_format: str) -> Dict[str, Any]:
        """Check for required fields in dictionary data"""
        
        required_fields = {
            'qa_format': ['question', 'answer'],
            'chat_format': ['messages'],
            'instruction_format': ['instruction', 'output'],
            'narrative_format': ['content']
        }
        
        format_fields = required_fields.get(expected_format, [])
        missing_fields = [field for field in format_fields if field not in data]
        present_fields = [field for field in format_fields if field in data]
        
        score = len(present_fields) / len(format_fields) if format_fields else 1.0
        
        return {
            'required_fields': format_fields,
            'present_fields': present_fields,
            'missing_fields': missing_fields,
            'score': score
        }
    
    def validate_field_types(self, data: Dict, expected_format: str) -> Dict[str, Any]:
        """Validate field types in dictionary data"""
        
        type_issues = []
        score = 1.0
        
        for field, value in data.items():
            if field == 'messages' and expected_format == 'chat_format':
                if not isinstance(value, list):
                    type_issues.append(f"Field '{field}' should be list, got {type(value).__name__}")
                    score -= 0.3
            elif field in ['question', 'answer', 'instruction', 'output', 'content']:
                if not isinstance(value, str):
                    type_issues.append(f"Field '{field}' should be string, got {type(value).__name__}")
                    score -= 0.2
        
        score = max(0.0, score)
        
        return {
            'type_issues': type_issues,
            'score': score,
            'issue_count': len(type_issues)
        }
    
    def validate_nested_structure(self, data: Dict, expected_format: str) -> Dict[str, Any]:
        """Validate nested structure in dictionary data"""
        
        nested_issues = []
        score = 1.0
        
        if expected_format == 'chat_format' and 'messages' in data:
            messages = data['messages']
            if isinstance(messages, list):
                for i, message in enumerate(messages):
                    if isinstance(message, dict):
                        if 'role' not in message:
                            nested_issues.append(f"Message {i} missing 'role' field")
                            score -= 0.1
                        elif message['role'] not in ['system', 'user', 'assistant']:
                            nested_issues.append(f"Message {i} has invalid role: {message['role']}")
                            score -= 0.1
                        
                        if 'content' not in message:
                            nested_issues.append(f"Message {i} missing 'content' field")
                            score -= 0.1
                        elif not isinstance(message['content'], str):
                            nested_issues.append(f"Message {i} content should be string")
                            score -= 0.1
        
        score = max(0.0, score)
        
        return {
            'nested_issues': nested_issues,
            'score': score,
            'issue_count': len(nested_issues)
        }
    
    def assess_data_completeness(self, data: Dict, expected_format: str) -> Dict[str, Any]:
        """Assess completeness of data content"""
        
        completeness_issues = []
        score = 1.0
        
        # Check for empty or very short content
        for field, value in data.items():
            if isinstance(value, str):
                if not value.strip():
                    completeness_issues.append(f"Field '{field}' is empty")
                    score -= 0.3
                elif len(value.strip()) < 5:
                    completeness_issues.append(f"Field '{field}' is very short")
                    score -= 0.1
            elif isinstance(value, list):
                if not value:
                    completeness_issues.append(f"Field '{field}' is empty list")
                    score -= 0.3
        
        score = max(0.0, score)
        
        return {
            'completeness_issues': completeness_issues,
            'score': score,
            'issue_count': len(completeness_issues)
        }
    
    def compare_structure_preservation(self, current_text: str, original_text: str, expected_format: str) -> float:
        """Compare structure preservation between original and current text"""
        
        try:
            # Detect formats
            original_format = self.detect_format(original_text)
            current_format = self.detect_format(current_text)
            
            # Base score for format preservation
            format_preservation = 1.0 if original_format == current_format else 0.5
            
            # Check structure indicators preservation
            original_indicators = self.check_structure_indicators(original_text, expected_format)
            current_indicators = self.check_structure_indicators(current_text, expected_format)
            
            indicator_preservation = current_indicators['score'] / max(original_indicators['score'], 0.1)
            indicator_preservation = min(1.0, indicator_preservation)
            
            # Check element preservation
            original_elements = self.check_element_completeness(original_text, expected_format)
            current_elements = self.check_element_completeness(current_text, expected_format)
            
            element_preservation = current_elements['score'] / max(original_elements['score'], 0.1)
            element_preservation = min(1.0, element_preservation)
            
            # Combined preservation score
            preservation_score = (format_preservation * 0.4 + 
                                indicator_preservation * 0.3 + 
                                element_preservation * 0.3)
            
            return preservation_score
        
        except Exception as e:
            self.logger.warning(f"Structure preservation comparison failed: {str(e)}")
            return 0.5
    
    def calculate_format_drift(self, current_text: str, original_text: str, expected_format: str) -> float:
        """Calculate format drift between original and current text"""
        
        try:
            preservation_score = self.compare_structure_preservation(current_text, original_text, expected_format)
            drift_score = 1.0 - preservation_score
            return drift_score
        
        except Exception as e:
            self.logger.warning(f"Format drift calculation failed: {str(e)}")
            return 0.5
    
    def compare_dict_structure_preservation(self, current_data: Dict, original_data: Dict, expected_format: str) -> float:
        """Compare structure preservation between original and current dictionary data"""
        
        try:
            # Check field preservation
            original_fields = set(original_data.keys())
            current_fields = set(current_data.keys())
            
            field_preservation = len(original_fields.intersection(current_fields)) / len(original_fields) if original_fields else 1.0
            
            # Check structure preservation for specific formats
            if expected_format == 'chat_format':
                if 'messages' in original_data and 'messages' in current_data:
                    original_msg_count = len(original_data['messages']) if isinstance(original_data['messages'], list) else 0
                    current_msg_count = len(current_data['messages']) if isinstance(current_data['messages'], list) else 0
                    
                    msg_preservation = min(current_msg_count / max(original_msg_count, 1), 1.0)
                    return (field_preservation + msg_preservation) / 2
            
            return field_preservation
        
        except Exception as e:
            self.logger.warning(f"Dict structure preservation comparison failed: {str(e)}")
            return 0.5
    
    def calculate_schema_drift(self, current_data: Dict, original_data: Dict, expected_format: str) -> float:
        """Calculate schema drift between original and current dictionary data"""
        
        try:
            preservation_score = self.compare_dict_structure_preservation(current_data, original_data, expected_format)
            drift_score = 1.0 - preservation_score
            return drift_score
        
        except Exception as e:
            self.logger.warning(f"Schema drift calculation failed: {str(e)}")
            return 0.5
    
    def calculate_overall_compliance(self, results: Dict[str, Any]) -> float:
        """Calculate overall structure compliance score"""
        
        # Collect relevant scores
        scores = []
        
        # Text-based scores
        if 'pattern_matches' in results:
            scores.append(results['pattern_matches'].get('score', 0.5))
        if 'structure_indicators' in results:
            scores.append(results['structure_indicators'].get('score', 0.5))
        if 'element_completeness' in results:
            scores.append(results['element_completeness'].get('score', 0.5))
        if 'format_consistency' in results:
            scores.append(results['format_consistency'].get('score', 0.5))
        
        # Dict-based scores
        if 'schema_compliance' in results:
            scores.append(results['schema_compliance'].get('score', 0.5))
        if 'required_fields' in results:
            scores.append(results['required_fields'].get('score', 0.5))
        if 'field_types' in results:
            scores.append(results['field_types'].get('score', 0.5))
        
        # Calculate weighted average
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.5
    
    def calculate_structure_preservation(self, results: Dict[str, Any], original_content: Optional[Union[str, Dict]]) -> float:
        """Calculate structure preservation score"""
        
        if original_content is None:
            return 1.0  # No original to compare against
        
        # Use existing preservation scores
        if 'structure_preservation_score' in results:
            return results['structure_preservation_score']
        else:
            return 0.8  # Default reasonable preservation
    
    def calculate_format_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in format detection and validation"""
        
        # Collect confidence indicators
        confidence_factors = []
        
        # Pattern match confidence
        if 'pattern_matches' in results:
            pattern_score = results['pattern_matches'].get('score', 0.5)
            confidence_factors.append(pattern_score)
        
        # Structure indicator confidence
        if 'structure_indicators' in results:
            indicator_score = results['structure_indicators'].get('score', 0.5)
            confidence_factors.append(indicator_score)
        
        # Element completeness confidence
        if 'element_completeness' in results:
            element_score = results['element_completeness'].get('score', 0.5)
            confidence_factors.append(element_score)
        
        # Schema compliance confidence
        if 'schema_compliance' in results:
            schema_score = results['schema_compliance'].get('score', 0.5)
            confidence_factors.append(schema_score)
        
        # Calculate confidence based on consistency of factors
        if confidence_factors:
            mean_score = sum(confidence_factors) / len(confidence_factors)
            variance = sum((score - mean_score) ** 2 for score in confidence_factors) / len(confidence_factors)
            
            # Higher confidence when scores are consistent and high
            confidence = mean_score * (1.0 - min(variance, 0.5))
            return confidence
        else:
            return 0.5
    
    def assess_structure_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall structure quality"""
        
        overall_score = results.get('overall_compliance', 0.5)
        confidence = results.get('format_confidence', 0.5)
        
        # Determine quality level
        if overall_score >= self.thresholds['excellent'] and confidence >= 0.8:
            quality_level = 'excellent'
            status = 'pass'
            message = "Excellent structure compliance"
        elif overall_score >= self.thresholds['good'] and confidence >= 0.6:
            quality_level = 'good'
            status = 'pass'
            message = "Good structure compliance"
        elif overall_score >= self.thresholds['acceptable']:
            quality_level = 'acceptable'
            status = 'pass'
            message = "Acceptable structure compliance"
        elif overall_score >= self.thresholds['concerning']:
            quality_level = 'concerning'
            status = 'review'
            message = "Concerning structure issues"
        else:
            quality_level = 'poor'
            status = 'fail'
            message = "Poor structure compliance"
        
        # Generate recommendations
        recommendations = self.generate_structure_recommendations(results)
        
        return {
            'quality_level': quality_level,
            'status': status,
            'message': message,
            'recommendations': recommendations,
            'pass_threshold': overall_score >= self.thresholds['acceptable'],
            'confidence': confidence
        }
    
    def generate_structure_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for structure improvement"""
        
        recommendations = []
        expected_format = results.get('expected_format', 'unknown')
        
        # Pattern match recommendations
        if 'pattern_matches' in results and results['pattern_matches'].get('score', 0.5) < 0.7:
            recommendations.append(f"Improve adherence to {expected_format.replace('_', ' ')} patterns")
        
        # Element completeness recommendations
        if 'element_completeness' in results:
            missing_elements = results['element_completeness'].get('missing_elements', [])
            if missing_elements:
                recommendations.append(f"Add missing elements: {', '.join(missing_elements)}")
        
        # Format consistency recommendations
        if 'format_consistency' in results:
            consistency_issues = results['format_consistency'].get('consistency_issues', [])
            if consistency_issues:
                recommendations.append("Address format consistency issues")
        
        # Schema compliance recommendations
        if 'schema_compliance' in results:
            compliance_issues = results['schema_compliance'].get('compliance_issues', [])
            if compliance_issues:
                recommendations.append("Fix schema compliance issues")
        
        # Structure clarity recommendations
        if 'structure_clarity' in results and results['structure_clarity'].get('clarity_score', 1.0) < 0.8:
            recommendations.append("Improve structure clarity and organization")
        
        return recommendations
    
    def get_fallback_structure_results(self, expected_format: str) -> Dict[str, Any]:
        """Return fallback results when validation fails"""
        
        return {
            'expected_format': expected_format,
            'content_type': 'unknown',
            'format_detected': 'unknown',
            'overall_compliance': 0.5,
            'structure_preservation': 0.5,
            'format_confidence': 0.3,
            'quality_assessment': {
                'quality_level': 'unknown',
                'status': 'review',
                'message': 'Structure validation failed - manual review required',
                'recommendations': ['Manual structure review required due to validation error'],
                'pass_threshold': False,
                'confidence': 0.3
            },
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def render_structure_validation(self, content: Union[str, Dict], expected_format: str, original_content: Optional[Union[str, Dict]] = None) -> Dict[str, Any]:
        """Render structure validation interface"""
        
        st.subheader("🏗️ Structure Validation")
        
        # Validate structure
        with st.spinner("Validating structure..."):
            results = self.validate_structure(content, expected_format, original_content)
        
        # Display results
        self.render_structure_metrics(results)
        self.render_structure_assessment(results)
        self.render_structure_recommendations(results)
        
        return results
    
    def render_structure_metrics(self, results: Dict[str, Any]):
        """Render structure validation metrics"""
        
        st.markdown("**🏗️ Structure Metrics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            compliance = results.get('overall_compliance', 0.5)
            st.metric(
                "Overall Compliance",
                f"{compliance:.3f}",
                delta=f"{compliance - 0.8:.3f}" if compliance != 0.8 else None
            )
        
        with col2:
            preservation = results.get('structure_preservation', 0.5)
            st.metric(
                "Structure Preservation",
                f"{preservation:.3f}",
                delta=f"{preservation - 0.9:.3f}" if preservation != 0.9 else None
            )
        
        with col3:
            confidence = results.get('format_confidence', 0.5)
            st.metric(
                "Format Confidence",
                f"{confidence:.3f}",
                delta=f"{confidence - 0.8:.3f}" if confidence != 0.8 else None
            )
        
        with col4:
            detected_format = results.get('format_detected', 'unknown')
            expected_format = results.get('expected_format', 'unknown')
            format_match = "✅" if detected_format == expected_format else "❌"
            st.metric(
                "Format Match",
                format_match,
                delta=f"Detected: {detected_format}"
            )
    
    def render_structure_assessment(self, results: Dict[str, Any]):
        """Render structure quality assessment"""
        
        assessment = results.get('quality_assessment', {})
        quality_level = assessment.get('quality_level', 'unknown')
        status = assessment.get('status', 'review')
        message = assessment.get('message', 'Unknown status')
        confidence = assessment.get('confidence', 0.5)
        
        st.markdown("**🏗️ Structure Quality Assessment:**")
        
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
    
    def render_structure_recommendations(self, results: Dict[str, Any]):
        """Render structure improvement recommendations"""
        
        assessment = results.get('quality_assessment', {})
        recommendations = assessment.get('recommendations', [])
        
        if recommendations:
            st.markdown("**💡 Structure Improvement Recommendations:**")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations - structure validation looks good!")

# Integration function for main app
def validate_content_structure(content: Union[str, Dict], expected_format: str, original_content: Optional[Union[str, Dict]] = None) -> Dict[str, Any]:
    """
    Validate content structure
    
    Usage:
    from modules.structure_validator import validate_content_structure
    
    results = validate_content_structure(enhanced_content, 'qa_format', original_content)
    """
    
    validator = StructureValidator()
    return validator.validate_structure(content, expected_format, original_content)

# Quick structure check
def quick_structure_check(content: Union[str, Dict], expected_format: str) -> float:
    """
    Quick structure validation returning compliance score
    
    Usage:
    from modules.structure_validator import quick_structure_check
    
    score = quick_structure_check(content, 'chat_format')
    """
    
    validator = StructureValidator()
    results = validator.validate_structure(content, expected_format)
    return results.get('overall_compliance', 0.5)

if __name__ == "__main__":
    # Test the structure validator
    st.set_page_config(page_title="Structure Validator Test", layout="wide")
    
    st.title("Structure Validator Test")
    
    # Sample content
    content = st.text_area(
        "Content to Validate",
        value="Q: What is machine learning?\nA: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        height=100
    )
    
    # Expected format selection
    expected_format = st.selectbox(
        "Expected Format",
        ['qa_format', 'chat_format', 'instruction_format', 'narrative_format']
    )
    
    if st.button("Validate Structure"):
        validator = StructureValidator()
        results = validator.render_structure_validation(content, expected_format)

