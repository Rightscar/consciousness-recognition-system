"""
Testing Support Module
=====================

Provides testing infrastructure and backend function extraction for the Enhanced Universal AI Training Data Creator.
Separates business logic from UI rendering to enable comprehensive unit testing.

Features:
- Backend function extraction from UI methods
- Test data generation and validation
- Mock data providers for testing
- Test result validation
- Performance benchmarking
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TestingSupport:
    """Testing support utilities and backend function extraction"""
    
    def __init__(self):
        self.test_results = []
        self.performance_benchmarks = {}
    
    # Backend Functions Extracted from UI Methods
    # ===========================================
    
    def compute_content_statistics(self, content: str) -> Dict[str, Any]:
        """
        Extracted backend function: Compute content statistics
        Originally from _render_content_analysis method
        """
        if not content:
            return {
                'total_chars': 0,
                'total_words': 0,
                'total_lines': 0,
                'avg_words_per_line': 0,
                'complexity_score': 0,
                'readability_score': 0,
                'dialogue_ratio': 0
            }
        
        lines = content.splitlines()
        words = content.split()
        
        # Basic statistics
        stats = {
            'total_chars': len(content),
            'total_words': len(words),
            'total_lines': len(lines),
            'avg_words_per_line': len(words) / len(lines) if lines else 0
        }
        
        # Complexity score (vocabulary diversity)
        unique_words = set(word.lower() for word in words)
        stats['complexity_score'] = len(unique_words) / len(words) if words else 0
        
        # Simple readability score (average word length)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        stats['readability_score'] = min(avg_word_length / 6.0, 1.0)  # Normalize to 0-1
        
        # Dialogue ratio (lines with dialogue markers)
        dialogue_lines = sum(1 for line in lines if any(marker in line for marker in ['"', "'", ":", "?"]))
        stats['dialogue_ratio'] = dialogue_lines / len(lines) if lines else 0
        
        return stats
    
    def detect_content_type(self, content: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extracted backend function: Detect content type
        Originally from smart_content_detector logic
        """
        if not content:
            return "unknown", 0.0, {}
        
        lines = content.splitlines()
        total_lines = len(lines)
        
        # Count different patterns
        qa_patterns = 0
        dialogue_patterns = 0
        monologue_patterns = 0
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Q&A patterns
            if any(pattern in line_lower for pattern in ['q:', 'a:', 'question:', 'answer:', '?']):
                qa_patterns += 1
            
            # Dialogue patterns
            elif any(pattern in line for pattern in ['"', "'", ':']):
                dialogue_patterns += 1
            
            # Monologue patterns (longer lines without dialogue markers)
            elif len(line.split()) > 10 and not any(marker in line for marker in ['"', "'", ':']):
                monologue_patterns += 1
        
        # Calculate confidence scores
        qa_score = qa_patterns / total_lines if total_lines > 0 else 0
        dialogue_score = dialogue_patterns / total_lines if total_lines > 0 else 0
        monologue_score = monologue_patterns / total_lines if total_lines > 0 else 0
        
        # Determine content type
        scores = {
            'qa': qa_score,
            'dialogue': dialogue_score,
            'monologue': monologue_score
        }
        
        content_type = max(scores, key=scores.get)
        confidence = scores[content_type]
        
        detection_metadata = {
            'scores': scores,
            'total_lines': total_lines,
            'qa_patterns': qa_patterns,
            'dialogue_patterns': dialogue_patterns,
            'monologue_patterns': monologue_patterns
        }
        
        return content_type, confidence, detection_metadata
    
    def validate_enhancement_quality(self, original: str, enhanced: str) -> Dict[str, Any]:
        """
        Extracted backend function: Validate enhancement quality
        Originally from quality analysis logic
        """
        if not original or not enhanced:
            return {
                'length_ratio': 0,
                'similarity_score': 0,
                'quality_score': 0,
                'flags': ['empty_content'],
                'passed': False
            }
        
        # Length ratio analysis
        length_ratio = len(enhanced) / len(original) if original else 0
        
        # Simple similarity score (word overlap)
        original_words = set(original.lower().split())
        enhanced_words = set(enhanced.lower().split())
        
        if original_words:
            similarity_score = len(original_words & enhanced_words) / len(original_words)
        else:
            similarity_score = 0
        
        # Quality flags
        flags = []
        
        if length_ratio > 3.0:
            flags.append('excessive_expansion')
        elif length_ratio < 0.5:
            flags.append('excessive_reduction')
        
        if similarity_score < 0.3:
            flags.append('low_similarity')
        
        if len(enhanced.split()) < 10:
            flags.append('too_short')
        
        # Overall quality score
        length_score = 1.0 - abs(length_ratio - 1.5) / 2.0  # Optimal ratio around 1.5
        length_score = max(0, min(1, length_score))
        
        quality_score = (similarity_score + length_score) / 2
        
        # Pass/fail threshold
        passed = quality_score >= 0.6 and len(flags) == 0
        
        return {
            'length_ratio': length_ratio,
            'similarity_score': similarity_score,
            'quality_score': quality_score,
            'length_score': length_score,
            'flags': flags,
            'passed': passed
        }
    
    def process_manual_review_data(self, review_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracted backend function: Process manual review data
        Originally from manual review processing logic
        """
        if not review_data:
            return {
                'total_items': 0,
                'approved_items': 0,
                'rejected_items': 0,
                'approval_rate': 0,
                'common_issues': [],
                'processed_data': []
            }
        
        approved = [item for item in review_data if item.get('approved', False)]
        rejected = [item for item in review_data if not item.get('approved', False)]
        
        # Analyze common issues
        issue_counts = {}
        for item in rejected:
            issues = item.get('issues', [])
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Process approved data for export
        processed_data = []
        for item in approved:
            processed_item = {
                'input': item.get('original', ''),
                'output': item.get('enhanced', ''),
                'metadata': {
                    'tone': item.get('tone', ''),
                    'quality_score': item.get('quality_score', 0),
                    'review_timestamp': datetime.now().isoformat()
                }
            }
            processed_data.append(processed_item)
        
        return {
            'total_items': len(review_data),
            'approved_items': len(approved),
            'rejected_items': len(rejected),
            'approval_rate': len(approved) / len(review_data) if review_data else 0,
            'common_issues': common_issues,
            'processed_data': processed_data
        }
    
    def generate_export_metadata(self, data: List[Dict[str, Any]], 
                                export_format: str) -> Dict[str, Any]:
        """
        Extracted backend function: Generate export metadata
        Originally from export processing logic
        """
        if not data:
            return {
                'item_count': 0,
                'total_size': 0,
                'format': export_format,
                'created_at': datetime.now().isoformat(),
                'schema_version': '1.0'
            }
        
        # Calculate statistics
        total_input_chars = sum(len(item.get('input', '')) for item in data)
        total_output_chars = sum(len(item.get('output', '')) for item in data)
        
        # Estimate file size
        if export_format.lower() == 'json':
            estimated_size = len(json.dumps(data, indent=2).encode('utf-8'))
        elif export_format.lower() == 'jsonl':
            estimated_size = sum(len(json.dumps(item).encode('utf-8')) + 1 for item in data)
        else:
            estimated_size = total_input_chars + total_output_chars
        
        return {
            'item_count': len(data),
            'total_input_chars': total_input_chars,
            'total_output_chars': total_output_chars,
            'estimated_size': estimated_size,
            'format': export_format,
            'created_at': datetime.now().isoformat(),
            'schema_version': '1.0',
            'avg_input_length': total_input_chars / len(data) if data else 0,
            'avg_output_length': total_output_chars / len(data) if data else 0
        }
    
    # Test Data Generation
    # ===================
    
    def generate_test_content(self, content_type: str = "qa", length: int = 500) -> str:
        """Generate test content for different content types"""
        
        if content_type == "qa":
            return self._generate_qa_content(length)
        elif content_type == "dialogue":
            return self._generate_dialogue_content(length)
        elif content_type == "monologue":
            return self._generate_monologue_content(length)
        else:
            return "Test content for unknown type."
    
    def _generate_qa_content(self, length: int) -> str:
        """Generate Q&A style test content"""
        qa_pairs = [
            ("Q: What is consciousness?", "A: Consciousness is the state of being aware of one's surroundings and thoughts."),
            ("Q: How can we cultivate mindfulness?", "A: Through regular meditation practice and present-moment awareness."),
            ("Q: What is the nature of reality?", "A: Reality is often understood as the fundamental nature of existence."),
        ]
        
        content = []
        current_length = 0
        
        while current_length < length:
            for q, a in qa_pairs:
                content.append(q)
                content.append(a)
                content.append("")  # Empty line
                current_length += len(q) + len(a) + 2
                
                if current_length >= length:
                    break
        
        return "\n".join(content)
    
    def _generate_dialogue_content(self, length: int) -> str:
        """Generate dialogue style test content"""
        dialogue_lines = [
            'Teacher: "Welcome to our discussion on consciousness."',
            'Student: "I\'m curious about the nature of awareness."',
            'Teacher: "Awareness is like a mirror that reflects all experiences."',
            'Student: "How can we develop deeper awareness?"',
            'Teacher: "Through practice and patient observation."',
        ]
        
        content = []
        current_length = 0
        
        while current_length < length:
            for line in dialogue_lines:
                content.append(line)
                current_length += len(line) + 1
                
                if current_length >= length:
                    break
        
        return "\n".join(content)
    
    def _generate_monologue_content(self, length: int) -> str:
        """Generate monologue style test content"""
        paragraphs = [
            "Consciousness represents one of the most profound mysteries of human existence. It encompasses our ability to be aware of ourselves and our environment, to experience thoughts and emotions, and to reflect upon our own mental states.",
            "The study of consciousness bridges multiple disciplines, from neuroscience and psychology to philosophy and contemplative traditions. Each approach offers unique insights into the nature of awareness and the mechanisms underlying conscious experience.",
            "Meditation and mindfulness practices have been developed across cultures as methods for exploring consciousness directly. These practices often involve sustained attention, present-moment awareness, and the cultivation of insight into the nature of mind.",
        ]
        
        content = []
        current_length = 0
        
        while current_length < length:
            for paragraph in paragraphs:
                content.append(paragraph)
                content.append("")  # Empty line
                current_length += len(paragraph) + 2
                
                if current_length >= length:
                    break
        
        return "\n".join(content)
    
    # Performance Benchmarking
    # =======================
    
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        func_name = func.__name__
        if func_name not in self.performance_benchmarks:
            self.performance_benchmarks[func_name] = []
        
        self.performance_benchmarks[func_name].append(execution_time)
        
        logger.debug(f"BENCHMARK: {func_name} took {execution_time:.3f}s")
        
        return result
    
    def get_benchmark_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance benchmark summary"""
        summary = {}
        
        for func_name, times in self.performance_benchmarks.items():
            summary[func_name] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        return summary
    
    # Test Validation
    # ==============
    
    def validate_test_result(self, expected: Any, actual: Any, tolerance: float = 0.01) -> bool:
        """Validate test results with tolerance for floating point comparisons"""
        if isinstance(expected, float) and isinstance(actual, float):
            return abs(expected - actual) <= tolerance
        elif isinstance(expected, dict) and isinstance(actual, dict):
            return self._validate_dict_result(expected, actual, tolerance)
        else:
            return expected == actual
    
    def _validate_dict_result(self, expected: Dict, actual: Dict, tolerance: float) -> bool:
        """Validate dictionary results recursively"""
        if set(expected.keys()) != set(actual.keys()):
            return False
        
        for key in expected:
            if not self.validate_test_result(expected[key], actual[key], tolerance):
                return False
        
        return True
    
    def run_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        # Test content statistics
        test_content = self.generate_test_content("qa", 200)
        stats = self.compute_content_statistics(test_content)
        
        test_results['total_tests'] += 1
        if stats['total_chars'] > 0 and stats['total_words'] > 0:
            test_results['passed_tests'] += 1
            test_results['test_details'].append({
                'test': 'content_statistics',
                'status': 'PASSED',
                'details': 'Statistics computed correctly'
            })
        else:
            test_results['failed_tests'] += 1
            test_results['test_details'].append({
                'test': 'content_statistics',
                'status': 'FAILED',
                'details': 'Statistics computation failed'
            })
        
        # Test content type detection
        content_type, confidence, metadata = self.detect_content_type(test_content)
        
        test_results['total_tests'] += 1
        if content_type == "qa" and confidence > 0:
            test_results['passed_tests'] += 1
            test_results['test_details'].append({
                'test': 'content_detection',
                'status': 'PASSED',
                'details': f'Detected {content_type} with confidence {confidence:.3f}'
            })
        else:
            test_results['failed_tests'] += 1
            test_results['test_details'].append({
                'test': 'content_detection',
                'status': 'FAILED',
                'details': f'Expected qa, got {content_type} with confidence {confidence:.3f}'
            })
        
        # Test quality validation
        original = "What is consciousness?"
        enhanced = "Consciousness is the state of being aware of one's thoughts, feelings, and surroundings."
        quality = self.validate_enhancement_quality(original, enhanced)
        
        test_results['total_tests'] += 1
        if quality['quality_score'] > 0.5:
            test_results['passed_tests'] += 1
            test_results['test_details'].append({
                'test': 'quality_validation',
                'status': 'PASSED',
                'details': f'Quality score: {quality["quality_score"]:.3f}'
            })
        else:
            test_results['failed_tests'] += 1
            test_results['test_details'].append({
                'test': 'quality_validation',
                'status': 'FAILED',
                'details': f'Low quality score: {quality["quality_score"]:.3f}'
            })
        
        return test_results


# Global testing support instance
testing_support = TestingSupport()

