"""
Comprehensive Test Suite for Enhanced Validation Framework
=========================================================

Tests all validation layers to ensure the "expected string or bytes-like object, 
got 'list'" error is completely eliminated and the system is bulletproof.

Author: Consciousness Recognition System
Version: 1.0 - Comprehensive Validation Testing
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import traceback

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import validation modules
try:
    from modules.text_validator_enhanced import (
        EnhancedTextValidator, validate_text_enhanced, validate_extraction_enhanced,
        emergency_text_fix_enhanced
    )
    from modules.early_type_checker import (
        EarlyTypeChecker, check_text_type, emergency_text_conversion,
        ensure_string, ensure_list, ensure_dict
    )
    from modules.enhanced_detector import EnhancedDialogueDetector
    from modules.universal_extractor import UniversalTextExtractor
    
    print("✅ All validation modules imported successfully")
    
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)


class ValidationTestSuite:
    """Comprehensive test suite for validation framework."""
    
    def __init__(self):
        """Initialize test suite."""
        self.text_validator = EnhancedTextValidator(strict_mode=False, debug_mode=True)
        self.type_checker = EarlyTypeChecker(auto_convert=True, emergency_mode=True)
        self.detector = EnhancedDialogueDetector()
        self.extractor = UniversalTextExtractor()
        
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("🧪 Starting Comprehensive Validation Test Suite")
        print("=" * 60)
        
        # Test categories
        test_categories = [
            ("Text Validator Tests", self._test_text_validator),
            ("Type Checker Tests", self._test_type_checker),
            ("Integration Tests", self._test_integration),
            ("Edge Case Tests", self._test_edge_cases),
            ("Error Recovery Tests", self._test_error_recovery),
            ("Performance Tests", self._test_performance)
        ]
        
        for category_name, test_function in test_categories:
            print(f"\n📋 {category_name}")
            print("-" * 40)
            
            try:
                test_function()
            except Exception as e:
                self._record_error(f"{category_name} failed", e)
        
        # Final results
        self._print_final_results()
        return self.test_results
    
    def _test_text_validator(self):
        """Test enhanced text validator."""
        
        # Test 1: String input (should pass)
        self._run_test(
            "String input validation",
            lambda: self.text_validator.validate_and_normalize_text("Test string", "test_string"),
            expected_type=str
        )
        
        # Test 2: List input (should convert)
        test_list = ["Line 1", "Line 2", "Line 3"]
        self._run_test(
            "List input conversion",
            lambda: self.text_validator.validate_and_normalize_text(test_list, "test_list"),
            expected_type=str,
            should_contain="Line 1"
        )
        
        # Test 3: Dictionary input (should extract text)
        test_dict = {"text": "Extracted content", "other": "ignored"}
        self._run_test(
            "Dictionary input extraction",
            lambda: self.text_validator.validate_and_normalize_text(test_dict, "test_dict"),
            expected_type=str,
            should_contain="Extracted content"
        )
        
        # Test 4: Bytes input (should decode)
        test_bytes = "Test bytes content".encode('utf-8')
        self._run_test(
            "Bytes input decoding",
            lambda: self.text_validator.validate_and_normalize_text(test_bytes, "test_bytes"),
            expected_type=str,
            should_contain="Test bytes content"
        )
        
        # Test 5: Nested list (should flatten)
        nested_list = [["Item 1", "Item 2"], ["Item 3", "Item 4"]]
        self._run_test(
            "Nested list flattening",
            lambda: self.text_validator.validate_and_normalize_text(nested_list, "test_nested"),
            expected_type=str,
            should_contain="Item 1"
        )
        
        # Test 6: Empty input handling
        self._run_test(
            "Empty input handling",
            lambda: self.text_validator.validate_and_normalize_text("", "test_empty"),
            expected_type=str
        )
        
        # Test 7: None input handling
        self._run_test(
            "None input handling",
            lambda: self.text_validator.validate_and_normalize_text(None, "test_none"),
            expected_type=str
        )
    
    def _test_type_checker(self):
        """Test early type checker."""
        
        # Test 1: Correct type (should pass)
        is_valid, result, message = check_text_type("Test string", "type_test_1")
        self._assert_test("Correct type check", is_valid and isinstance(result, str))
        
        # Test 2: List to string conversion
        test_list = ["Item A", "Item B"]
        is_valid, result, message = check_text_type(test_list, "type_test_2")
        self._assert_test("List to string conversion", is_valid and isinstance(result, str) and "Item A" in result)
        
        # Test 3: Dictionary to string conversion
        test_dict = {"text": "Dictionary content"}
        is_valid, result, message = check_text_type(test_dict, "type_test_3")
        self._assert_test("Dict to string conversion", is_valid and isinstance(result, str))
        
        # Test 4: Emergency conversion
        emergency_result = emergency_text_conversion({"complex": ["nested", "data"]}, "emergency_test")
        self._assert_test("Emergency conversion", isinstance(emergency_result, str))
        
        # Test 5: Ensure functions
        string_result = ensure_string(["List", "to", "string"], "ensure_test")
        self._assert_test("Ensure string function", isinstance(string_result, str))
        
        list_result = ensure_list("String to list", "ensure_test")
        self._assert_test("Ensure list function", isinstance(list_result, list))
        
        dict_result = ensure_dict("String to dict", "ensure_test")
        self._assert_test("Ensure dict function", isinstance(dict_result, dict))
    
    def _test_integration(self):
        """Test integration between validation components."""
        
        # Test 1: Full pipeline with string
        test_string = "Q: What is consciousness? A: Pure awareness itself."
        
        # Step 1: Enhanced validation
        validated_text = validate_text_enhanced(test_string, "integration_test_1")
        self._assert_test("Integration step 1 - validation", isinstance(validated_text, str))
        
        # Step 2: Type checking
        is_valid, checked_text, message = check_text_type(validated_text, "integration_test_1")
        self._assert_test("Integration step 2 - type check", is_valid and isinstance(checked_text, str))
        
        # Step 3: Detector (this was the original failure point)
        try:
            detection_result = self.detector.detect_dialogues_with_progress(
                checked_text, mode="regex", show_progress=False
            )
            self._assert_test("Integration step 3 - detector", detection_result['success'])
        except Exception as e:
            self._record_error("Detector integration test", e)
        
        # Test 2: Full pipeline with list (the problematic case)
        test_list = [
            "Q: What is the nature of reality?",
            "A: Reality is consciousness appearing as the world.",
            "Q: How do we realize this?",
            "A: Through direct inquiry and presence."
        ]
        
        # Step 1: Enhanced validation
        validated_list_text = validate_text_enhanced(test_list, "integration_test_2")
        self._assert_test("Integration list step 1 - validation", isinstance(validated_list_text, str))
        
        # Step 2: Type checking
        is_valid, checked_list_text, message = check_text_type(validated_list_text, "integration_test_2")
        self._assert_test("Integration list step 2 - type check", is_valid and isinstance(checked_list_text, str))
        
        # Step 3: Detector (critical test - this should NOT fail)
        try:
            detection_result = self.detector.detect_dialogues_with_progress(
                checked_list_text, mode="regex", show_progress=False
            )
            self._assert_test("Integration list step 3 - detector", detection_result['success'])
            
            # Verify dialogues were found
            dialogues = detection_result.get('dialogues', [])
            self._assert_test("Integration list step 4 - dialogues found", len(dialogues) > 0)
            
        except Exception as e:
            self._record_error("Detector integration test with list", e)
    
    def _test_edge_cases(self):
        """Test edge cases and problematic inputs."""
        
        # Test 1: Very large list
        large_list = [f"Item {i}" for i in range(1000)]
        self._run_test(
            "Large list handling",
            lambda: validate_text_enhanced(large_list, "large_list_test"),
            expected_type=str
        )
        
        # Test 2: Deeply nested structure
        nested_structure = {
            "level1": {
                "level2": {
                    "level3": ["Deep", "nested", "content"]
                }
            }
        }
        self._run_test(
            "Deeply nested structure",
            lambda: validate_text_enhanced(nested_structure, "nested_test"),
            expected_type=str
        )
        
        # Test 3: Mixed types in list
        mixed_list = ["String", 123, {"key": "value"}, None, ["nested", "list"]]
        self._run_test(
            "Mixed types in list",
            lambda: validate_text_enhanced(mixed_list, "mixed_test"),
            expected_type=str
        )
        
        # Test 4: Unicode and special characters
        unicode_text = "Spiritual text with unicode: 🧘 ॐ नमः शिवाय"
        self._run_test(
            "Unicode handling",
            lambda: validate_text_enhanced(unicode_text, "unicode_test"),
            expected_type=str,
            should_contain="🧘"
        )
        
        # Test 5: Very long string
        long_string = "A" * 100000
        self._run_test(
            "Very long string",
            lambda: validate_text_enhanced(long_string, "long_string_test"),
            expected_type=str
        )
        
        # Test 6: Empty containers
        empty_cases = [[], {}, "", None]
        for i, empty_case in enumerate(empty_cases):
            self._run_test(
                f"Empty case {i+1}: {type(empty_case).__name__}",
                lambda case=empty_case: validate_text_enhanced(case, f"empty_test_{i}"),
                expected_type=str
            )
    
    def _test_error_recovery(self):
        """Test error recovery mechanisms."""
        
        # Test 1: Corrupted data recovery
        class CorruptedData:
            def __str__(self):
                raise Exception("Corrupted data cannot be converted to string")
        
        corrupted = CorruptedData()
        self._run_test(
            "Corrupted data recovery",
            lambda: emergency_text_fix_enhanced(corrupted, "corrupted_test"),
            expected_type=str
        )
        
        # Test 2: Circular reference handling
        circular_dict = {}
        circular_dict['self'] = circular_dict
        
        self._run_test(
            "Circular reference handling",
            lambda: emergency_text_fix_enhanced(circular_dict, "circular_test"),
            expected_type=str
        )
        
        # Test 3: Invalid encoding recovery
        invalid_bytes = b'\xff\xfe\xfd'
        self._run_test(
            "Invalid encoding recovery",
            lambda: validate_text_enhanced(invalid_bytes, "invalid_encoding_test"),
            expected_type=str
        )
    
    def _test_performance(self):
        """Test performance with various input sizes."""
        
        import time
        
        # Test 1: Small input performance
        small_input = ["Small"] * 10
        start_time = time.time()
        result = validate_text_enhanced(small_input, "perf_small")
        small_time = time.time() - start_time
        
        self._assert_test("Small input performance", small_time < 1.0)  # Should be very fast
        
        # Test 2: Medium input performance
        medium_input = ["Medium"] * 1000
        start_time = time.time()
        result = validate_text_enhanced(medium_input, "perf_medium")
        medium_time = time.time() - start_time
        
        self._assert_test("Medium input performance", medium_time < 5.0)  # Should be reasonable
        
        # Test 3: Large input performance
        large_input = ["Large"] * 10000
        start_time = time.time()
        result = validate_text_enhanced(large_input, "perf_large")
        large_time = time.time() - start_time
        
        self._assert_test("Large input performance", large_time < 30.0)  # Should complete
        
        print(f"Performance results: Small={small_time:.3f}s, Medium={medium_time:.3f}s, Large={large_time:.3f}s")
    
    def _run_test(self, test_name: str, test_function, expected_type=None, should_contain=None):
        """Run a single test with error handling."""
        try:
            result = test_function()
            
            # Type check
            if expected_type and not isinstance(result, expected_type):
                self._record_error(test_name, f"Expected {expected_type}, got {type(result)}")
                return
            
            # Content check
            if should_contain and should_contain not in str(result):
                self._record_error(test_name, f"Result should contain '{should_contain}'")
                return
            
            self._assert_test(test_name, True)
            
        except Exception as e:
            self._record_error(test_name, e)
    
    def _assert_test(self, test_name: str, condition: bool):
        """Assert a test condition."""
        if condition:
            print(f"✅ {test_name}")
            self.test_results['passed'] += 1
        else:
            print(f"❌ {test_name}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"{test_name}: Assertion failed")
    
    def _record_error(self, test_name: str, error):
        """Record a test error."""
        print(f"❌ {test_name}: {str(error)}")
        self.test_results['failed'] += 1
        self.test_results['errors'].append(f"{test_name}: {str(error)}")
    
    def _print_final_results(self):
        """Print final test results."""
        print("\n" + "=" * 60)
        print("🏁 FINAL TEST RESULTS")
        print("=" * 60)
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        success_rate = (self.test_results['passed'] / max(total_tests, 1)) * 100
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.test_results['passed']}")
        print(f"Failed: {self.test_results['failed']}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.test_results['errors']:
            print("\n❌ ERRORS:")
            for error in self.test_results['errors']:
                print(f"  • {error}")
        
        if success_rate >= 95:
            print("\n🎉 EXCELLENT! Validation framework is working perfectly!")
        elif success_rate >= 90:
            print("\n✅ GOOD! Validation framework is working well with minor issues.")
        elif success_rate >= 80:
            print("\n⚠️ FAIR! Validation framework needs some improvements.")
        else:
            print("\n❌ POOR! Validation framework has significant issues.")


def create_test_pdf():
    """Create a test PDF for integration testing."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
        
        # Create PDF content
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Add test content that should trigger the list issue
        test_content = [
            "Q: What is the nature of consciousness?",
            "A: Consciousness is the fundamental reality underlying all experience.",
            "",
            "Q: How can we directly experience this truth?",
            "A: Through meditation, self-inquiry, and present moment awareness.",
            "",
            "Q: What is the relationship between awareness and thoughts?",
            "A: Thoughts arise within awareness but are not awareness itself."
        ]
        
        y_position = 750
        for line in test_content:
            c.drawString(100, y_position, line)
            y_position -= 30
        
        c.save()
        return pdf_path
        
    except ImportError:
        print("⚠️ reportlab not available, skipping PDF creation test")
        return None


def test_real_pdf_processing():
    """Test with a real PDF to ensure no list errors occur."""
    print("\n📄 Testing Real PDF Processing")
    print("-" * 40)
    
    pdf_path = create_test_pdf()
    if not pdf_path:
        print("⚠️ Skipping PDF test - reportlab not available")
        return
    
    try:
        # Initialize components
        extractor = UniversalTextExtractor()
        detector = EnhancedDialogueDetector()
        
        # Extract text
        print("📖 Extracting text from test PDF...")
        extraction_result = extractor.extract_text(pdf_path)
        
        # Validate extraction
        print("🛡️ Validating extraction result...")
        text = validate_extraction_enhanced(extraction_result, "test_pdf")
        
        if text is None:
            print("❌ Extraction validation failed")
            return
        
        # Type check
        print("🔍 Type checking extracted text...")
        is_valid, validated_text, message = check_text_type(text, "test_pdf_text")
        
        if not is_valid:
            print(f"❌ Type check failed: {message}")
            return
        
        # Run detector (the critical test)
        print("🎯 Running dialogue detection...")
        detection_result = detector.detect_dialogues_with_progress(
            validated_text, mode="regex", show_progress=False
        )
        
        if detection_result['success']:
            dialogues = detection_result.get('dialogues', [])
            print(f"✅ PDF processing successful! Found {len(dialogues)} dialogues")
        else:
            print(f"❌ Detection failed: {detection_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ PDF processing failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Clean up
        try:
            os.unlink(pdf_path)
        except:
            pass


def main():
    """Run the complete test suite."""
    print("🧘 Consciousness Recognition System - Enhanced Validation Test Suite")
    print("=" * 80)
    
    # Run validation tests
    test_suite = ValidationTestSuite()
    results = test_suite.run_all_tests()
    
    # Test real PDF processing
    test_real_pdf_processing()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 VALIDATION FRAMEWORK TEST SUMMARY")
    print("=" * 80)
    
    if results['failed'] == 0:
        print("🎉 ALL TESTS PASSED! The validation framework is bulletproof!")
        print("✅ The 'expected string or bytes-like object, got list' error is eliminated!")
        return True
    else:
        print(f"⚠️ {results['failed']} tests failed. Review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

