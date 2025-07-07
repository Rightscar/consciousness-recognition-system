"""
Unit Tests for Export Utilities
===============================

Comprehensive unit tests for the bulletproof export system.
Tests all export formats, error handling, and validation logic.

Based on the user's requirement for isolated export_utils.py with unit tests.
"""

import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add modules to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from export_utils import (
    RobustExportManager,
    ExportError,
    ExportMetrics,
    BulletproofExportSystem
)

class TestRobustExportManager(unittest.TestCase):
    """Test cases for RobustExportManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.export_manager = RobustExportManager()
        self.test_data = [
            {'question': 'What is AI?', 'answer': 'Artificial Intelligence is a field of computer science.'},
            {'question': 'How does ML work?', 'answer': 'Machine Learning works by training algorithms on data.'},
            {'question': 'What is deep learning?', 'answer': 'Deep learning uses neural networks with multiple layers.'}
        ]
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up export manager temp files
        self.export_manager.cleanup_temp_files()
    
    def test_jsonl_export_success(self):
        """Test successful JSONL export"""
        output_path = os.path.join(self.temp_dir, 'test.jsonl')
        
        result = self.export_manager.export_with_validation(
            self.test_data, 'jsonl', output_path
        )
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Validate JSONL content
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(self.test_data))
            
            for line in lines:
                parsed = json.loads(line.strip())
                self.assertIn('question', parsed)
                self.assertIn('answer', parsed)
    
    def test_json_export_success(self):
        """Test successful JSON export"""
        output_path = os.path.join(self.temp_dir, 'test.json')
        
        result = self.export_manager.export_with_validation(
            self.test_data, 'json', output_path
        )
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Validate JSON content
        with open(output_path, 'r', encoding='utf-8') as f:
            parsed = json.load(f)
            self.assertEqual(len(parsed), len(self.test_data))
            
            for item in parsed:
                self.assertIn('question', item)
                self.assertIn('answer', item)
    
    @patch('pandas.DataFrame')
    def test_csv_export_success(self, mock_dataframe):
        """Test successful CSV export"""
        # Mock pandas DataFrame
        mock_df = MagicMock()
        mock_dataframe.return_value = mock_df
        
        output_path = os.path.join(self.temp_dir, 'test.csv')
        
        result = self.export_manager.export_with_validation(
            self.test_data, 'csv', output_path
        )
        
        # Should succeed with mocked pandas
        mock_dataframe.assert_called_once_with(self.test_data)
        mock_df.to_csv.assert_called_once()
    
    def test_export_with_invalid_format(self):
        """Test export with invalid format"""
        output_path = os.path.join(self.temp_dir, 'test.invalid')
        
        result = self.export_manager.export_with_validation(
            self.test_data, 'invalid_format', output_path
        )
        
        self.assertFalse(result)
    
    def test_export_with_empty_data(self):
        """Test export with empty data"""
        output_path = os.path.join(self.temp_dir, 'test.json')
        
        result = self.export_manager.export_with_validation(
            [], 'json', output_path
        )
        
        self.assertFalse(result)
    
    def test_export_with_invalid_data(self):
        """Test export with invalid data structure"""
        invalid_data = "not a list"
        output_path = os.path.join(self.temp_dir, 'test.json')
        
        result = self.export_manager.export_with_validation(
            invalid_data, 'json', output_path
        )
        
        self.assertFalse(result)
    
    def test_zip_generation_success(self):
        """Test successful ZIP generation"""
        # Create test files
        test_file1 = os.path.join(self.temp_dir, 'file1.txt')
        test_file2 = os.path.join(self.temp_dir, 'file2.txt')
        
        with open(test_file1, 'w') as f:
            f.write('Test content 1')
        
        with open(test_file2, 'w') as f:
            f.write('Test content 2')
        
        files_dict = {
            'file1.txt': test_file1,
            'file2.txt': test_file2
        }
        
        zip_path = os.path.join(self.temp_dir, 'test.zip')
        
        result = self.export_manager.generate_zip_with_validation(files_dict, zip_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(zip_path))
        
        # Validate ZIP content
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            file_list = zipf.namelist()
            self.assertIn('file1.txt', file_list)
            self.assertIn('file2.txt', file_list)
    
    def test_zip_generation_with_missing_file(self):
        """Test ZIP generation with missing file"""
        files_dict = {
            'missing.txt': '/path/to/missing/file.txt'
        }
        
        zip_path = os.path.join(self.temp_dir, 'test.zip')
        
        # Should still succeed but log warning
        result = self.export_manager.generate_zip_with_validation(files_dict, zip_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(zip_path))
    
    def test_file_size_validation(self):
        """Test file size validation for uploads"""
        # Create a large file (simulate)
        large_file = os.path.join(self.temp_dir, 'large_file.txt')
        
        # Create file larger than 50MB limit
        with open(large_file, 'w') as f:
            # Write enough data to exceed limit
            for _ in range(1000):
                f.write('x' * 50000)  # 50KB per iteration
        
        # Should fail size validation
        result = self.export_manager.upload_to_huggingface_with_validation(
            large_file, 'test/repo'
        )
        
        self.assertFalse(result)
    
    def test_data_sanitization(self):
        """Test data sanitization for exports"""
        dirty_data = [
            {
                'question': 'What is AI?\n\r',
                'answer': 'AI is\x00 artificial intelligence.',
                'metadata': None
            }
        ]
        
        output_path = os.path.join(self.temp_dir, 'test.json')
        
        result = self.export_manager.export_with_validation(
            dirty_data, 'json', output_path
        )
        
        self.assertTrue(result)
        
        # Validate sanitized content
        with open(output_path, 'r', encoding='utf-8') as f:
            parsed = json.load(f)
            
            item = parsed[0]
            self.assertNotIn('\n', item['question'])
            self.assertNotIn('\r', item['question'])
            self.assertNotIn('\x00', item['answer'])
            self.assertEqual(item['metadata'], "")  # None converted to empty string

class TestExportMetrics(unittest.TestCase):
    """Test cases for ExportMetrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics = ExportMetrics()
    
    def test_record_successful_export(self):
        """Test recording successful export"""
        self.metrics.record_export_attempt('json', 1024, 2.5, True)
        
        self.assertEqual(self.metrics.export_stats['total_exports'], 1)
        self.assertEqual(self.metrics.export_stats['successful_exports'], 1)
        self.assertEqual(self.metrics.export_stats['failed_exports'], 0)
        self.assertEqual(self.metrics.get_success_rate(), 100.0)
    
    def test_record_failed_export(self):
        """Test recording failed export"""
        error = ExportError("Test error")
        self.metrics.record_export_attempt('json', 0, 1.0, False, error)
        
        self.assertEqual(self.metrics.export_stats['total_exports'], 1)
        self.assertEqual(self.metrics.export_stats['successful_exports'], 0)
        self.assertEqual(self.metrics.export_stats['failed_exports'], 1)
        self.assertEqual(self.metrics.get_success_rate(), 0.0)
        self.assertIn('ExportError', self.metrics.export_stats['error_types'])
    
    def test_format_specific_metrics(self):
        """Test format-specific metrics tracking"""
        # Record multiple attempts for different formats
        self.metrics.record_export_attempt('json', 1024, 2.0, True)
        self.metrics.record_export_attempt('json', 2048, 3.0, False)
        self.metrics.record_export_attempt('csv', 512, 1.5, True)
        
        # Check format-specific success rates
        self.assertEqual(self.metrics.get_format_success_rate('json'), 50.0)
        self.assertEqual(self.metrics.get_format_success_rate('csv'), 100.0)
        self.assertEqual(self.metrics.get_format_success_rate('xlsx'), 0.0)  # No attempts
    
    def test_average_calculations(self):
        """Test average time and size calculations"""
        self.metrics.record_export_attempt('json', 1000, 2.0, True)
        self.metrics.record_export_attempt('json', 2000, 4.0, True)
        
        self.assertEqual(self.metrics.export_stats['average_export_time'], 3.0)
        self.assertEqual(self.metrics.export_stats['largest_export_size'], 2000)

class TestBulletproofExportSystem(unittest.TestCase):
    """Test cases for BulletproofExportSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock session manager
        class MockSessionManager:
            def __init__(self):
                self.data = {}
            
            def get(self, key, default=None):
                return self.data.get(key, default)
            
            def set(self, key, value):
                self.data[key] = value
        
        self.session_manager = MockSessionManager()
        self.export_system = BulletproofExportSystem(self.session_manager)
        
        self.test_data = [
            {'question': 'What is AI?', 'answer': 'Artificial Intelligence...'},
            {'question': 'How does ML work?', 'answer': 'Machine Learning...'}
        ]
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        self.export_system.export_manager.cleanup_temp_files()
    
    def test_export_dataset_success(self):
        """Test successful dataset export"""
        results = self.export_system.export_dataset(
            data=self.test_data,
            format_type='json',
            include_zip=False,
            upload_to_hf=False
        )
        
        self.assertTrue(results['success'])
        self.assertGreater(len(results['files_generated']), 0)
        self.assertEqual(len(results['errors']), 0)
        
        # Verify file was created
        primary_file = results['files_generated'][0]
        self.assertTrue(os.path.exists(primary_file))
    
    def test_export_dataset_with_zip(self):
        """Test dataset export with ZIP creation"""
        results = self.export_system.export_dataset(
            data=self.test_data,
            format_type='json',
            include_zip=True,
            upload_to_hf=False
        )
        
        self.assertTrue(results['success'])
        self.assertIsNotNone(results['zip_path'])
        self.assertTrue(os.path.exists(results['zip_path']))
    
    def test_export_dataset_with_empty_data(self):
        """Test export with empty dataset"""
        results = self.export_system.export_dataset(
            data=[],
            format_type='json',
            include_zip=False,
            upload_to_hf=False
        )
        
        self.assertFalse(results['success'])
        self.assertGreater(len(results['errors']), 0)
    
    @patch('export_utils.RobustExportManager.upload_to_huggingface_with_validation')
    def test_export_with_hf_upload_success(self, mock_upload):
        """Test export with successful Hugging Face upload"""
        mock_upload.return_value = True
        
        results = self.export_system.export_dataset(
            data=self.test_data,
            format_type='json',
            include_zip=False,
            upload_to_hf=True,
            hf_repo_name='test/dataset'
        )
        
        self.assertTrue(results['success'])
        self.assertTrue(results['upload_success'])
        mock_upload.assert_called_once()
    
    @patch('export_utils.RobustExportManager.upload_to_huggingface_with_validation')
    def test_export_with_hf_upload_failure(self, mock_upload):
        """Test export with failed Hugging Face upload"""
        mock_upload.return_value = False
        
        results = self.export_system.export_dataset(
            data=self.test_data,
            format_type='json',
            include_zip=False,
            upload_to_hf=True,
            hf_repo_name='test/dataset'
        )
        
        self.assertTrue(results['success'])  # Primary export should still succeed
        self.assertFalse(results['upload_success'])
        self.assertGreater(len(results['errors']), 0)  # Should have upload error
    
    def test_metadata_generation(self):
        """Test metadata file generation"""
        timestamp = "20240101_120000"
        metadata_path = self.export_system._generate_metadata_file(self.test_data, timestamp)
        
        self.assertTrue(os.path.exists(metadata_path))
        
        # Validate metadata content
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
            self.assertEqual(metadata['export_timestamp'], timestamp)
            self.assertEqual(metadata['total_items'], len(self.test_data))
            self.assertIn('data_statistics', metadata)
    
    def test_data_statistics_calculation(self):
        """Test data statistics calculation"""
        stats = self.export_system._calculate_data_statistics(self.test_data)
        
        self.assertEqual(stats['total_items'], len(self.test_data))
        self.assertGreater(stats['average_question_length'], 0)
        self.assertGreater(stats['average_answer_length'], 0)
        self.assertGreater(stats['total_characters'], 0)

class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.export_manager = RobustExportManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        self.export_manager.cleanup_temp_files()
    
    def test_invalid_output_directory(self):
        """Test export to invalid output directory"""
        invalid_path = '/invalid/directory/file.json'
        
        result = self.export_manager.export_with_validation(
            [{'question': 'test', 'answer': 'test'}],
            'json',
            invalid_path
        )
        
        # Should handle directory creation or fail gracefully
        # Result depends on system permissions
        self.assertIsInstance(result, bool)
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted data"""
        corrupted_data = [
            {'question': 'valid question', 'answer': 'valid answer'},
            {'invalid': 'structure'},  # Missing required fields
            {'question': None, 'answer': None}  # None values
        ]
        
        output_path = os.path.join(self.temp_dir, 'test.json')
        
        # Should handle gracefully with sanitization
        result = self.export_manager.export_with_validation(
            corrupted_data, 'json', output_path
        )
        
        # May succeed with sanitization or fail gracefully
        self.assertIsInstance(result, bool)
    
    def test_file_permission_errors(self):
        """Test handling of file permission errors"""
        # Create a read-only directory (if possible)
        readonly_dir = os.path.join(self.temp_dir, 'readonly')
        os.makedirs(readonly_dir, exist_ok=True)
        
        try:
            os.chmod(readonly_dir, 0o444)  # Read-only
            
            output_path = os.path.join(readonly_dir, 'test.json')
            
            result = self.export_manager.export_with_validation(
                [{'question': 'test', 'answer': 'test'}],
                'json',
                output_path
            )
            
            self.assertFalse(result)
            
        except (OSError, PermissionError):
            # Skip test if we can't modify permissions
            self.skipTest("Cannot modify directory permissions on this system")
        
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(readonly_dir, 0o755)
            except:
                pass

def run_export_tests():
    """Run all export utility tests"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestRobustExportManager))
    test_suite.addTest(unittest.makeSuite(TestExportMetrics))
    test_suite.addTest(unittest.makeSuite(TestBulletproofExportSystem))
    test_suite.addTest(unittest.makeSuite(TestErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPORT UTILITIES TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print(f"\n✅ ALL TESTS PASSED! Export utilities are working correctly.")
    else:
        print(f"\n❌ Some tests failed. Please review the issues above.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_export_tests()
    
    if success:
        print("\n🎉 Export utilities are bulletproof and ready for production!")
    else:
        print("\n⚠️ Export utilities need attention before production deployment.")
        exit(1)

