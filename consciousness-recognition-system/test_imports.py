#!/usr/bin/env python3
"""
Import Test Script for Enhanced Universal AI Training Data Creator
================================================================

This script tests all module imports to ensure they work correctly.
Run this before starting the main application to verify everything is set up properly.
"""

import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def test_imports():
    """Test all module imports"""
    
    print("🧪 Testing Enhanced Universal AI Training Data Creator imports...")
    print("=" * 60)
    
    # Test core module imports
    modules_to_test = [
        ('manual_review', 'ManualReviewInterface'),
        ('dynamic_prompt_engine', 'DynamicPromptEngine'),
        ('smart_content_detector', 'SmartContentDetector'),
        ('enhanced_comparison_viewer', 'EnhancedComparisonViewer'),
        ('enhanced_sidebar_metrics', 'EnhancedSidebarMetrics'),
        ('enhanced_theming', 'EnhancedTheming'),
        ('enhanced_zip_export', 'EnhancedZipExporter'),
        ('enhanced_huggingface_upload', 'EnhancedHuggingFaceUploader'),
        ('enhanced_universal_extractor', 'EnhancedUniversalExtractor'),
        ('enhanced_custom_prompt_engine', 'EnhancedCustomPromptEngine'),
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            class_obj = getattr(module, class_name)
            print(f"✅ {class_name} imported successfully from {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
        except AttributeError as e:
            print(f"❌ Failed to find {class_name} in {module_name}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error importing {module_name}: {e}")
    
    print("=" * 60)
    
    if success_count == total_count:
        print("🎉 ALL IMPORTS SUCCESSFUL!")
        print("✅ Enhanced Universal AI Training Data Creator is ready to run!")
        return True
    else:
        print(f"⚠️  {success_count}/{total_count} imports successful")
        print("❌ Some modules failed to import. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

