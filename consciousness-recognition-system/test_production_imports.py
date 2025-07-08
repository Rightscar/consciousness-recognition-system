#!/usr/bin/env python3
"""
Production Import Test
=====================

Comprehensive import test for the Enhanced Universal AI Training Data Creator Production Version.
Tests all modules including new production enhancements.
"""

import sys
import os
import traceback
from datetime import datetime

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def test_imports():
    """Test all module imports"""
    
    print("🧪 Enhanced Universal AI Training Data Creator - Production Import Test")
    print("=" * 80)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Core modules to test
    core_modules = [
        ('ManualReviewInterface', 'manual_review'),
        ('DynamicPromptEngine', 'dynamic_prompt_engine'),
        ('SmartContentDetector', 'smart_content_detector'),
        ('EnhancedComparisonViewer', 'enhanced_comparison_viewer'),
        ('EnhancedSidebarMetrics', 'enhanced_sidebar_metrics'),
        ('EnhancedTheming', 'enhanced_theming'),
        ('EnhancedZipExporter', 'enhanced_zip_export'),
        ('EnhancedHuggingFaceUploader', 'enhanced_huggingface_upload'),
        ('EnhancedUniversalExtractor', 'enhanced_universal_extractor'),
        ('EnhancedCustomPromptEngine', 'enhanced_custom_prompt_engine')
    ]
    
    # Production enhancement modules
    production_modules = [
        ('EnhancedCaching', 'enhanced_caching'),
        ('EnhancedDebugging', 'enhanced_debugging'),
        ('EnhancedLogging', 'enhanced_logging'),
        ('TestingSupport', 'testing_support'),
        ('EnhancedQualityControl', 'quality_control_enhanced'),
        ('EnhancedUIPolish', 'ui_polish_enhanced'),
        ('MetadataSchemaValidator', 'metadata_schema_validator'),
        ('ExportConfirmations', 'export_confirmations')
    ]
    
    all_modules = core_modules + production_modules
    
    successful_imports = 0
    failed_imports = 0
    
    print("📦 Testing Core Modules:")
    print("-" * 40)
    
    for class_name, module_name in core_modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {class_name} imported successfully")
            successful_imports += 1
        except ImportError as e:
            print(f"❌ {class_name} import failed: {e}")
            failed_imports += 1
        except AttributeError as e:
            print(f"❌ {class_name} not found in module: {e}")
            failed_imports += 1
        except Exception as e:
            print(f"❌ {class_name} unexpected error: {e}")
            failed_imports += 1
    
    print()
    print("🚀 Testing Production Enhancement Modules:")
    print("-" * 50)
    
    for class_name, module_name in production_modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {class_name} imported successfully")
            successful_imports += 1
        except ImportError as e:
            print(f"❌ {class_name} import failed: {e}")
            failed_imports += 1
        except AttributeError as e:
            print(f"❌ {class_name} not found in module: {e}")
            failed_imports += 1
        except Exception as e:
            print(f"❌ {class_name} unexpected error: {e}")
            failed_imports += 1
    
    print()
    print("🧪 Testing Global Instances:")
    print("-" * 35)
    
    # Test global instances
    global_instances = [
        ('enhanced_caching', 'enhanced_caching'),
        ('enhanced_debugging', 'enhanced_debugging'),
        ('enhanced_logging', 'enhanced_logging'),
        ('testing_support', 'testing_support'),
        ('enhanced_quality_control', 'quality_control_enhanced'),
        ('ui_polish', 'ui_polish_enhanced'),
        ('metadata_validator', 'metadata_schema_validator'),
        ('export_confirmations', 'export_confirmations')
    ]
    
    for instance_name, module_name in global_instances:
        try:
            module = __import__(module_name, fromlist=[instance_name])
            getattr(module, instance_name)
            print(f"✅ {instance_name} instance available")
            successful_imports += 1
        except ImportError as e:
            print(f"❌ {instance_name} import failed: {e}")
            failed_imports += 1
        except AttributeError as e:
            print(f"❌ {instance_name} not found in module: {e}")
            failed_imports += 1
        except Exception as e:
            print(f"❌ {instance_name} unexpected error: {e}")
            failed_imports += 1
    
    print()
    print("🎯 Testing Main Application:")
    print("-" * 35)
    
    # Test main application import
    try:
        sys.path.append('.')
        import enhanced_app_production
        print("✅ enhanced_app_production imported successfully")
        successful_imports += 1
    except ImportError as e:
        print(f"❌ enhanced_app_production import failed: {e}")
        failed_imports += 1
    except Exception as e:
        print(f"❌ enhanced_app_production unexpected error: {e}")
        failed_imports += 1
    
    # Test application class
    try:
        app_class = getattr(enhanced_app_production, 'EnhancedUniversalAITrainingDataCreatorProduction')
        print("✅ EnhancedUniversalAITrainingDataCreatorProduction class available")
        successful_imports += 1
    except AttributeError as e:
        print(f"❌ Application class not found: {e}")
        failed_imports += 1
    except Exception as e:
        print(f"❌ Application class error: {e}")
        failed_imports += 1
    
    print()
    print("=" * 80)
    print("📊 IMPORT TEST SUMMARY")
    print("=" * 80)
    
    total_tests = successful_imports + failed_imports
    success_rate = (successful_imports / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_imports}")
    print(f"Failed: {failed_imports}")
    print(f"Success Rate: {success_rate:.1f}%")
    print()
    
    if failed_imports == 0:
        print("🎉 ALL IMPORTS SUCCESSFUL!")
        print("✅ Enhanced Universal AI Training Data Creator Production is ready to run!")
        print()
        print("🚀 To run the application:")
        print("   streamlit run enhanced_app_production.py")
        return True
    else:
        print("❌ SOME IMPORTS FAILED!")
        print("Please check the failed imports above and ensure all dependencies are installed.")
        print()
        print("📋 To install dependencies:")
        print("   pip install -r requirements_production.txt")
        return False

def test_dependencies():
    """Test critical dependencies"""
    
    print()
    print("🔍 Testing Critical Dependencies:")
    print("-" * 40)
    
    critical_deps = [
        'streamlit',
        'pandas',
        'numpy',
        'pydantic',
        'requests',
        'openai'
    ]
    
    optional_deps = [
        'sentence_transformers',
        'transformers',
        'torch',
        'huggingface_hub'
    ]
    
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} missing (CRITICAL)")
    
    print()
    print("🔧 Testing Optional Dependencies:")
    print("-" * 40)
    
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"⚠️ {dep} missing (optional - some features may be limited)")

if __name__ == "__main__":
    try:
        success = test_imports()
        test_dependencies()
        
        print()
        print("=" * 80)
        
        if success:
            print("🎉 PRODUCTION SYSTEM READY!")
            exit(0)
        else:
            print("❌ PRODUCTION SYSTEM NOT READY")
            exit(1)
            
    except Exception as e:
        print(f"❌ Test script error: {e}")
        print()
        print("Stack trace:")
        traceback.print_exc()
        exit(1)

