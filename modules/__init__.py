"""
Universal AI Training Data Creator - Modules Package
==================================================

This package contains all the integrated modules for the Universal AI Training Data Creator:

- enhanced_pdf_extractor: 4-engine PDF extraction with confidence scoring
- universal_extractor: Multi-format content extraction (PDF, TXT, DOCX, EPUB)
- custom_prompt_engine: GPT enhancement with 6 spiritual tones
- manual_edit_features: Inline editing and quality management
- preview_testing_features: JSONL preview and testing capabilities
- text_validator_enhanced: Comprehensive text validation and normalization

All modules are designed to work together seamlessly in the integrated application.
"""

__version__ = "1.0.0"
__author__ = "Universal AI Training Data Creator"

# Import key classes for easy access
try:
    from .enhanced_pdf_extractor import EnhancedPDFExtractor
    from .universal_extractor import UniversalExtractor
    from .custom_prompt_engine import CustomPromptEngine
    from .manual_edit_features import ManualEditManager
    from .preview_testing_features import PreviewTestingManager
    from .text_validator_enhanced import EnhancedTextValidator
    
    __all__ = [
        'EnhancedPDFExtractor',
        'UniversalExtractor', 
        'CustomPromptEngine',
        'ManualEditManager',
        'PreviewTestingManager',
        'EnhancedTextValidator'
    ]
    
except ImportError as e:
    # Graceful handling if some modules are missing
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []

