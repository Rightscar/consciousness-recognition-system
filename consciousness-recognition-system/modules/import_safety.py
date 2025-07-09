"""
Import Safety Module
Comprehensive error prevention system for conditional imports and type hints
"""

import sys
import logging
from typing import Any, Dict, Optional, Type, Union
from functools import wraps

logger = logging.getLogger(__name__)

class ImportSafetyManager:
    """Manages safe imports and provides fallback classes for type hints"""
    
    def __init__(self):
        self.available_modules = {}
        self.fallback_classes = {}
        self._initialize_fallbacks()
    
    def _initialize_fallbacks(self):
        """Initialize fallback classes for common conditional imports"""
        
        # PIL/Pillow fallbacks
        class DummyImage:
            class Image:
                pass
        
        # spaCy fallbacks
        class DummyDoc:
            pass
        
        class DummySpan:
            pass
        
        class DummyToken:
            pass
        
        # NLTK fallbacks
        class DummyTree:
            pass
        
        # SentenceTransformers fallbacks
        class DummySentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass
            
            def encode(self, *args, **kwargs):
                return []
        
        # Store fallbacks
        self.fallback_classes.update({
            'PIL.Image': DummyImage,
            'spacy.Doc': DummyDoc,
            'spacy.Span': DummySpan,
            'spacy.Token': DummyToken,
            'nltk.Tree': DummyTree,
            'sentence_transformers.SentenceTransformer': DummySentenceTransformer,
        })
    
    def safe_import(self, module_name: str, class_name: Optional[str] = None, 
                   fallback_key: Optional[str] = None) -> tuple[bool, Any]:
        """
        Safely import a module or class with fallback support
        
        Args:
            module_name: Name of the module to import
            class_name: Specific class to import from module
            fallback_key: Key for fallback class if import fails
            
        Returns:
            Tuple of (success: bool, imported_object: Any)
        """
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            
            if class_name:
                imported_object = getattr(module, class_name)
            else:
                imported_object = module
            
            self.available_modules[f"{module_name}.{class_name}" if class_name else module_name] = True
            logger.info(f"✅ Successfully imported {module_name}.{class_name if class_name else ''}")
            
            return True, imported_object
            
        except ImportError as e:
            logger.warning(f"⚠️ Failed to import {module_name}.{class_name if class_name else ''}: {e}")
            
            # Use fallback if available
            if fallback_key and fallback_key in self.fallback_classes:
                fallback = self.fallback_classes[fallback_key]
                if class_name:
                    fallback = getattr(fallback, class_name, fallback)
                
                self.available_modules[f"{module_name}.{class_name}" if class_name else module_name] = False
                logger.info(f"🔄 Using fallback for {module_name}.{class_name if class_name else ''}")
                
                return False, fallback
            
            return False, None
    
    def is_available(self, module_key: str) -> bool:
        """Check if a module/class is available"""
        return self.available_modules.get(module_key, False)
    
    def get_import_status(self) -> Dict[str, bool]:
        """Get status of all attempted imports"""
        return self.available_modules.copy()
    
    def create_safe_type_hint(self, type_name: str, fallback_type: Type = Any) -> Type:
        """Create a safe type hint that works even if the actual type is not available"""
        if type_name in self.available_modules and self.available_modules[type_name]:
            # Try to get the actual type
            try:
                parts = type_name.split('.')
                module_name = '.'.join(parts[:-1])
                class_name = parts[-1]
                
                module = sys.modules.get(module_name)
                if module:
                    return getattr(module, class_name, fallback_type)
            except (AttributeError, KeyError):
                pass
        
        return fallback_type

# Global instance
import_safety = ImportSafetyManager()

def safe_import_decorator(module_name: str, class_name: Optional[str] = None, 
                         fallback_key: Optional[str] = None):
    """Decorator for safe imports in functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            success, imported_object = import_safety.safe_import(
                module_name, class_name, fallback_key
            )
            
            # Add import info to kwargs
            kwargs['_import_success'] = success
            kwargs['_imported_object'] = imported_object
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_dependencies() -> Dict[str, Dict[str, bool]]:
    """Check all critical dependencies for the application"""
    
    dependencies = {
        'core': {
            'streamlit': False,
            'openai': False,
            'pandas': False,
            'numpy': False,
        },
        'nlp': {
            'spacy': False,
            'nltk': False,
            'sentence_transformers': False,
            'textstat': False,
        },
        'ocr': {
            'pytesseract': False,
            'PIL': False,
            'pdf2image': False,
            'PyPDF2': False,
        },
        'ml': {
            'scikit-learn': False,
            'transformers': False,
        }
    }
    
    # Check each dependency
    for category, modules in dependencies.items():
        for module_name in modules:
            try:
                __import__(module_name)
                dependencies[category][module_name] = True
            except ImportError:
                dependencies[category][module_name] = False
    
    return dependencies

def get_fallback_recommendations() -> Dict[str, str]:
    """Get recommendations for missing dependencies"""
    
    recommendations = {
        'spacy': "Install with: pip install spacy && python -m spacy download en_core_web_sm",
        'nltk': "Install with: pip install nltk && python -c \"import nltk; nltk.download('punkt')\"",
        'sentence_transformers': "Install with: pip install sentence-transformers",
        'pytesseract': "Install with: pip install pytesseract (requires Tesseract OCR)",
        'PIL': "Install with: pip install Pillow",
        'pdf2image': "Install with: pip install pdf2image",
        'textstat': "Install with: pip install textstat",
        'scikit-learn': "Install with: pip install scikit-learn",
    }
    
    return recommendations

def create_import_report() -> str:
    """Create a comprehensive import status report"""
    
    dependencies = check_dependencies()
    recommendations = get_fallback_recommendations()
    
    report = ["# Import Status Report\n"]
    
    for category, modules in dependencies.items():
        report.append(f"## {category.title()} Dependencies\n")
        
        for module_name, available in modules.items():
            status = "✅ Available" if available else "❌ Missing"
            report.append(f"- **{module_name}**: {status}")
            
            if not available and module_name in recommendations:
                report.append(f"  - *Fix*: {recommendations[module_name]}")
        
        report.append("")
    
    # Add import safety status
    report.append("## Import Safety Status\n")
    import_status = import_safety.get_import_status()
    
    if import_status:
        for module_key, available in import_status.items():
            status = "✅ Available" if available else "🔄 Using Fallback"
            report.append(f"- **{module_key}**: {status}")
    else:
        report.append("- No imports attempted yet")
    
    return "\n".join(report)

# Pre-initialize common imports with fallbacks
def initialize_common_imports():
    """Initialize common imports that might cause NameErrors"""
    
    # PIL Image
    import_safety.safe_import('PIL', 'Image', 'PIL.Image')
    
    # spaCy types
    import_safety.safe_import('spacy.tokens', 'Doc', 'spacy.Doc')
    import_safety.safe_import('spacy.tokens', 'Span', 'spacy.Span')
    import_safety.safe_import('spacy.tokens', 'Token', 'spacy.Token')
    
    # SentenceTransformers
    import_safety.safe_import('sentence_transformers', 'SentenceTransformer', 'sentence_transformers.SentenceTransformer')
    
    logger.info("🔧 Common imports initialized with fallbacks")

# Initialize on module load
initialize_common_imports()

