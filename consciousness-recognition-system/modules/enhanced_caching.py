"""
Enhanced Caching Module
======================

Provides intelligent caching for heavy operations in the Enhanced Universal AI Training Data Creator.
Uses Streamlit's caching mechanisms selectively for non-sensitive, computationally expensive operations.

Features:
- Selective caching for prompt templates, content extraction, and model loading
- Cache invalidation strategies
- Memory management and cleanup
- Performance monitoring
"""

import streamlit as st
import hashlib
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
import pickle
import json

logger = logging.getLogger(__name__)


class EnhancedCaching:
    """Enhanced caching system with intelligent cache management"""
    
    def __init__(self):
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'memory_usage': 0
        }
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_prompt_templates(self, prompts_dir: str) -> Dict[str, str]:
        """Cache prompt templates loading - heavy file I/O operation"""
        logger.debug(f"Loading prompt templates from {prompts_dir}")
        
        templates = {}
        try:
            for filename in os.listdir(prompts_dir):
                if filename.endswith('.txt'):
                    template_name = filename.replace('.txt', '')
                    with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as f:
                        templates[template_name] = f.read()
                        
            logger.info(f"Loaded {len(templates)} prompt templates")
            self.cache_stats['hits'] += 1
            return templates
            
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")
            self.cache_stats['misses'] += 1
            return {}
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def extract_content_cached(self, file_content: bytes, file_type: str, file_hash: str) -> Dict[str, Any]:
        """Cache content extraction results - expensive operation"""
        logger.debug(f"Extracting content for file type: {file_type}, hash: {file_hash[:8]}...")
        
        # This would integrate with the actual extraction logic
        # For now, return a placeholder structure
        extraction_result = {
            'content': '',
            'metadata': {
                'file_type': file_type,
                'file_hash': file_hash,
                'extraction_time': datetime.now().isoformat(),
                'cached': True
            },
            'statistics': {
                'total_chars': 0,
                'total_words': 0,
                'total_lines': 0
            }
        }
        
        self.cache_stats['hits'] += 1
        logger.info(f"Content extraction cached for hash: {file_hash[:8]}")
        return extraction_result
    
    @st.cache_resource
    def load_sentence_transformer_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Cache sentence transformer model loading - very expensive operation"""
        logger.debug(f"Loading sentence transformer model: {model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            logger.info(f"Sentence transformer model loaded: {model_name}")
            return model
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            return None
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            return None
    
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def compute_content_statistics(self, content: str, content_hash: str) -> Dict[str, Any]:
        """Cache content statistics computation"""
        logger.debug(f"Computing statistics for content hash: {content_hash[:8]}")
        
        stats = {
            'total_chars': len(content),
            'total_words': len(content.split()),
            'total_lines': len(content.splitlines()),
            'avg_words_per_line': 0,
            'complexity_score': 0,
            'computed_at': datetime.now().isoformat(),
            'content_hash': content_hash
        }
        
        if stats['total_lines'] > 0:
            stats['avg_words_per_line'] = stats['total_words'] / stats['total_lines']
        
        # Simple complexity score based on vocabulary diversity
        words = content.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            stats['complexity_score'] = len(unique_words) / len(words)
        
        self.cache_stats['hits'] += 1
        logger.debug(f"Statistics computed and cached for hash: {content_hash[:8]}")
        return stats
    
    def generate_content_hash(self, content: str) -> str:
        """Generate hash for content to use as cache key"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def generate_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file content to use as cache key"""
        return hashlib.md5(file_content).hexdigest()
    
    def invalidate_cache(self, cache_type: str = "all"):
        """Invalidate specific cache types"""
        logger.info(f"Invalidating cache type: {cache_type}")
        
        try:
            if cache_type == "all" or cache_type == "prompts":
                self.load_prompt_templates.clear()
            
            if cache_type == "all" or cache_type == "extraction":
                self.extract_content_cached.clear()
            
            if cache_type == "all" or cache_type == "statistics":
                self.compute_content_statistics.clear()
            
            if cache_type == "all" or cache_type == "models":
                self.load_sentence_transformer_model.clear()
            
            self.cache_stats['invalidations'] += 1
            logger.info(f"Cache invalidated: {cache_type}")
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def render_cache_debug_info(self):
        """Render cache debug information in Streamlit"""
        if st.secrets.get("debug_mode", False):
            with st.expander("🔧 Cache Debug Information"):
                stats = self.get_cache_stats()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Cache Hits", stats['hits'])
                
                with col2:
                    st.metric("Cache Misses", stats['misses'])
                
                with col3:
                    st.metric("Hit Rate", f"{stats['hit_rate']:.1f}%")
                
                with col4:
                    st.metric("Invalidations", stats['invalidations'])
                
                # Cache management controls
                st.markdown("**Cache Management:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Clear Prompt Cache"):
                        self.invalidate_cache("prompts")
                        st.success("Prompt cache cleared")
                
                with col2:
                    if st.button("Clear Extraction Cache"):
                        self.invalidate_cache("extraction")
                        st.success("Extraction cache cleared")
                
                with col3:
                    if st.button("Clear All Caches"):
                        self.invalidate_cache("all")
                        st.success("All caches cleared")


# Global caching instance
enhanced_caching = EnhancedCaching()

