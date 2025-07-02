"""
Enhanced Dialogue Detector with Lazy Model Loading

Optimized for memory efficiency and performance.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import re
import time

try:
    from .detector import DialogueDetector
    from .scorer import ConsciousnessScorer, DialogueMode
except ImportError:
    # Fallback for standalone usage
    import sys
    sys.path.append('..')
    from detector import DialogueDetector
    from scorer import ConsciousnessScorer, DialogueMode


@st.cache_resource
def load_sentence_transformer():
    """
    Load sentence transformer model with caching.
    
    Returns:
        SentenceTransformer model
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        with st.spinner("Loading AI model for semantic detection..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
        
        st.success("✅ AI model loaded successfully!")
        return model
        
    except ImportError:
        st.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


class EnhancedDialogueDetector(DialogueDetector):
    """Enhanced dialogue detector with lazy loading and performance optimizations."""
    
    def __init__(self):
        """Initialize enhanced detector without loading model."""
        super().__init__()
        self._model = None
        self._model_loaded = False
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if not self._model_loaded:
            self._model = load_sentence_transformer()
            self._model_loaded = True
        return self._model
    
    def detect_dialogues_with_progress(
        self,
        text: str,
        mode: str = "auto",
        semantic_threshold: float = 0.4,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Detect dialogues with progress tracking.
        
        Args:
            text: Input text to analyze
            mode: Detection mode ("auto", "regex", "semantic")
            semantic_threshold: Threshold for semantic similarity
            show_progress: Whether to show progress in Streamlit
            
        Returns:
            Dict with detection results
        """
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Step 1: Text preprocessing
            if show_progress:
                progress_bar.progress(10)
                status_text.text("Preprocessing text...")
            
            lines = text.split('\n')
            total_lines = len(lines)
            
            # Step 2: Regex detection
            if show_progress:
                progress_bar.progress(30)
                status_text.text("Detecting structured dialogues...")
            
            regex_dialogues = []
            if mode in ["auto", "regex"]:
                regex_dialogues = self._detect_regex_dialogues(lines)
            
            # Step 3: Semantic detection
            semantic_dialogues = []
            if mode in ["auto", "semantic"]:
                if show_progress:
                    progress_bar.progress(50)
                    status_text.text("Loading AI model for semantic detection...")
                
                # Lazy load model only when needed
                if self.model is not None:
                    if show_progress:
                        progress_bar.progress(70)
                        status_text.text("Performing semantic analysis...")
                    
                    semantic_dialogues = self._detect_semantic_dialogues_with_progress(
                        lines, semantic_threshold, progress_bar, status_text
                    )
            
            # Step 4: Combine and deduplicate
            if show_progress:
                progress_bar.progress(90)
                status_text.text("Combining results...")
            
            all_dialogues = self._combine_dialogues(regex_dialogues, semantic_dialogues)
            
            # Step 5: Score dialogues
            if show_progress:
                progress_bar.progress(95)
                status_text.text("Scoring consciousness recognition...")
            
            scorer = ConsciousnessScorer()
            scored_dialogues = []
            
            for dialogue in all_dialogues:
                score_result = scorer.score_dialogue(dialogue['question'], dialogue['answer'])
                dialogue.update(score_result)
                scored_dialogues.append(dialogue)
            
            if show_progress:
                progress_bar.progress(100)
                status_text.text("✅ Detection completed!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
            
            return {
                'success': True,
                'dialogues': scored_dialogues,
                'total_found': len(scored_dialogues),
                'regex_count': len(regex_dialogues),
                'semantic_count': len(semantic_dialogues),
                'processing_stats': {
                    'total_lines': total_lines,
                    'mode': mode,
                    'semantic_threshold': semantic_threshold
                }
            }
            
        except Exception as e:
            if show_progress:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Detection failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'dialogues': [],
                'total_found': 0
            }
    
    def _detect_semantic_dialogues_with_progress(
        self,
        lines: List[str],
        threshold: float,
        progress_bar,
        status_text
    ) -> List[Dict[str, Any]]:
        """
        Detect semantic dialogues with progress updates.
        
        Args:
            lines: Text lines to analyze
            threshold: Semantic similarity threshold
            progress_bar: Streamlit progress bar
            status_text: Streamlit status text
            
        Returns:
            List of detected dialogues
        """
        if self.model is None:
            return []
        
        # Non-dual anchor phrases for semantic comparison
        anchor_phrases = [
            "You are awareness itself",
            "Remain as you are",
            "There is no seeker, only seeking",
            "Consciousness is your true nature",
            "Be still and know",
            "What you are looking for is what is looking"
        ]
        
        # Get anchor embeddings
        anchor_embeddings = self.model.encode(anchor_phrases)
        
        dialogues = []
        total_lines = len(lines)
        
        # Process lines in batches to show progress
        batch_size = max(1, total_lines // 20)  # 20 progress updates
        
        for i in range(0, total_lines, batch_size):
            batch_end = min(i + batch_size, total_lines)
            batch_lines = lines[i:batch_end]
            
            # Update progress
            progress = 70 + (i / total_lines) * 20  # 70-90% range
            progress_bar.progress(progress / 100)
            status_text.text(f"Analyzing lines {i+1}-{batch_end} of {total_lines}...")
            
            # Process batch
            for line_idx, line in enumerate(batch_lines):
                actual_idx = i + line_idx
                
                if len(line.strip()) > 50:  # Only analyze substantial lines
                    try:
                        line_embedding = self.model.encode([line])
                        similarities = self.model.similarity(line_embedding, anchor_embeddings)
                        max_similarity = float(similarities.max())
                        
                        if max_similarity > threshold:
                            # Create dialogue from semantic match
                            context_start = max(0, actual_idx - 2)
                            context_end = min(len(lines), actual_idx + 3)
                            context = '\n'.join(lines[context_start:context_end])
                            
                            dialogues.append({
                                'question': f"Context from line {actual_idx + 1}",
                                'answer': line.strip(),
                                'source': 'semantic',
                                'line_number': actual_idx + 1,
                                'similarity_score': max_similarity,
                                'context': context
                            })
                    except Exception as e:
                        # Skip problematic lines
                        continue
            
            # Small delay to prevent UI freezing
            time.sleep(0.001)
        
        return dialogues
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Returns:
            Dict with memory usage stats
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'model_loaded': self._model_loaded,
            'model_memory_estimate': 500 if self._model_loaded else 0  # Approximate MB
        }
    
    def clear_model_cache(self):
        """Clear the loaded model to free memory."""
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
        self._model = None
        self._model_loaded = False
        st.success("🧹 Model cache cleared!")


def create_enhanced_detector() -> EnhancedDialogueDetector:
    """
    Factory function to create enhanced detector.
    
    Returns:
        EnhancedDialogueDetector instance
    """
    return EnhancedDialogueDetector()

