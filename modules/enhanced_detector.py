"""
Enhanced Dialogue Detector with Multi-Mode Processing

Supports both traditional dialogue detection and new passage extraction
for handling diverse spiritual text formats.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import time
import psutil
import os

try:
    from .detector import DialogueDetector
    from .content_analyzer import ContentAnalyzer, PassageExtractor, SyntheticQAGenerator, ContentType
    from .scorer import ConsciousnessScorer
except ImportError:
    # Fallback for standalone usage
    import sys
    sys.path.append('..')
    from detector import DialogueDetector
    from content_analyzer import ContentAnalyzer, PassageExtractor, SyntheticQAGenerator, ContentType
    from scorer import ConsciousnessScorer


class EnhancedDialogueDetector(DialogueDetector):
    """Enhanced detector with lazy loading and multi-mode processing."""
    
    def __init__(self):
        """Initialize enhanced detector."""
        super().__init__()
        self._model = None
        self._model_loaded = False
        self.content_analyzer = ContentAnalyzer()
        self.passage_extractor = PassageExtractor()
        self.qa_generator = SyntheticQAGenerator()
        self.scorer = ConsciousnessScorer()
    
    @st.cache_resource
    def _load_model(_self):
        """Load sentence transformer model with caching."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except ImportError:
            st.warning("sentence-transformers not available. Semantic detection disabled.")
            return None
        except Exception as e:
            st.error(f"Failed to load semantic model: {e}")
            return None
    
    def get_model(self):
        """Get model with lazy loading."""
        if not self._model_loaded:
            self._model = self._load_model()
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
        Enhanced dialogue detection with multi-mode processing.
        
        Args:
            text: Input text to analyze
            mode: Detection mode ("auto", "regex", "semantic", "multi_mode")
            semantic_threshold: Threshold for semantic similarity
            show_progress: Whether to show progress indicators
            
        Returns:
            Detection results with dialogues and analysis
        """
        try:
            # 🛡️ DEFENSIVE: Ensure text is always a string (backup safety check)
            if isinstance(text, list):
                text = "\n\n".join(str(item) for item in text if item)
            elif not isinstance(text, str):
                text = str(text)
            
            progress_bar = None
            status_text = None
            
            if show_progress:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Analyzing content type...")
            
            # Step 1: Analyze content type
            content_analysis = self.content_analyzer.analyze_content(text)
            content_type = content_analysis['content_type']
            
            if show_progress:
                progress_bar.progress(20)
                status_text.text(f"Content type: {content_type.value}")
            
            # Step 2: Choose processing strategy
            processing_strategy = content_analysis['processing_strategy']
            all_dialogues = []
            
            # Step 3: Process based on content type
            if content_type in [ContentType.DIALOGUE_HEAVY, ContentType.MIXED]:
                # Traditional dialogue extraction
                if show_progress:
                    status_text.text("Extracting dialogues...")
                
                dialogues = self._extract_traditional_dialogues(
                    text, mode, semantic_threshold, show_progress, progress_bar, 20, 60
                )
                all_dialogues.extend(dialogues)
            
            if content_type in [ContentType.PROSE_HEAVY, ContentType.MIXED, ContentType.POETRY, ContentType.INSTRUCTIONAL]:
                # Passage extraction and synthetic Q&A generation
                if show_progress:
                    status_text.text("Extracting meaningful passages...")
                    progress_bar.progress(60)
                
                passages = self.passage_extractor.extract_passages(
                    text, content_type, min_consciousness_score=0.3
                )
                
                if show_progress:
                    status_text.text("Generating synthetic Q&A pairs...")
                    progress_bar.progress(80)
                
                # Generate synthetic Q&A from passages
                for passage in passages:
                    synthetic_qa = self.qa_generator.generate_qa_from_passage(passage)
                    
                    # Score each synthetic Q&A
                    for qa in synthetic_qa:
                        scoring_result = self.scorer.score_dialogue(qa['question'], qa['answer'])
                        qa.update(scoring_result)
                        qa['detection_method'] = 'passage_extraction'
                        qa['original_passage_score'] = passage.get('consciousness_score', 0.5)
                    
                    all_dialogues.extend(synthetic_qa)
            
            # Step 4: Final processing and scoring
            if show_progress:
                status_text.text("Finalizing results...")
                progress_bar.progress(90)
            
            # Remove duplicates and sort by score
            unique_dialogues = self._remove_duplicates(all_dialogues)
            unique_dialogues.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
            
            if show_progress:
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)  # Brief pause to show completion
                progress_bar.empty()
                status_text.empty()
            
            return {
                'success': True,
                'dialogues': unique_dialogues,
                'content_analysis': content_analysis,
                'processing_strategy': processing_strategy,
                'total_found': len(unique_dialogues),
                'traditional_dialogues': len([d for d in unique_dialogues if d.get('detection_method') != 'passage_extraction']),
                'synthetic_dialogues': len([d for d in unique_dialogues if d.get('detection_method') == 'passage_extraction']),
                'content_type': content_type.value
            }
            
        except Exception as e:
            if show_progress:
                st.error(f"Detection failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'dialogues': [],
                'content_analysis': None
            }
    
    def _extract_traditional_dialogues(
        self,
        text: str,
        mode: str,
        semantic_threshold: float,
        show_progress: bool,
        progress_bar,
        start_progress: int,
        end_progress: int
    ) -> List[Dict[str, Any]]:
        """Extract traditional Q&A dialogues."""
        dialogues = []
        
        # Regex detection
        if mode in ["regex", "auto"]:
            if show_progress:
                progress_bar.progress(start_progress + 10)
            
            regex_dialogues = self._detect_regex_dialogues(text)
            for dialogue in regex_dialogues:
                scoring_result = self.scorer.score_dialogue(dialogue['question'], dialogue['answer'])
                dialogue.update(scoring_result)
                dialogue['detection_method'] = 'regex'
            
            dialogues.extend(regex_dialogues)
        
        # Semantic detection
        if mode in ["semantic", "auto"]:
            if show_progress:
                progress_bar.progress(start_progress + 20)
            
            model = self.get_model()
            if model is not None:
                semantic_dialogues = self._detect_semantic_dialogues_enhanced(
                    text, semantic_threshold, model
                )
                for dialogue in semantic_dialogues:
                    scoring_result = self.scorer.score_dialogue(dialogue['question'], dialogue['answer'])
                    dialogue.update(scoring_result)
                    dialogue['detection_method'] = 'semantic'
                
                dialogues.extend(semantic_dialogues)
        
        if show_progress:
            progress_bar.progress(end_progress)
        
        return dialogues
    
    def _detect_semantic_dialogues_enhanced(
        self, 
        text: str, 
        threshold: float,
        model
    ) -> List[Dict[str, Any]]:
        """Enhanced semantic dialogue detection."""
        dialogues = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Consciousness-related query embeddings
        consciousness_queries = [
            "What is consciousness?",
            "Who am I?",
            "What is awareness?",
            "What is the nature of reality?",
            "How to find peace?",
            "What is enlightenment?",
            "What is the self?"
        ]
        
        try:
            query_embeddings = model.encode(consciousness_queries)
            
            # Look for question-answer pairs
            for i, para in enumerate(paragraphs):
                if len(para) > 50 and ('?' in para or any(q in para.lower() for q in ['what', 'how', 'why', 'who', 'when', 'where'])):
                    # This might be a question
                    if i + 1 < len(paragraphs):
                        next_para = paragraphs[i + 1]
                        if len(next_para) > 50:
                            # Check semantic similarity to consciousness topics
                            combined_text = f"{para} {next_para}"
                            text_embedding = model.encode([combined_text])
                            
                            # Calculate max similarity to consciousness queries
                            similarities = model.similarity(text_embedding, query_embeddings)
                            max_similarity = float(similarities.max())
                            
                            if max_similarity >= threshold:
                                dialogues.append({
                                    'question': para,
                                    'answer': next_para,
                                    'source': 'semantic',
                                    'confidence': max_similarity,
                                    'semantic_similarity': max_similarity
                                })
        
        except Exception as e:
            st.warning(f"Semantic detection error: {e}")
        
        return dialogues
    
    def _remove_duplicates(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate dialogues based on content similarity."""
        unique_dialogues = []
        seen_keys = set()
        
        for dialogue in dialogues:
            # Create a key based on first 100 characters of question and answer
            question_key = dialogue.get('question', '')[:100].lower().strip()
            answer_key = dialogue.get('answer', '')[:100].lower().strip()
            key = (question_key, answer_key)
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_dialogues.append(dialogue)
        
        return unique_dialogues
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'model_loaded': self._model_loaded,
                'model_available': self._model is not None
            }
        except Exception:
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'model_loaded': False,
                'model_available': False
            }
    
    def clear_model_cache(self):
        """Clear the model cache to free memory."""
        self._model = None
        self._model_loaded = False
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
    
    def analyze_text_only(self, text: str) -> Dict[str, Any]:
        """
        Analyze text without extracting dialogues.
        Useful for understanding content before processing.
        """
        return self.content_analyzer.analyze_content(text)

