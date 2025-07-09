"""
Advanced NLP Features Module
Additional NLP enhancements and alternatives for the Fine-Tune Data System
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import streamlit as st

# Core NLP libraries
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize, paragraph_tokenize
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Advanced NLP libraries
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

@dataclass
class ContentInsight:
    """Represents an insight extracted from content"""
    insight_type: str
    content: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_location: str = ""
    related_themes: List[str] = field(default_factory=list)

@dataclass
class NLPAnalysisResult:
    """Complete NLP analysis result"""
    readability_scores: Dict[str, float]
    sentiment_analysis: Dict[str, float]
    key_entities: List[Dict[str, Any]]
    topic_clusters: List[Dict[str, Any]]
    content_insights: List[ContentInsight]
    linguistic_features: Dict[str, Any]
    quality_metrics: Dict[str, float]

class AdvancedNLPProcessor:
    """Advanced NLP processing with multiple techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentence_model = None
        self.nlp = None
        self.lemmatizer = None
        self.stemmer = None
        self.sentiment_analyzer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models and tools"""
        try:
            # Initialize NLTK components
            if NLTK_AVAILABLE:
                self._download_nltk_data()
                self.lemmatizer = WordNetLemmatizer()
                self.stemmer = PorterStemmer()
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize sentence transformer
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.logger.info("Loaded sentence transformer model")
                except Exception as e:
                    self.logger.warning(f"Failed to load sentence transformer: {e}")
            
            # Initialize spaCy if available
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.logger.info("Loaded spaCy model")
                except OSError:
                    self.logger.warning("spaCy model not available")
                    
        except Exception as e:
            self.logger.error(f"Model initialization error: {e}")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'vader_lexicon'
        ]
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                except:
                    pass
    
    def analyze_content_comprehensively(self, content: str) -> NLPAnalysisResult:
        """Perform comprehensive NLP analysis"""
        try:
            # Readability analysis
            readability_scores = self._analyze_readability(content)
            
            # Sentiment analysis
            sentiment_analysis = self._analyze_sentiment(content)
            
            # Entity extraction
            key_entities = self._extract_entities(content)
            
            # Topic clustering
            topic_clusters = self._perform_topic_clustering(content)
            
            # Content insights
            content_insights = self._extract_content_insights(content)
            
            # Linguistic features
            linguistic_features = self._analyze_linguistic_features(content)
            
            # Quality metrics
            quality_metrics = self._calculate_quality_metrics(content)
            
            return NLPAnalysisResult(
                readability_scores=readability_scores,
                sentiment_analysis=sentiment_analysis,
                key_entities=key_entities,
                topic_clusters=topic_clusters,
                content_insights=content_insights,
                linguistic_features=linguistic_features,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis error: {e}")
            return self._create_empty_result()
    
    def _analyze_readability(self, content: str) -> Dict[str, float]:
        """Analyze text readability"""
        scores = {}
        
        try:
            if TEXTSTAT_AVAILABLE:
                scores['flesch_reading_ease'] = flesch_reading_ease(content)
                scores['flesch_kincaid_grade'] = flesch_kincaid_grade(content)
                scores['automated_readability_index'] = automated_readability_index(content)
            
            # Basic readability metrics
            sentences = sent_tokenize(content) if NLTK_AVAILABLE else content.split('.')
            words = word_tokenize(content) if NLTK_AVAILABLE else content.split()
            
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            scores['avg_sentence_length'] = avg_sentence_length
            scores['avg_word_length'] = avg_word_length
            scores['total_sentences'] = len(sentences)
            scores['total_words'] = len(words)
            
        except Exception as e:
            self.logger.error(f"Readability analysis error: {e}")
        
        return scores
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of content"""
        sentiment = {}
        
        try:
            if NLTK_AVAILABLE and self.sentiment_analyzer:
                # Overall sentiment
                scores = self.sentiment_analyzer.polarity_scores(content)
                sentiment.update(scores)
                
                # Sentence-level sentiment analysis
                sentences = sent_tokenize(content)
                sentence_sentiments = [
                    self.sentiment_analyzer.polarity_scores(sent)['compound']
                    for sent in sentences
                ]
                
                sentiment['sentence_sentiment_variance'] = np.var(sentence_sentiments) if sentence_sentiments else 0
                sentiment['positive_sentences'] = sum(1 for s in sentence_sentiments if s > 0.1)
                sentiment['negative_sentences'] = sum(1 for s in sentence_sentiments if s < -0.1)
                sentiment['neutral_sentences'] = sum(1 for s in sentence_sentiments if -0.1 <= s <= 0.1)
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
        
        return sentiment
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities and key concepts"""
        entities = []
        
        try:
            # spaCy entity extraction
            if SPACY_AVAILABLE and self.nlp:
                doc = self.nlp(content)
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'description': spacy.explain(ent.label_),
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8,  # spaCy doesn't provide confidence scores
                        'method': 'spacy'
                    })
            
            # NLTK entity extraction (fallback)
            elif NLTK_AVAILABLE:
                sentences = sent_tokenize(content)
                for sent in sentences[:5]:  # Limit for performance
                    tokens = word_tokenize(sent)
                    pos_tags = pos_tag(tokens)
                    chunks = ne_chunk(pos_tags)
                    
                    for chunk in chunks:
                        if hasattr(chunk, 'label'):
                            entity_text = ' '.join([token for token, pos in chunk.leaves()])
                            entities.append({
                                'text': entity_text,
                                'label': chunk.label(),
                                'description': chunk.label(),
                                'start': content.find(entity_text),
                                'end': content.find(entity_text) + len(entity_text),
                                'confidence': 0.6,
                                'method': 'nltk'
                            })
            
            # Remove duplicates and sort by confidence
            unique_entities = []
            seen_texts = set()
            
            for entity in sorted(entities, key=lambda x: x['confidence'], reverse=True):
                if entity['text'].lower() not in seen_texts:
                    unique_entities.append(entity)
                    seen_texts.add(entity['text'].lower())
            
            return unique_entities[:20]  # Return top 20 entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction error: {e}")
            return []
    
    def _perform_topic_clustering(self, content: str) -> List[Dict[str, Any]]:
        """Perform topic clustering on content"""
        clusters = []
        
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.sentence_model:
                return clusters
            
            # Split content into sentences
            sentences = sent_tokenize(content) if NLTK_AVAILABLE else content.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) < 3:
                return clusters
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(sentences)
            
            # Perform clustering
            n_clusters = min(5, len(sentences) // 3)  # Adaptive cluster count
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Organize clusters
            for cluster_id in range(n_clusters):
                cluster_sentences = [
                    sentences[i] for i, label in enumerate(cluster_labels) 
                    if label == cluster_id
                ]
                
                if cluster_sentences:
                    # Find representative sentence (closest to centroid)
                    cluster_embeddings = [
                        embeddings[i] for i, label in enumerate(cluster_labels)
                        if label == cluster_id
                    ]
                    
                    centroid = np.mean(cluster_embeddings, axis=0)
                    distances = [
                        cosine_similarity([emb], [centroid])[0][0]
                        for emb in cluster_embeddings
                    ]
                    
                    representative_idx = np.argmax(distances)
                    representative_sentence = cluster_sentences[representative_idx]
                    
                    # Generate topic keywords
                    topic_keywords = self._extract_topic_keywords(cluster_sentences)
                    
                    clusters.append({
                        'cluster_id': cluster_id,
                        'size': len(cluster_sentences),
                        'representative_sentence': representative_sentence,
                        'keywords': topic_keywords,
                        'sentences': cluster_sentences[:3],  # First 3 sentences
                        'coherence_score': float(np.mean(distances))
                    })
            
            return sorted(clusters, key=lambda x: x['size'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Topic clustering error: {e}")
            return []
    
    def _extract_topic_keywords(self, sentences: List[str]) -> List[str]:
        """Extract keywords from a cluster of sentences"""
        try:
            if not NLTK_AVAILABLE:
                return []
            
            # Combine sentences
            text = ' '.join(sentences)
            
            # Tokenize and filter
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            
            # Filter words
            filtered_words = [
                word for word in words
                if word.isalpha() and len(word) > 3 and word not in stop_words
            ]
            
            # Get most common words
            word_freq = Counter(filtered_words)
            keywords = [word for word, freq in word_freq.most_common(5)]
            
            return keywords
            
        except Exception as e:
            self.logger.error(f"Keyword extraction error: {e}")
            return []
    
    def _extract_content_insights(self, content: str) -> List[ContentInsight]:
        """Extract meaningful insights from content"""
        insights = []
        
        try:
            # Question detection
            questions = self._detect_questions(content)
            for question in questions:
                insights.append(ContentInsight(
                    insight_type="question",
                    content=question,
                    confidence=0.9,
                    metadata={'type': 'interrogative'},
                    source_location="content_analysis"
                ))
            
            # Quote detection
            quotes = self._detect_quotes(content)
            for quote in quotes:
                insights.append(ContentInsight(
                    insight_type="quote",
                    content=quote,
                    confidence=0.8,
                    metadata={'type': 'quotation'},
                    source_location="content_analysis"
                ))
            
            # Key concept detection
            key_concepts = self._detect_key_concepts(content)
            for concept in key_concepts:
                insights.append(ContentInsight(
                    insight_type="key_concept",
                    content=concept,
                    confidence=0.7,
                    metadata={'type': 'concept'},
                    source_location="content_analysis"
                ))
            
            # Dialogue detection
            dialogues = self._detect_dialogues(content)
            for dialogue in dialogues:
                insights.append(ContentInsight(
                    insight_type="dialogue",
                    content=dialogue,
                    confidence=0.8,
                    metadata={'type': 'conversation'},
                    source_location="content_analysis"
                ))
            
            return insights[:15]  # Return top 15 insights
            
        except Exception as e:
            self.logger.error(f"Content insights error: {e}")
            return []
    
    def _detect_questions(self, content: str) -> List[str]:
        """Detect questions in content"""
        questions = []
        
        # Simple regex for questions
        question_pattern = r'[.!?]\s*([A-Z][^.!?]*\?)'
        matches = re.findall(question_pattern, content)
        questions.extend(matches)
        
        # Questions at start of sentences
        sentences = sent_tokenize(content) if NLTK_AVAILABLE else content.split('.')
        for sentence in sentences:
            if sentence.strip().endswith('?'):
                questions.append(sentence.strip())
        
        return list(set(questions))[:5]  # Return unique questions
    
    def _detect_quotes(self, content: str) -> List[str]:
        """Detect quoted text"""
        quotes = []
        
        # Detect quoted text
        quote_patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'"([^"]+)"',
            r"'([^']+)'",
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, content)
            quotes.extend([match for match in matches if len(match) > 10])
        
        return list(set(quotes))[:5]
    
    def _detect_key_concepts(self, content: str) -> List[str]:
        """Detect key concepts and important phrases"""
        concepts = []
        
        try:
            if NLTK_AVAILABLE:
                # Extract noun phrases
                sentences = sent_tokenize(content)
                for sentence in sentences[:10]:  # Limit for performance
                    tokens = word_tokenize(sentence)
                    pos_tags = pos_tag(tokens)
                    
                    # Find noun phrases
                    noun_phrase = []
                    for word, pos in pos_tags:
                        if pos.startswith('N'):  # Noun
                            noun_phrase.append(word)
                        elif pos.startswith('J') and noun_phrase:  # Adjective
                            noun_phrase.append(word)
                        else:
                            if len(noun_phrase) > 1:
                                concepts.append(' '.join(noun_phrase))
                            noun_phrase = []
                    
                    if len(noun_phrase) > 1:
                        concepts.append(' '.join(noun_phrase))
            
            # Filter and deduplicate
            filtered_concepts = [
                concept for concept in set(concepts)
                if len(concept) > 5 and len(concept.split()) <= 4
            ]
            
            return filtered_concepts[:10]
            
        except Exception as e:
            self.logger.error(f"Key concept detection error: {e}")
            return []
    
    def _detect_dialogues(self, content: str) -> List[str]:
        """Detect dialogue patterns"""
        dialogues = []
        
        # Detect dialogue patterns
        dialogue_patterns = [
            r'([A-Z][a-z]+):\s*"([^"]+)"',  # Name: "speech"
            r'([A-Z][a-z]+)\s+said[,:]?\s*"([^"]+)"',  # Name said "speech"
            r'"([^"]+)"\s*[,.]?\s*([A-Z][a-z]+)\s+(?:said|asked|replied)',  # "speech" Name said
        ]
        
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    dialogue = f"{match[0]}: {match[1]}"
                    dialogues.append(dialogue)
        
        return list(set(dialogues))[:5]
    
    def _analyze_linguistic_features(self, content: str) -> Dict[str, Any]:
        """Analyze linguistic features"""
        features = {}
        
        try:
            if NLTK_AVAILABLE:
                words = word_tokenize(content)
                sentences = sent_tokenize(content)
                
                # POS tag distribution
                pos_tags = pos_tag(words)
                pos_counts = Counter([pos for word, pos in pos_tags])
                
                features['pos_distribution'] = dict(pos_counts.most_common(10))
                features['lexical_diversity'] = len(set(words)) / len(words) if words else 0
                features['sentence_complexity'] = sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
                
                # Vocabulary sophistication
                long_words = [word for word in words if len(word) > 6]
                features['long_word_ratio'] = len(long_words) / len(words) if words else 0
                
                # Punctuation analysis
                punctuation_count = sum(1 for char in content if char in '.,!?;:')
                features['punctuation_density'] = punctuation_count / len(content) if content else 0
            
        except Exception as e:
            self.logger.error(f"Linguistic analysis error: {e}")
        
        return features
    
    def _calculate_quality_metrics(self, content: str) -> Dict[str, float]:
        """Calculate content quality metrics"""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['character_count'] = len(content)
            metrics['word_count'] = len(content.split())
            metrics['sentence_count'] = len(sent_tokenize(content)) if NLTK_AVAILABLE else content.count('.')
            
            # Coherence metrics
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.sentence_model:
                sentences = sent_tokenize(content) if NLTK_AVAILABLE else content.split('.')
                if len(sentences) > 1:
                    embeddings = self.sentence_model.encode(sentences)
                    
                    # Calculate average similarity between consecutive sentences
                    similarities = []
                    for i in range(len(embeddings) - 1):
                        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                        similarities.append(sim)
                    
                    metrics['coherence_score'] = float(np.mean(similarities)) if similarities else 0
                    metrics['coherence_variance'] = float(np.var(similarities)) if similarities else 0
            
            # Information density
            unique_words = len(set(content.lower().split()))
            total_words = len(content.split())
            metrics['information_density'] = unique_words / total_words if total_words else 0
            
            # Structure metrics
            paragraphs = content.split('\n\n')
            metrics['paragraph_count'] = len([p for p in paragraphs if p.strip()])
            metrics['avg_paragraph_length'] = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
            
        except Exception as e:
            self.logger.error(f"Quality metrics error: {e}")
        
        return metrics
    
    def _create_empty_result(self) -> NLPAnalysisResult:
        """Create empty analysis result"""
        return NLPAnalysisResult(
            readability_scores={},
            sentiment_analysis={},
            key_entities=[],
            topic_clusters=[],
            content_insights=[],
            linguistic_features={},
            quality_metrics={}
        )
    
    def generate_content_suggestions(self, content: str, target_type: str = "question_answer") -> List[Dict[str, Any]]:
        """Generate suggestions for content improvement"""
        suggestions = []
        
        try:
            analysis = self.analyze_content_comprehensively(content)
            
            # Readability suggestions
            if analysis.readability_scores.get('flesch_reading_ease', 50) < 30:
                suggestions.append({
                    'type': 'readability',
                    'suggestion': 'Consider simplifying sentence structure for better readability',
                    'priority': 'medium',
                    'details': 'Current text may be difficult to read'
                })
            
            # Content structure suggestions
            if len(analysis.content_insights) < 3:
                suggestions.append({
                    'type': 'content_structure',
                    'suggestion': 'Add more questions or key concepts to improve training value',
                    'priority': 'high',
                    'details': 'More structured content elements would enhance AI training'
                })
            
            # Sentiment balance suggestions
            sentiment = analysis.sentiment_analysis
            if sentiment.get('compound', 0) < -0.5:
                suggestions.append({
                    'type': 'sentiment',
                    'suggestion': 'Consider balancing negative sentiment with constructive elements',
                    'priority': 'low',
                    'details': 'Very negative content may need balance'
                })
            
            # Topic diversity suggestions
            if len(analysis.topic_clusters) < 2:
                suggestions.append({
                    'type': 'topic_diversity',
                    'suggestion': 'Content appears to focus on a single topic - consider adding variety',
                    'priority': 'medium',
                    'details': 'More diverse topics could improve training data quality'
                })
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Content suggestions error: {e}")
            return []

# Streamlit UI Components for Advanced NLP Features
def create_nlp_analysis_ui():
    """Create UI for advanced NLP analysis"""
    st.subheader("🧠 Advanced NLP Analysis")
    
    # Analysis options
    analysis_options = st.multiselect(
        "Select analysis types:",
        [
            "Readability Analysis",
            "Sentiment Analysis", 
            "Entity Extraction",
            "Topic Clustering",
            "Content Insights",
            "Linguistic Features",
            "Quality Metrics"
        ],
        default=["Readability Analysis", "Content Insights", "Quality Metrics"]
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            max_entities = st.slider("Max entities to extract", 5, 50, 20)
            max_clusters = st.slider("Max topic clusters", 2, 10, 5)
        
        with col2:
            min_insight_confidence = st.slider("Min insight confidence", 0.5, 1.0, 0.7)
            include_suggestions = st.checkbox("Include improvement suggestions", True)
    
    return {
        'analysis_options': analysis_options,
        'max_entities': max_entities,
        'max_clusters': max_clusters,
        'min_insight_confidence': min_insight_confidence,
        'include_suggestions': include_suggestions
    }

def display_nlp_analysis_results(analysis: NLPAnalysisResult, options: Dict[str, Any]):
    """Display NLP analysis results"""
    
    # Readability Analysis
    if "Readability Analysis" in options['analysis_options']:
        st.subheader("📖 Readability Analysis")
        
        if analysis.readability_scores:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                flesch_score = analysis.readability_scores.get('flesch_reading_ease', 0)
                st.metric("Flesch Reading Ease", f"{flesch_score:.1f}")
                
                if flesch_score >= 60:
                    st.success("Easy to read")
                elif flesch_score >= 30:
                    st.warning("Moderately difficult")
                else:
                    st.error("Difficult to read")
            
            with col2:
                grade_level = analysis.readability_scores.get('flesch_kincaid_grade', 0)
                st.metric("Grade Level", f"{grade_level:.1f}")
            
            with col3:
                avg_sentence = analysis.readability_scores.get('avg_sentence_length', 0)
                st.metric("Avg Sentence Length", f"{avg_sentence:.1f} words")
    
    # Sentiment Analysis
    if "Sentiment Analysis" in options['analysis_options']:
        st.subheader("😊 Sentiment Analysis")
        
        if analysis.sentiment_analysis:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                compound = analysis.sentiment_analysis.get('compound', 0)
                st.metric("Overall Sentiment", f"{compound:.2f}")
            
            with col2:
                positive = analysis.sentiment_analysis.get('pos', 0)
                st.metric("Positive", f"{positive:.2f}")
            
            with col3:
                neutral = analysis.sentiment_analysis.get('neu', 0)
                st.metric("Neutral", f"{neutral:.2f}")
            
            with col4:
                negative = analysis.sentiment_analysis.get('neg', 0)
                st.metric("Negative", f"{negative:.2f}")
    
    # Entity Extraction
    if "Entity Extraction" in options['analysis_options']:
        st.subheader("🏷️ Key Entities")
        
        if analysis.key_entities:
            for entity in analysis.key_entities[:options['max_entities']]:
                with st.expander(f"{entity['text']} ({entity['label']})", expanded=False):
                    st.write(f"**Type:** {entity['description']}")
                    st.write(f"**Confidence:** {entity['confidence']:.2f}")
                    st.write(f"**Method:** {entity['method']}")
        else:
            st.info("No entities extracted")
    
    # Topic Clustering
    if "Topic Clustering" in options['analysis_options']:
        st.subheader("🎯 Topic Clusters")
        
        if analysis.topic_clusters:
            for cluster in analysis.topic_clusters[:options['max_clusters']]:
                with st.expander(f"Topic {cluster['cluster_id'] + 1} ({cluster['size']} sentences)", expanded=False):
                    st.write(f"**Keywords:** {', '.join(cluster['keywords'])}")
                    st.write(f"**Representative:** {cluster['representative_sentence']}")
                    st.write(f"**Coherence:** {cluster['coherence_score']:.2f}")
        else:
            st.info("No topic clusters found")
    
    # Content Insights
    if "Content Insights" in options['analysis_options']:
        st.subheader("💡 Content Insights")
        
        if analysis.content_insights:
            insight_types = {}
            for insight in analysis.content_insights:
                if insight.confidence >= options['min_insight_confidence']:
                    if insight.insight_type not in insight_types:
                        insight_types[insight.insight_type] = []
                    insight_types[insight.insight_type].append(insight)
            
            for insight_type, insights in insight_types.items():
                st.write(f"**{insight_type.title()}s ({len(insights)}):**")
                for insight in insights[:5]:  # Show first 5
                    st.write(f"- {insight.content[:100]}...")
        else:
            st.info("No content insights found")
    
    # Quality Metrics
    if "Quality Metrics" in options['analysis_options']:
        st.subheader("📊 Quality Metrics")
        
        if analysis.quality_metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                word_count = analysis.quality_metrics.get('word_count', 0)
                st.metric("Word Count", f"{word_count:,}")
            
            with col2:
                coherence = analysis.quality_metrics.get('coherence_score', 0)
                st.metric("Coherence Score", f"{coherence:.2f}")
            
            with col3:
                info_density = analysis.quality_metrics.get('information_density', 0)
                st.metric("Information Density", f"{info_density:.2f}")

def check_advanced_nlp_dependencies():
    """Check availability of advanced NLP dependencies"""
    status = {
        'nltk': NLTK_AVAILABLE,
        'textstat': TEXTSTAT_AVAILABLE,
        'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
        'spacy': SPACY_AVAILABLE
    }
    
    return status

# Example usage
if __name__ == "__main__":
    processor = AdvancedNLPProcessor()
    
    sample_text = """
    What is the nature of consciousness? This question has puzzled philosophers and scientists for centuries.
    The mind appears to be more than just the brain, yet how can we understand this relationship?
    
    "The real question is not whether machines think but whether men do," said B.F. Skinner.
    This provocative statement challenges our assumptions about consciousness and intelligence.
    
    In meditation, we often discover that our thoughts are not who we are. We can observe them,
    which suggests there is an observer separate from the thoughts themselves.
    """
    
    result = processor.analyze_content_comprehensively(sample_text)
    print(f"Analysis complete: {len(result.content_insights)} insights found")

