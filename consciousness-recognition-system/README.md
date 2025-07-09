# 🧠 Fine-Tune Data System - NLP Enhanced

**Advanced AI Training Data Creation with Intelligent Theme Discovery**

Transform any document into high-quality AI training data with intelligent theme discovery, advanced NLP analysis, and production-ready deployment.

## ✨ Key Features

### 🔍 **Intelligent Theme Discovery**
- **spaCy-powered content analysis** - Extract thematically relevant content chunks
- **Semantic similarity matching** - Find related content using sentence transformers
- **Interactive theme selection** - Approve/edit discovered content before enhancement
- **Predefined theme categories** - Spiritual, Psychology, Philosophy, Self-Development

### 🧠 **Advanced NLP Analysis**
- **Readability analysis** - Flesch-Kincaid, automated readability index
- **Sentiment analysis** - VADER sentiment with sentence-level variance
- **Entity extraction** - Named entities and key concepts using spaCy/NLTK
- **Topic clustering** - Automatic content clustering with keyword extraction
- **Content insights** - Questions, quotes, dialogues, key concepts
- **Quality metrics** - Coherence scoring, information density, structure analysis

### 📄 **Universal Document Processing**
- **OCR-enhanced PDF processing** - Handle scanned documents with timeout protection
- **Large file support** - Process 100+ page documents without memory issues
- **Multiple formats** - PDF, DOCX, TXT, MD with intelligent content extraction
- **Memory optimization** - Efficient processing for Render.com free tier

### ✨ **AI Enhancement Engine**
- **13 tone categories** - Professional, conversational, academic, creative, etc.
- **Multiple output types** - Q&A, summaries, insights, dialogues, instructions
- **Async processing** - Handle large datasets efficiently
- **Quality control** - Automatic routing to manual review for low-quality outputs

### 📋 **Production Features**
- **Manual review interface** - Edit and approve AI-enhanced content
- **Export flexibility** - JSONL, JSON, TXT, ZIP formats
- **Session persistence** - Auto-save progress, resume interrupted sessions
- **Memory monitoring** - Real-time memory usage and cleanup
- **Health monitoring** - Built-in health checks for deployment

## 🚀 Quick Deploy to Render.com

### **1-Click Deployment**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### **Manual Deployment**

1. **Fork/Clone Repository**
   ```bash
   git clone https://github.com/yourusername/fine-tune-data-system.git
   cd fine-tune-data-system
   ```

2. **Deploy on Render.com**
   - Create new Web Service
   - Connect your GitHub repository
   - Render will automatically use `render.yaml` configuration
   - Set environment variable: `OPENAI_API_KEY`
   - Deploy! (Takes 5-10 minutes)

3. **Environment Variables**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   LOW_MEM_MODE=true
   DEBUG_MODE=false
   ```

## 🛠️ Local Development

### **Prerequisites**
- Python 3.11+
- OpenAI API key

### **Installation**
```bash
# Clone repository
git clone https://github.com/yourusername/fine-tune-data-system.git
cd fine-tune-data-system

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Set environment variables
cp .env.example .env
# Edit .env with your OpenAI API key

# Run application
streamlit run app.py
```

## 📖 Usage Guide

### **Step 1: Upload Documents**
- Upload PDF, DOCX, TXT, or MD files
- OCR automatically processes scanned PDFs
- Large files handled with timeout protection

### **Step 2: Theme Discovery**
- Define themes/keywords relevant to your content
- Choose from predefined categories or create custom themes
- AI discovers and extracts thematically relevant chunks
- Review and approve discovered content

### **Step 3: NLP Analysis**
- Comprehensive analysis of content quality and structure
- Readability, sentiment, entities, topics, insights
- Content improvement suggestions
- Quality metrics for training data assessment

### **Step 4: AI Enhancement**
- Transform content into training-ready format
- Choose output type (Q&A, Summary, Insight, etc.)
- Select tone (Professional, Academic, Conversational, etc.)
- Async processing with progress tracking

### **Step 5: Manual Review**
- Review and edit AI-enhanced content
- Quality scoring and filtering
- Approve/reject individual items
- Bulk operations for efficiency

### **Step 6: Export & Deploy**
- Export in multiple formats (JSONL, JSON, TXT, ZIP)
- Include metadata and quality metrics
- Ready for fine-tuning or further processing

## 🔧 Advanced Configuration

### **NLP Dependencies**
The system automatically detects and uses available NLP libraries:
- **spaCy** - Advanced entity extraction and linguistic analysis
- **NLTK** - Sentiment analysis and text processing
- **sentence-transformers** - Semantic similarity and clustering
- **textstat** - Readability analysis

### **Memory Optimization**
- **Low Memory Mode** - Optimized for 512MB environments
- **Disk-based storage** - Large objects stored on disk
- **Auto-cleanup** - Automatic session cleanup
- **Progress persistence** - Resume interrupted sessions

### **OCR Configuration**
```python
OCRConfig(
    timeout_per_page=30,      # Seconds per page
    max_total_timeout=1800,   # 30 minutes total
    memory_limit_mb=400,      # Memory limit
    languages=['eng']         # OCR languages
)
```

## 📊 System Requirements

### **Render.com (Recommended)**
- **Plan**: Free tier (512MB RAM)
- **Build time**: 5-10 minutes
- **Startup time**: 30-60 seconds
- **Storage**: 1GB persistent disk

### **Local Development**
- **RAM**: 2GB+ recommended
- **Storage**: 1GB for models and cache
- **Python**: 3.11+ required

## 🔍 Troubleshooting

### **Common Issues**

**Memory Issues on Render**
- Ensure `LOW_MEM_MODE=true` is set
- Use theme discovery to reduce content size
- Process files in smaller batches

**OCR Timeouts**
- Large PDFs may timeout on free tier
- Pre-process PDFs to reduce size
- Use theme discovery to focus on relevant content

**NLP Model Loading**
- Models download automatically on first run
- May take 2-3 minutes on cold start
- Models cached for subsequent runs

**Import Errors**
- Check `requirements.txt` compatibility
- Ensure all dependencies installed
- Check Python version (3.11+ required)

### **Health Check**
Visit `/healthcheck` endpoint to verify system status:
```json
{
  "status": "healthy",
  "imports": true,
  "nlp_available": true
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **spaCy** - Industrial-strength NLP
- **NLTK** - Natural Language Toolkit
- **Streamlit** - Beautiful web apps for ML
- **OpenAI** - GPT API for content enhancement
- **Render.com** - Simple cloud deployment

---

**Built with ❤️ for the AI community**

Transform your documents into high-quality training data with intelligent theme discovery and advanced NLP analysis.

