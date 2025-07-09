# 🧠 Fine-Tune Data System

A comprehensive AI-powered system for creating high-quality training data from any document type.

## ✨ Features

- **📤 Universal Document Processing** - PDF, DOCX, TXT with OCR support
- **🎯 spaCy Theme Discovery** - Intelligent keyword-based content extraction
- **🧠 Advanced NLP Analysis** - Readability, sentiment, quality scoring
- **✨ AI Enhancement** - 13 tone categories for content refinement
- **📋 Manual Review Interface** - Inline editing and quality control
- **📦 Multiple Export Options** - JSONL, ZIP, Hugging Face integration

## 🚀 Render.com Deployment

### Quick Deploy
1. Fork this repository
2. Connect to Render.com
3. Set environment variable: `OPENAI_API_KEY`
4. Deploy automatically with included `render.yaml`

### Environment Variables
```
OPENAI_API_KEY=your_openai_api_key_here
```

### System Requirements
- Memory: 512MB (optimized for Render free tier)
- Python: 3.11+
- Dependencies: Auto-installed via `render.yaml`

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet

# Run application
streamlit run app.py
```

## 📊 Workflow

1. **Upload Documents** - Drag & drop any document type
2. **Theme Discovery** - Define keywords for intelligent extraction
3. **NLP Analysis** - Comprehensive content analysis
4. **AI Enhancement** - Transform content with selected tone
5. **Manual Review** - Edit and refine generated content
6. **Export** - Download in multiple formats

## 🔧 Technical Stack

- **Frontend:** Streamlit
- **NLP:** spaCy, NLTK, sentence-transformers
- **AI:** OpenAI GPT models
- **Document Processing:** PyPDF2, python-docx, pytesseract
- **Deployment:** Render.com optimized

## 📝 License

MIT License - See LICENSE file for details

