# 🧠 Fine-Tune Data System - NLP Enhanced

**Advanced AI training data creation with intelligent theme discovery and comprehensive NLP analysis**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 🚀 **Quick Deploy to Render.com**

### **One-Click Deployment**
1. **Fork this repository** to your GitHub account
2. **Connect to Render.com** and create a new Web Service
3. **Connect your GitHub repo** and select this repository
4. **Set Environment Variable:**
   - `OPENAI_API_KEY` = your OpenAI API key
5. **Deploy!** - Render will automatically use the included `render.yaml`

### **Manual Deployment Steps**
```bash
# 1. Clone repository
git clone https://github.com/yourusername/fine-tune-data-system.git
cd fine-tune-data-system

# 2. Push to your GitHub repository
git remote set-url origin https://github.com/yourusername/fine-tune-data-system.git
git push -u origin main

# 3. Deploy on Render.com
# - Create Web Service
# - Connect GitHub repository
# - Set OPENAI_API_KEY environment variable
# - Deploy automatically with render.yaml
```

## ✨ **Features**

### **🎯 Core Capabilities**
- **📤 Universal Document Upload** - PDF, TXT, DOCX, EPUB with OCR support
- **🔍 Intelligent Theme Discovery** - spaCy-powered keyword extraction and semantic analysis
- **📊 Advanced NLP Analysis** - Readability, sentiment, entities, topic clustering
- **✨ AI Enhancement** - 13 tone categories for content refinement
- **📋 Manual Review Interface** - Inline editing and quality control
- **📦 Export & Deploy** - JSONL, ZIP, and Hugging Face integration

### **🧠 NLP-Enhanced Workflow**
1. **📤 Upload & Extract** - Universal document processing with OCR
2. **🎯 Define Themes** - Input keywords like "suffering", "mind", "choice", "truth"
3. **🔍 Intelligent Discovery** - spaCy PhraseMatcher + semantic similarity finds relevant chunks
4. **📋 Review & Approve** - Each chunk shows keyword match + surrounding context with ✅/❌ options
5. **✏️ Edit & Refine** - Edit chunks, refine keywords, re-run matching
6. **🚀 Send to GPT** - Specify output type (Q&A, Summary, Insight) + tone (Business/Neutral/Reflective)
7. **💾 Structured Export** - All responses tied to original content for training

### **🛡️ Production Features**
- **Memory Optimization** - Works perfectly on Render.com free tier (512MB)
- **Large File Handling** - Timeout protection for 100+ page PDFs
- **Session Persistence** - Auto-save and resume functionality
- **Health Monitoring** - Built-in health checks and error recovery
- **Import Safety** - Graceful fallbacks when dependencies unavailable

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
LOW_MEM_MODE=true          # Enable memory optimization
DEBUG_MODE=false           # Enable debug logging
RENDER_DEPLOYMENT=true     # Render.com specific optimizations
```

### **Render.com Optimization**
The system is specifically optimized for Render.com deployment:
- ✅ **Memory efficient** - Works within 512MB free tier limit
- ✅ **Fast startup** - Optimized dependency loading
- ✅ **Health checks** - Built-in monitoring endpoints
- ✅ **Error recovery** - Graceful handling of hibernation/wake cycles

## 📋 **Requirements**

### **System Requirements**
- **Python 3.11+**
- **Memory:** 512MB minimum (optimized for Render.com free tier)
- **Storage:** 1GB for temporary files and models

### **Dependencies**
All dependencies are automatically installed via `requirements.txt`:
- **Core:** Streamlit, OpenAI, Pandas, NumPy
- **NLP:** spaCy, NLTK, sentence-transformers, textstat
- **Document Processing:** PyPDF2, python-docx, ebooklib
- **OCR (Optional):** pytesseract, Pillow, pdf2image

## 🏗️ **Architecture**

### **Project Structure**
```
fine-tune-data-system/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Production dependencies
├── render.yaml                     # Render.com deployment config
├── README.md                       # This file
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── healthcheck.py                  # Health monitoring endpoint
├── deployment_validator.py         # Pre-deployment validation
├── .streamlit/
│   ├── config.toml                 # Streamlit configuration
│   ├── config_render.toml          # Render-specific config
│   └── secrets.toml                # Local secrets (not in repo)
├── modules/                        # Core processing modules
│   ├── spacy_theme_discovery.py    # Theme-based content discovery
│   ├── advanced_nlp_features.py    # Comprehensive NLP analysis
│   ├── large_file_ocr_handler.py   # Timeout-protected OCR
│   ├── import_safety.py            # Import error prevention
│   └── [30+ other modules]         # Complete feature set
├── prompts/                        # Tone enhancement templates
│   ├── neutral_professional.txt    # Business communication
│   ├── academic_scholarly.txt      # Research and analysis
│   ├── conversational_friendly.txt # Casual interaction
│   └── [10+ other tones]           # Complete tone library
├── logs/                           # Application logs
└── sessions/                       # Session persistence
```

### **Key Modules**
- **`spacy_theme_discovery.py`** - Intelligent keyword-based content discovery
- **`advanced_nlp_features.py`** - Comprehensive text analysis and quality scoring
- **`large_file_ocr_handler.py`** - Timeout-protected OCR for large documents
- **`import_safety.py`** - Bulletproof import handling with fallbacks
- **`enhanced_tone_manager.py`** - 13 tone categories for content enhancement

## 🔍 **Usage**

### **Basic Workflow**
1. **Upload Documents** - Drag & drop PDF, TXT, DOCX, or EPUB files
2. **Define Themes** - Enter keywords relevant to your content
3. **Review Discoveries** - Approve or reject theme-matched content chunks
4. **Analyze with NLP** - Get comprehensive text analysis and quality scores
5. **Enhance with AI** - Choose tone and enhancement type for each chunk
6. **Manual Review** - Edit and refine the enhanced content
7. **Export Results** - Download as JSONL, ZIP, or upload to Hugging Face

### **Advanced Features**
- **Large File Processing** - Handles 100+ page PDFs with timeout protection
- **Session Recovery** - Resume work after browser refresh or hibernation
- **Quality Scoring** - Automatic content quality assessment
- **Batch Processing** - Process multiple documents simultaneously
- **Custom Prompts** - Define your own enhancement prompts

## 🛠️ **Development**

### **Local Development**
```bash
# Clone repository
git clone https://github.com/yourusername/fine-tune-data-system.git
cd fine-tune-data-system

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Set environment variables
cp .env.example .env
# Edit .env with your OpenAI API key

# Run locally
streamlit run app.py
```

### **Testing**
```bash
# Validate deployment readiness
python deployment_validator.py

# Test imports
python -c "import app; print('✅ All imports successful')"

# Run health check
python healthcheck.py
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Memory Issues on Render.com**
- ✅ **Solution:** `LOW_MEM_MODE=true` is enabled by default
- ✅ **Automatic:** Large models are loaded lazily and cached efficiently

#### **OCR Dependencies Missing**
- ✅ **Solution:** OCR is optional - system works without Tesseract
- ✅ **Fallback:** Graceful degradation to text-only processing

#### **spaCy Model Download Fails**
- ✅ **Solution:** Automatic retry in buildCommand
- ✅ **Fallback:** Basic NLP features work without spaCy models

#### **Import Errors**
- ✅ **Solution:** Comprehensive import safety system
- ✅ **Prevention:** All conditional imports have fallback classes

### **Health Monitoring**
- **Health Check Endpoint:** `/healthcheck` - Returns system status
- **Logs:** Available in Render dashboard under "Logs" tab
- **Session Recovery:** Automatic resume after hibernation

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 **Support**

- **Issues:** [GitHub Issues](https://github.com/yourusername/fine-tune-data-system/issues)
- **Documentation:** This README and inline code documentation
- **Health Check:** Visit `/healthcheck` on your deployed app

---

**Deploy your AI training data creation system in minutes with Render.com!** 🚀

