# 🧠 Enhanced Universal AI Training Data Creator

> **Transform any content into high-quality AI training data with advanced enhancements**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

## 🎯 Overview

The Enhanced Universal AI Training Data Creator is a comprehensive, professional-grade application that transforms any content into high-quality AI training datasets. This enhanced version includes **5 Core Enhancements** and **3 Optional Add-ons** that significantly boost usability, control, and data quality.

### ✨ Key Features

- 📁 **Universal Content Extraction**: PDF, TXT, DOCX, Markdown support
- 🤖 **AI-Powered Enhancement**: GPT-based content improvement
- 🎭 **Dynamic Prompt Engineering**: Multiple spiritual tones and styles
- 📋 **Manual Review System**: Complete control over final dataset
- 🔍 **Smart Content Detection**: Automatic Q&A vs monologue processing
- 📊 **Real-time Analytics**: Comprehensive metrics and progress tracking
- 🎨 **Advanced Theming**: Professional customization options
- 📦 **Professional Export**: Multiple formats with documentation
- 🤗 **ML Integration**: Direct Hugging Face Hub upload

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd consciousness-recognition-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_enhanced.txt
   ```

3. **Set up environment**:
   ```bash
   # Create .env file with your API keys
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Run the application**:
   ```bash
   streamlit run enhanced_app.py
   ```

### Basic Workflow

1. **📁 Upload**: Choose your content file (PDF, TXT, DOCX, MD)
2. **🔍 Extract**: Automatic content extraction with smart detection
3. **✨ Enhance**: Select spiritual tone and enhance with GPT
4. **📋 Review**: Manual review and validation of results
5. **📤 Export**: Choose from multiple professional export options

---

## 🔧 Core Enhancements

### 1. 📋 Manual Review Before Export
- ✅ Individual item approval/rejection
- ✏️ Direct content editing interface
- 🎯 Quality-based filtering and sorting
- 📊 Real-time approval statistics

### 2. 🎭 Dynamic Prompt Templates Per Tone
- 📁 Organized prompt library with spiritual tones
- 🔄 Dynamic loading based on user selection
- 👀 Prompt preview functionality
- 🎨 Multiple traditions: Advaita, Zen, Sufi, Christian Mysticism

### 3. 🔍 Smart Q&A vs Monologue Detection
- 🤖 Automatic content type detection
- 📝 Q&A pattern recognition
- 📖 Passage extraction for narrative content
- 🔄 Hybrid processing for mixed formats

### 4. 🔍 Raw vs Enhanced Comparison Viewer
- 📊 Side-by-side before/after comparison
- 📈 Improvement metrics and statistics
- 🎯 Quality score analysis
- 💰 Cost tracking per enhancement

### 5. 📊 Sidebar Metrics Dashboard
- 📁 Real-time file and processing status
- 📊 Content statistics and quality distribution
- ✨ Enhancement progress and costs
- 📤 Export readiness indicators

---

## 🎨 Optional Add-ons

### 1. 🎨 Enhanced Theming System
- 🌈 7 pre-built professional themes
- 🔤 Typography and layout customization
- 👀 Live preview functionality
- ♿ Accessibility options (high contrast, large text)

### 2. 📦 Enhanced ZIP Export
- 📁 Multiple formats: JSON, JSONL, CSV, XLSX, TXT
- 📊 Comprehensive quality reports
- 📚 Complete documentation package
- 🏷️ Session metadata and analytics

### 3. 🤗 Enhanced Hugging Face Upload
- ✅ Comprehensive dataset validation
- 📋 Automatic model card generation
- 🔒 Privacy and licensing controls
- 📊 Quality filtering and preparation

---

## 📁 Project Structure

```
consciousness-recognition-system/
├── enhanced_app.py                 # Main enhanced application
├── modules/                        # Enhanced modules
│   ├── manual_review.py           # Core Enhancement 1
│   ├── dynamic_prompt_engine.py   # Core Enhancement 2
│   ├── smart_content_detector.py  # Core Enhancement 3
│   ├── enhanced_comparison_viewer.py # Core Enhancement 4
│   ├── enhanced_sidebar_metrics.py # Core Enhancement 5
│   ├── enhanced_theming.py        # Optional Add-on 1
│   ├── enhanced_zip_export.py     # Optional Add-on 2
│   └── enhanced_huggingface_upload.py # Optional Add-on 3
├── prompts/                       # Dynamic prompt templates
│   ├── universal_wisdom.txt
│   ├── advaita_vedanta.txt
│   ├── zen_buddhism.txt
│   ├── sufi_mysticism.txt
│   ├── christian_mysticism.txt
│   └── mindfulness_meditation.txt
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── requirements_enhanced.txt      # Enhanced dependencies
├── README_ENHANCED.md            # This file
├── ENHANCED_FEATURES_GUIDE.md    # Comprehensive features guide
└── .env.example                  # Environment variables template
```

---

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
```

### Streamlit Configuration

The `.streamlit/config.toml` file contains optimized settings for the enhanced application:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
enableCORS = false
```

---

## 📊 Usage Examples

### Basic Content Processing

```python
# Example: Processing a spiritual text
1. Upload your PDF/TXT file
2. Select "Advaita Vedanta" tone
3. Set enhancement strength to 0.7
4. Review and approve enhanced content
5. Export as JSONL for training
```

### Advanced Workflow

```python
# Example: Professional dataset creation
1. Upload multiple content files
2. Use smart content detection
3. Apply different tones for variety
4. Manual review with quality filtering
5. Generate comprehensive ZIP package
6. Upload to Hugging Face Hub
```

### API Integration

```python
# Example: Using exported data
from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("your-username/your-dataset")

# Or load local JSONL
import json
with open('ai_training_data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
```

---

## 🎯 Best Practices

### Content Preparation
- Use high-quality, relevant source materials
- Ensure proper formatting and structure
- Focus on spiritual/consciousness topics for best results

### Enhancement Strategy
- Start with moderate enhancement strength (0.7)
- Choose appropriate spiritual tone for your use case
- Set reasonable quality thresholds (0.6+)

### Review Process
- Review at least 10-20% of enhanced content manually
- Use quality filters to focus on problematic items
- Edit content directly in the interface when needed

### Export Optimization
- Include metadata for training transparency
- Use JSONL format for most ML applications
- Generate ZIP packages for comprehensive archival

---

## 🔍 Quality Assurance

### Validation Features
- ✅ Automatic content type detection
- 📊 Quality score calculation (0.0-1.0)
- 🎯 Threshold-based filtering
- 📈 Improvement tracking and analytics

### Quality Metrics
- **Excellent (0.8-1.0)**: Ready for immediate training
- **Good (0.6-0.8)**: High quality with minor improvements
- **Fair (0.4-0.6)**: Acceptable quality, may need review
- **Poor (0.0-0.4)**: Requires attention or rejection

---

## 🚀 Performance

### Optimization Features
- 📊 Batch processing for efficiency
- 💰 Cost tracking and management
- ⚡ Smart caching and state management
- 🔄 Progress tracking and resumption

### Scalability
- Handles files up to 200MB
- Processes thousands of examples
- Efficient memory management
- Parallel processing where possible

---

## 🤝 Contributing

We welcome contributions to improve the Enhanced Universal AI Training Data Creator!

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies**:
   ```bash
   pip install -r requirements_enhanced.txt
   pip install pytest pytest-streamlit
   ```
4. **Make your changes**
5. **Run tests**:
   ```bash
   pytest tests/
   ```
6. **Submit a pull request**

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility

---

## 📞 Support

### Getting Help

1. **📚 Documentation**: Check `ENHANCED_FEATURES_GUIDE.md` for detailed feature documentation
2. **🐛 Issues**: Report bugs and request features via GitHub Issues
3. **💬 Discussions**: Join community discussions for questions and ideas

### Troubleshooting

#### Common Issues
- **Module Import Errors**: Ensure all files are in correct directory structure
- **API Failures**: Check API key configuration and internet connectivity
- **Memory Issues**: Use content limiting options for large files

#### Performance Tips
- Process large files in smaller batches
- Use quality thresholds to filter content
- Monitor API usage and costs

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for GPT API and language models
- **Streamlit** for the excellent web framework
- **Hugging Face** for ML ecosystem integration
- **Community Contributors** for feedback and improvements

---

## 🔮 Roadmap

### Upcoming Features
- 🌐 Multi-language support
- 🤖 Additional AI provider integrations
- 📱 Mobile-responsive interface
- 🔄 Automated quality improvement suggestions

### Long-term Vision
- 🧠 Advanced consciousness detection algorithms
- 🌍 Community-driven prompt library
- 📊 Advanced analytics and insights
- 🤝 Collaborative dataset creation

---

**Ready to create professional AI training datasets? Get started with the Enhanced Universal AI Training Data Creator today!**

[![Get Started](https://img.shields.io/badge/Get%20Started-FF6B6B?style=for-the-badge&logo=rocket&logoColor=white)](#-quick-start)
[![Documentation](https://img.shields.io/badge/Documentation-4285F4?style=for-the-badge&logo=googledocs&logoColor=white)](ENHANCED_FEATURES_GUIDE.md)

