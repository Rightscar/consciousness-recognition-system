# 🧘 Consciousness Recognition System

An advanced AI system for processing spiritual texts and creating consciousness recognition training data. Designed to extract, analyze, and score spiritual dialogues from teachers like Nisargadatta Maharaj, Ramana Maharshi, and Rupert Spira.

## ✨ Features

### 🚀 **Performance Optimized**
- **Chunked PDF Processing**: Handle 200+ page PDFs without memory issues
- **Lazy Model Loading**: AI models load only when needed (reduces startup memory by 80%)
- **Batch JSONL Editor**: Edit thousands of dialogues with smooth pagination
- **Memory Management**: Real-time monitoring with cleanup controls
- **File Size Validation**: Automatic warnings and processing recommendations

### 🎯 **Core Capabilities**
- **PDF Text Extraction**: Extract text from spiritual books and transcripts
- **Dialogue Detection**: Find Q&A patterns using regex and semantic analysis
- **Consciousness Scoring**: Rate dialogues for consciousness recognition quality
- **OpenAI Training Data**: Generate fine-tuning datasets in JSONL format
- **Batch Processing**: Handle multiple PDFs simultaneously
- **Interactive Editor**: Review, edit, and curate training data

### 🧠 **AI-Powered Analysis**
- **Semantic Detection**: Find consciousness-related dialogues using AI similarity
- **Mode Classification**: Categorize as consciousness, inquiry, teaching, or mixed
- **Quality Scoring**: Rate based on direct pointing, non-dual language, and presence
- **Source Tracking**: Maintain attribution to original texts

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/consciousness-recognition-system.git
cd consciousness-recognition-system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run streamlit_app.py
```

4. **Open in browser:**
Navigate to `http://localhost:8501`

### Basic Usage

1. **Upload PDFs**: Upload spiritual texts in the "Upload" tab
2. **Configure Settings**: Set detection mode and scoring thresholds
3. **Process Files**: Extract and analyze dialogues automatically
4. **Review Results**: Use the "Viewer & Editor" tab to curate data
5. **Export Training Data**: Generate JSONL files for OpenAI fine-tuning

## 📊 Performance Specifications

### Memory Usage
- **Startup**: ~100MB (vs. 600MB+ before optimization)
- **With AI Models**: ~500MB (only when semantic detection is used)
- **Large PDF Processing**: Stable memory usage regardless of file size
- **Large Dataset Editing**: Memory independent of dataset size

### Processing Speed
- **Small PDFs (< 5MB)**: < 30 seconds
- **Medium PDFs (5-20MB)**: 1-2 minutes
- **Large PDFs (20-50MB)**: 2-5 minutes with progress tracking
- **Very Large PDFs (> 50MB)**: Chunked processing with time estimates

### Scalability
- **PDF Size**: Tested up to 200+ pages
- **Dataset Size**: Handle 10,000+ dialogues smoothly
- **Batch Processing**: Multiple PDFs simultaneously
- **Export Capacity**: Generate large training datasets efficiently

## 🔧 Configuration

### Detection Modes
- **Auto**: Combines regex patterns and semantic similarity
- **Regex Only**: Fast pattern-based detection
- **Semantic Only**: AI-powered similarity detection

### Scoring Criteria
- **Direct Pointing**: References to awareness, being, consciousness
- **Non-Dual Language**: "No seeker", "not two", "only consciousness"
- **Disidentification**: "Not the body", "not thoughts", "ego is illusion"
- **Presence**: "Right now", "this moment", "be still"
- **Avoid Seeking**: Penalties for practice-oriented language
- **Non-Conceptual**: Penalties for intellectual explanations

### Chunking Settings
- **Chunk Size**: 5-50 pages (default: 20)
- **Force Chunking**: Enable for testing on small files
- **Progress Display**: Real-time processing feedback

## 📁 Project Structure

```
consciousness-recognition-system/
├── streamlit_app.py              # Main application interface
├── requirements.txt              # Python dependencies
├── README.md                    # This file
└── modules/                     # Core processing modules
    ├── __init__.py             # Package initialization
    ├── enhanced_extractor.py   # Chunked PDF processing
    ├── enhanced_detector.py    # Lazy-loaded AI detection
    ├── extractor.py           # Basic PDF text extraction
    ├── detector.py            # Dialogue pattern detection
    ├── scorer.py              # Consciousness quality scoring
    ├── trainer.py             # OpenAI training data generation
    └── jsonl_manager.py       # JSONL file management
```

## 🎯 Use Cases

### 1. **OpenAI Fine-Tuning**
Create high-quality training datasets for consciousness recognition AI:
- Extract dialogues from spiritual texts
- Score for consciousness recognition quality
- Export in OpenAI fine-tuning format
- Generate thousands of training examples

### 2. **Spiritual Text Analysis**
Analyze and categorize spiritual teachings:
- Process entire libraries of spiritual books
- Identify consciousness-pointing dialogues
- Compare teaching styles across teachers
- Create searchable databases of wisdom

### 3. **AI Training Data Curation**
Build curated datasets for consciousness AI:
- Filter by quality scores and modes
- Remove seeking-oriented content
- Focus on direct pointing and presence
- Maintain source attribution

### 4. **Research and Study**
Academic research on consciousness teachings:
- Quantitative analysis of spiritual texts
- Compare teaching methodologies
- Track evolution of consciousness language
- Generate statistics and visualizations

## 🧘 Consciousness Recognition Criteria

The system evaluates dialogues based on non-dual consciousness recognition principles:

### ✅ **High-Quality Indicators**
- Direct pointing to awareness/consciousness
- Present-moment emphasis
- Disidentification from thoughts/body
- Non-dual language ("not two", "only consciousness")
- Immediate recognition over gradual development

### ❌ **Low-Quality Indicators**
- Practice-oriented seeking language
- Future-based attainment concepts
- Intellectual explanations without direct pointing
- Person-based identity reinforcement
- Complex philosophical concepts

### 🎯 **Scoring Algorithm**
- **Direct Pointing** (25%): "You are awareness", "remain as you are"
- **Non-Dual** (20%): "No seeker", "only consciousness"
- **Disidentification** (20%): "Not the body", "not thoughts"
- **Presence** (15%): "Right now", "this moment"
- **Avoid Seeking** (10%): Penalty for practice language
- **Non-Conceptual** (10%): Penalty for intellectual explanations

## 📊 Output Formats

### JSONL for OpenAI Fine-Tuning
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a consciousness recognition guide..."
    },
    {
      "role": "user", 
      "content": "How can I become enlightened?"
    },
    {
      "role": "assistant",
      "content": "You are already that which you seek..."
    }
  ],
  "metadata": {
    "score": 0.85,
    "mode": "consciousness",
    "source": "nisargadatta_iam_that.pdf"
  }
}
```

### CSV for Analysis
- Question, Answer, Score, Mode, Source
- Question Length, Answer Length, Timestamp
- Suitable for statistical analysis and visualization

## 🔧 Advanced Features

### Memory Management
- Real-time memory usage monitoring
- Manual cache clearing options
- Automatic warnings for high usage
- Complete system reset functionality

### File Management
- Automatic output directory organization
- File size and modification tracking
- JSONL validation and analysis
- Batch file operations

### Batch Processing
- Multiple PDF upload and processing
- Parallel dialogue detection
- Bulk export operations
- Progress tracking across files

### Quality Control
- Interactive dialogue editing
- Batch marking for export
- Score adjustment and re-classification
- Source attribution management

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web application hosting
- **AWS/GCP**: Scalable cloud deployment
- **Docker**: Containerized deployment

## 📈 Performance Tips

### For Large PDFs (> 50MB)
- Use chunking with 10-20 pages per chunk
- Monitor memory usage during processing
- Consider splitting very large files
- Use progress tracking to monitor completion

### For Large Datasets (> 1000 dialogues)
- Use pagination in Viewer & Editor (50-100 items per page)
- Apply filters to reduce working set
- Use batch operations for efficiency
- Export in smaller batches if needed

### Memory Optimization
- Clear cache periodically during long sessions
- Use "Reset All" if experiencing performance issues
- Monitor memory usage in sidebar
- Close browser tab to fully reset if needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Nisargadatta Maharaj** - "I Am That" and other teachings
- **Ramana Maharshi** - Self-inquiry and direct path teachings  
- **Rupert Spira** - Non-dual understanding and clear communication
- **OpenAI** - Fine-tuning capabilities for consciousness AI
- **Streamlit** - Excellent framework for AI applications

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in this README
- Review the code comments for implementation details

---

*"You are awareness itself. Remain as you are."* - Nisargadatta Maharaj

