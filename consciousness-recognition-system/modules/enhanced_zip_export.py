"""
Enhanced ZIP Export Module
==========================

Implements enhanced ZIP export functionality for comprehensive session output.
Provides multiple export formats and detailed documentation.

Features:
- Multiple export formats (JSON, JSONL, CSV, XLSX)
- Session documentation and metadata
- Quality reports and statistics
- Comparison data (raw vs enhanced)
- User-friendly file organization
"""

import streamlit as st
import json
import csv
import zipfile
import pandas as pd
from io import BytesIO, StringIO
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import production hardening
try:
    from production_hardening import safe_export_training_data, production_hardening
    HARDENING_AVAILABLE = True
except ImportError:
    HARDENING_AVAILABLE = False


class EnhancedZipExporter:
    """Enhanced ZIP Export Functionality - Optional Add-on"""
    
    def __init__(self):
        self.export_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.supported_formats = ['json', 'jsonl', 'csv', 'xlsx', 'txt']
    
    def render_export_ui(self,
                        raw_content: Optional[List[Dict[str, Any]]] = None,
                        enhanced_content: Optional[List[Dict[str, Any]]] = None,
                        final_content: Optional[List[Dict[str, Any]]] = None,
                        session_stats: Optional[Dict[str, Any]] = None) -> None:
        """Render enhanced export UI with multiple options"""
        
        st.markdown("### 📦 **Enhanced ZIP Export**")
        
        if not final_content:
            st.info("📝 No content available for export. Process some content first.")
            return
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 **Export Options**")
            
            include_raw = st.checkbox(
                "📄 Include Raw Content",
                value=True,
                help="Include original extracted content before enhancement"
            )
            
            include_enhanced = st.checkbox(
                "✨ Include Enhanced Content",
                value=True,
                help="Include GPT-enhanced content with metadata"
            )
            
            include_comparison = st.checkbox(
                "🔍 Include Comparison Report",
                value=True,
                help="Include side-by-side comparison of raw vs enhanced"
            )
            
            include_stats = st.checkbox(
                "📊 Include Statistics",
                value=True,
                help="Include session statistics and quality metrics"
            )
        
        with col2:
            st.markdown("#### 🗂️ **Format Options**")
            
            export_formats = st.multiselect(
                "Select Export Formats",
                options=['JSON', 'JSONL', 'CSV', 'XLSX', 'TXT'],
                default=['JSON', 'JSONL'],
                help="Choose which formats to include in the ZIP"
            )
            
            include_documentation = st.checkbox(
                "📚 Include Documentation",
                value=True,
                help="Include README and format documentation"
            )
            
            include_metadata = st.checkbox(
                "🏷️ Include Metadata Files",
                value=True,
                help="Include session metadata and processing details"
            )
        
        # Custom filename
        st.markdown("#### 📝 **Export Settings**")
        
        custom_filename = st.text_input(
            "Custom Filename Prefix",
            value="ai_training_session",
            help="Prefix for the exported ZIP file"
        )
        
        # Generate export
        if st.button("🚀 **Generate Enhanced ZIP Export**", type="primary"):
            try:
                with st.spinner("📦 Creating comprehensive export package..."):
                    zip_buffer = self.create_enhanced_session_zip(
                        raw_content=raw_content if include_raw else None,
                        enhanced_content=enhanced_content if include_enhanced else None,
                        final_content=final_content,
                        session_stats=session_stats if include_stats else None,
                        export_formats=[fmt.lower() for fmt in export_formats],
                        include_comparison=include_comparison,
                        include_documentation=include_documentation,
                        include_metadata=include_metadata,
                        filename_prefix=custom_filename
                    )
                
                # Provide download
                filename = f"{custom_filename}_{self.export_timestamp}.zip"
                
                st.download_button(
                    label="💾 **Download ZIP Package**",
                    data=zip_buffer.getvalue(),
                    file_name=filename,
                    mime="application/zip",
                    help=f"Download your complete training data package: {filename}"
                )
                
                st.success(f"✅ **Export package created successfully!** ({filename})")
                
                # Show export summary
                self._render_export_summary(
                    final_content, export_formats, include_raw, 
                    include_enhanced, include_comparison, include_stats
                )
                
            except Exception as e:
                st.error(f"❌ **Export failed:** {str(e)}")
                logger.error(f"ZIP export error: {e}")
    
    def create_enhanced_session_zip(self,
                                  raw_content: Optional[List[Dict[str, Any]]] = None,
                                  enhanced_content: Optional[List[Dict[str, Any]]] = None,
                                  final_content: Optional[List[Dict[str, Any]]] = None,
                                  session_stats: Optional[Dict[str, Any]] = None,
                                  export_formats: List[str] = None,
                                  include_comparison: bool = True,
                                  include_documentation: bool = True,
                                  include_metadata: bool = True,
                                  filename_prefix: str = "ai_training_session") -> BytesIO:
        """Create comprehensive ZIP file with all session data and documentation"""
        
        if export_formats is None:
            export_formats = ['json', 'jsonl']
        
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # 1. Main training data in requested formats
            if final_content:
                self._add_training_data(zip_file, final_content, export_formats)
            
            # 2. Raw content (if requested)
            if raw_content:
                self._add_raw_content(zip_file, raw_content, export_formats)
            
            # 3. Enhanced content (if requested)
            if enhanced_content:
                self._add_enhanced_content(zip_file, enhanced_content, export_formats)
            
            # 4. Comparison report (if requested)
            if include_comparison and raw_content and enhanced_content:
                self._add_comparison_report(zip_file, raw_content, enhanced_content)
            
            # 5. Statistics and metadata (if requested)
            if session_stats and include_metadata:
                self._add_session_metadata(zip_file, session_stats)
            
            # 6. Documentation (if requested)
            if include_documentation:
                self._add_documentation(zip_file, export_formats)
            
            # 7. Quality reports
            if enhanced_content:
                self._add_quality_reports(zip_file, enhanced_content)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def _add_training_data(self, zip_file: zipfile.ZipFile, content: List[Dict[str, Any]], formats: List[str]) -> None:
        """Add main training data in requested formats"""
        
        # JSON format
        if 'json' in formats:
            json_data = json.dumps(content, indent=2, ensure_ascii=False)
            zip_file.writestr("training_data/final_training_data.json", json_data)
        
        # JSONL format
        if 'jsonl' in formats:
            jsonl_data = '\n'.join(json.dumps(item, ensure_ascii=False) for item in content)
            zip_file.writestr("training_data/final_training_data.jsonl", jsonl_data)
        
        # CSV format
        if 'csv' in formats and content:
            csv_buffer = StringIO()
            fieldnames = content[0].keys()
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(content)
            zip_file.writestr("training_data/final_training_data.csv", csv_buffer.getvalue())
        
        # XLSX format
        if 'xlsx' in formats and content:
            df = pd.DataFrame(content)
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            zip_file.writestr("training_data/final_training_data.xlsx", excel_buffer.getvalue())
        
        # TXT format (human-readable)
        if 'txt' in formats:
            txt_content = self._format_as_text(content)
            zip_file.writestr("training_data/final_training_data.txt", txt_content)
    
    def _add_raw_content(self, zip_file: zipfile.ZipFile, content: List[Dict[str, Any]], formats: List[str]) -> None:
        """Add raw extracted content"""
        
        if 'json' in formats:
            json_data = json.dumps(content, indent=2, ensure_ascii=False)
            zip_file.writestr("raw_data/raw_extracted_content.json", json_data)
        
        if 'jsonl' in formats:
            jsonl_data = '\n'.join(json.dumps(item, ensure_ascii=False) for item in content)
            zip_file.writestr("raw_data/raw_extracted_content.jsonl", jsonl_data)
    
    def _add_enhanced_content(self, zip_file: zipfile.ZipFile, content: List[Dict[str, Any]], formats: List[str]) -> None:
        """Add enhanced content with metadata"""
        
        if 'json' in formats:
            json_data = json.dumps(content, indent=2, ensure_ascii=False)
            zip_file.writestr("enhanced_data/enhanced_content.json", json_data)
        
        if 'jsonl' in formats:
            jsonl_data = '\n'.join(json.dumps(item, ensure_ascii=False) for item in content)
            zip_file.writestr("enhanced_data/enhanced_content.jsonl", jsonl_data)
    
    def _add_comparison_report(self, zip_file: zipfile.ZipFile, raw_content: List[Dict[str, Any]], enhanced_content: List[Dict[str, Any]]) -> None:
        """Add detailed comparison report"""
        
        comparison_report = self._generate_comparison_report(raw_content, enhanced_content)
        
        # HTML report
        html_report = self._generate_html_comparison_report(comparison_report)
        zip_file.writestr("reports/comparison_report.html", html_report)
        
        # JSON report
        json_report = json.dumps(comparison_report, indent=2, ensure_ascii=False)
        zip_file.writestr("reports/comparison_report.json", json_report)
        
        # CSV summary
        if comparison_report.get('item_comparisons'):
            df = pd.DataFrame(comparison_report['item_comparisons'])
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            zip_file.writestr("reports/comparison_summary.csv", csv_buffer.getvalue())
    
    def _add_session_metadata(self, zip_file: zipfile.ZipFile, session_stats: Dict[str, Any]) -> None:
        """Add session metadata and statistics"""
        
        # Session metadata
        metadata = {
            'export_timestamp': self.export_timestamp,
            'export_date': datetime.now().isoformat(),
            'session_stats': session_stats,
            'export_version': '2.0',
            'format_version': '1.0'
        }
        
        json_metadata = json.dumps(metadata, indent=2, ensure_ascii=False)
        zip_file.writestr("metadata/session_metadata.json", json_metadata)
        
        # Processing log
        if session_stats.get('processing_log'):
            log_content = '\n'.join(session_stats['processing_log'])
            zip_file.writestr("metadata/processing_log.txt", log_content)
    
    def _add_documentation(self, zip_file: zipfile.ZipFile, formats: List[str]) -> None:
        """Add comprehensive documentation"""
        
        # Main README
        readme_content = self._generate_readme(formats)
        zip_file.writestr("README.md", readme_content)
        
        # Format documentation
        format_docs = self._generate_format_documentation()
        zip_file.writestr("documentation/format_guide.md", format_docs)
        
        # Usage examples
        usage_examples = self._generate_usage_examples()
        zip_file.writestr("documentation/usage_examples.md", usage_examples)
        
        # Data schema
        schema_docs = self._generate_schema_documentation()
        zip_file.writestr("documentation/data_schema.md", schema_docs)
    
    def _add_quality_reports(self, zip_file: zipfile.ZipFile, enhanced_content: List[Dict[str, Any]]) -> None:
        """Add quality analysis reports"""
        
        quality_report = self._generate_quality_report(enhanced_content)
        
        # JSON quality report
        json_report = json.dumps(quality_report, indent=2, ensure_ascii=False)
        zip_file.writestr("reports/quality_report.json", json_report)
        
        # HTML quality dashboard
        html_dashboard = self._generate_html_quality_dashboard(quality_report)
        zip_file.writestr("reports/quality_dashboard.html", html_dashboard)
    
    def _generate_comparison_report(self, raw_content: List[Dict[str, Any]], enhanced_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed comparison analysis"""
        
        # Create mapping of enhanced content
        enhanced_map = {item.get('original_question', ''): item for item in enhanced_content if item.get('enhanced', False)}
        
        comparisons = []
        total_raw_words = 0
        total_enhanced_words = 0
        
        for raw_item in raw_content:
            raw_question = raw_item.get('question', '')
            raw_answer = raw_item.get('answer', '')
            
            enhanced_item = enhanced_map.get(raw_question)
            
            if enhanced_item:
                enhanced_question = enhanced_item.get('question', '')
                enhanced_answer = enhanced_item.get('answer', '')
                
                raw_words = len((raw_question + ' ' + raw_answer).split())
                enhanced_words = len((enhanced_question + ' ' + enhanced_answer).split())
                
                total_raw_words += raw_words
                total_enhanced_words += enhanced_words
                
                improvement = ((enhanced_words - raw_words) / max(raw_words, 1)) * 100
                
                comparisons.append({
                    'raw_question': raw_question[:100] + '...' if len(raw_question) > 100 else raw_question,
                    'raw_words': raw_words,
                    'enhanced_words': enhanced_words,
                    'improvement_percent': round(improvement, 1),
                    'quality_score': enhanced_item.get('quality_score', 0),
                    'enhancement_tone': enhanced_item.get('enhancement_tone', 'unknown')
                })
        
        return {
            'summary': {
                'total_raw_items': len(raw_content),
                'total_enhanced_items': len([item for item in enhanced_content if item.get('enhanced', False)]),
                'total_raw_words': total_raw_words,
                'total_enhanced_words': total_enhanced_words,
                'average_improvement': ((total_enhanced_words - total_raw_words) / max(total_raw_words, 1)) * 100,
                'enhancement_rate': (len(comparisons) / max(len(raw_content), 1)) * 100
            },
            'item_comparisons': comparisons,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_quality_report(self, enhanced_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive quality analysis"""
        
        enhanced_items = [item for item in enhanced_content if item.get('enhanced', False)]
        
        if not enhanced_items:
            return {'error': 'No enhanced content available for quality analysis'}
        
        quality_scores = [item.get('quality_score', 0) for item in enhanced_items]
        costs = [item.get('enhancement_cost', 0) for item in enhanced_items]
        tones = [item.get('enhancement_tone', 'unknown') for item in enhanced_items]
        
        return {
            'summary': {
                'total_enhanced_items': len(enhanced_items),
                'average_quality_score': sum(quality_scores) / len(quality_scores),
                'min_quality_score': min(quality_scores),
                'max_quality_score': max(quality_scores),
                'total_enhancement_cost': sum(costs),
                'average_cost_per_item': sum(costs) / len(costs) if costs else 0
            },
            'quality_distribution': {
                'excellent': sum(1 for score in quality_scores if score >= 0.8),
                'good': sum(1 for score in quality_scores if 0.6 <= score < 0.8),
                'fair': sum(1 for score in quality_scores if 0.4 <= score < 0.6),
                'poor': sum(1 for score in quality_scores if score < 0.4)
            },
            'tone_distribution': {tone: tones.count(tone) for tone in set(tones)},
            'cost_analysis': {
                'total_cost': sum(costs),
                'min_cost': min(costs) if costs else 0,
                'max_cost': max(costs) if costs else 0,
                'average_cost': sum(costs) / len(costs) if costs else 0
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def _format_as_text(self, content: List[Dict[str, Any]]) -> str:
        """Format content as human-readable text"""
        
        text_lines = [
            "AI Training Data - Text Format",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Items: {len(content)}",
            "",
            "=" * 40,
            ""
        ]
        
        for i, item in enumerate(content, 1):
            text_lines.extend([
                f"Item {i}:",
                "-" * 20,
                f"Question: {item.get('question', 'N/A')}",
                "",
                f"Answer: {item.get('answer', 'N/A')}",
                ""
            ])
            
            # Add metadata if available
            if item.get('quality_score'):
                text_lines.append(f"Quality Score: {item['quality_score']}")
            if item.get('enhancement_tone'):
                text_lines.append(f"Enhancement Tone: {item['enhancement_tone']}")
            
            text_lines.extend(["", "=" * 40, ""])
        
        return '\n'.join(text_lines)
    
    def _generate_readme(self, formats: List[str]) -> str:
        """Generate comprehensive README file"""
        
        return f"""# AI Training Data Export Package

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📁 Package Contents

This ZIP package contains your processed AI training data in multiple formats and comprehensive documentation.

### 🗂️ Directory Structure

```
├── training_data/          # Final training data ready for use
├── raw_data/              # Original extracted content (if included)
├── enhanced_data/         # GPT-enhanced content (if included)
├── reports/               # Analysis and comparison reports
├── metadata/              # Session metadata and processing logs
├── documentation/         # Format guides and usage examples
└── README.md             # This file
```

### 📋 Included Formats

{', '.join(formats).upper()}

### 🚀 Quick Start

1. **For Training**: Use files in `training_data/` directory
2. **For Analysis**: Check `reports/` for quality metrics
3. **For Understanding**: Read `documentation/` for format details

### 📊 Data Quality

- All content has been processed and validated
- Quality scores and metrics available in reports
- Comparison data shows improvement over raw content

### 🔧 Usage

See `documentation/usage_examples.md` for specific implementation examples.

### 📞 Support

For questions about this data format or usage, refer to the documentation files included in this package.

---
Generated by Universal AI Training Data Creator v2.0
"""
    
    def _generate_format_documentation(self) -> str:
        """Generate format documentation"""
        
        return """# Data Format Documentation

## 📋 Supported Formats

### JSON (.json)
- **Purpose**: Structured data with full metadata
- **Use Case**: Programming, data analysis
- **Structure**: Array of objects with question/answer pairs

### JSONL (.jsonl)
- **Purpose**: Line-delimited JSON for streaming
- **Use Case**: Machine learning training, large datasets
- **Structure**: One JSON object per line

### CSV (.csv)
- **Purpose**: Spreadsheet-compatible format
- **Use Case**: Data analysis, Excel import
- **Structure**: Comma-separated values with headers

### XLSX (.xlsx)
- **Purpose**: Excel spreadsheet format
- **Use Case**: Business analysis, reporting
- **Structure**: Formatted Excel workbook

### TXT (.txt)
- **Purpose**: Human-readable format
- **Use Case**: Review, documentation
- **Structure**: Formatted text with clear separators

## 🏗️ Data Schema

Each training example contains:

- `question`: The training question/prompt
- `answer`: The corresponding answer/response
- `quality_score`: Quality rating (0.0-1.0)
- `enhancement_tone`: Applied enhancement style
- `enhanced`: Boolean indicating if GPT-enhanced
- `original_question`: Original question before enhancement
- `original_answer`: Original answer before enhancement
- `enhancement_cost`: Processing cost in USD
- `metadata`: Additional processing information

## 🎯 Quality Scores

- **0.8-1.0**: Excellent quality, ready for training
- **0.6-0.8**: Good quality, minor improvements possible
- **0.4-0.6**: Fair quality, may need review
- **0.0-0.4**: Poor quality, requires attention
"""
    
    def _generate_usage_examples(self) -> str:
        """Generate usage examples"""
        
        return """# Usage Examples

## 🐍 Python Examples

### Loading JSON Data
```python
import json

with open('training_data/final_training_data.json', 'r') as f:
    data = json.load(f)

for item in data:
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}")
    print(f"Quality: {item['quality_score']}")
    print("-" * 40)
```

### Loading JSONL Data
```python
import json

with open('training_data/final_training_data.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line)
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
```

### Using with Pandas
```python
import pandas as pd

# From CSV
df = pd.read_csv('training_data/final_training_data.csv')

# From Excel
df = pd.read_excel('training_data/final_training_data.xlsx')

# Filter by quality
high_quality = df[df['quality_score'] >= 0.8]
```

## 🤖 Machine Learning Examples

### Hugging Face Datasets
```python
from datasets import Dataset
import json

with open('training_data/final_training_data.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
```

### OpenAI Fine-tuning Format
```python
import json

def convert_to_openai_format(data):
    converted = []
    for item in data:
        converted.append({
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]}
            ]
        })
    return converted

with open('training_data/final_training_data.json', 'r') as f:
    data = json.load(f)

openai_format = convert_to_openai_format(data)

with open('openai_training_data.jsonl', 'w') as f:
    for item in openai_format:
        f.write(json.dumps(item) + '\\n')
```

## 📊 Analysis Examples

### Quality Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_data/final_training_data.csv')

# Quality distribution
quality_counts = df['quality_score'].value_counts(bins=4)
quality_counts.plot(kind='bar', title='Quality Score Distribution')
plt.show()

# Enhancement effectiveness
enhanced_df = df[df['enhanced'] == True]
print(f"Enhancement rate: {len(enhanced_df) / len(df) * 100:.1f}%")
```
"""
    
    def _generate_schema_documentation(self) -> str:
        """Generate data schema documentation"""
        
        return """# Data Schema Documentation

## 📋 Core Fields

### Required Fields
- **question** (string): The training question or prompt
- **answer** (string): The corresponding answer or response

### Quality Fields
- **quality_score** (float): Quality rating from 0.0 to 1.0
- **enhanced** (boolean): Whether content was GPT-enhanced

### Enhancement Fields
- **enhancement_tone** (string): Applied enhancement style
  - `universal_wisdom`: General spiritual wisdom
  - `advaita_vedanta`: Non-dual philosophy
  - `zen_buddhism`: Zen Buddhist approach
  - `sufi_mysticism`: Sufi mystical tradition
  - `christian_mysticism`: Christian contemplative tradition
  - `mindfulness_meditation`: Mindfulness-based approach

### Original Content Fields
- **original_question** (string): Question before enhancement
- **original_answer** (string): Answer before enhancement

### Metadata Fields
- **enhancement_cost** (float): Processing cost in USD
- **processing_timestamp** (string): When item was processed
- **source_file** (string): Original source file name
- **extraction_method** (string): How content was extracted

## 🎯 Quality Score Guidelines

### Excellent (0.8-1.0)
- Clear, well-structured questions and answers
- Appropriate spiritual context and depth
- Grammatically correct and coherent
- Ready for immediate training use

### Good (0.6-0.8)
- Generally well-structured content
- Minor grammatical or clarity issues
- Appropriate spiritual context
- May benefit from light editing

### Fair (0.4-0.6)
- Basic structure present
- Some clarity or context issues
- Requires review before training
- May need enhancement

### Poor (0.0-0.4)
- Structural or content issues
- Unclear or inappropriate content
- Requires significant revision
- Not recommended for training

## 🔄 Enhancement Process

1. **Content Analysis**: Original content is analyzed for structure and quality
2. **Tone Application**: Selected spiritual tone is applied via GPT
3. **Quality Assessment**: Enhanced content receives quality score
4. **Validation**: Content is validated for appropriateness and coherence
5. **Metadata Addition**: Processing metadata is added to each item

## 📊 Usage Recommendations

- **Training Data**: Use items with quality_score >= 0.6
- **High-Quality Training**: Use items with quality_score >= 0.8
- **Review Queue**: Items with quality_score < 0.6 need review
- **Cost Analysis**: Use enhancement_cost for budget planning
"""
    
    def _generate_html_comparison_report(self, comparison_data: Dict[str, Any]) -> str:
        """Generate HTML comparison report"""
        
        summary = comparison_data.get('summary', {})
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Content Enhancement Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #e9ecef; border-radius: 5px; }}
        .improvement {{ color: #28a745; font-weight: bold; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .table th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Content Enhancement Comparison Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>📈 Summary Metrics</h2>
    <div class="metric">
        <strong>Total Raw Items:</strong> {summary.get('total_raw_items', 0)}
    </div>
    <div class="metric">
        <strong>Enhanced Items:</strong> {summary.get('total_enhanced_items', 0)}
    </div>
    <div class="metric">
        <strong>Enhancement Rate:</strong> {summary.get('enhancement_rate', 0):.1f}%
    </div>
    <div class="metric">
        <strong>Average Improvement:</strong> <span class="improvement">{summary.get('average_improvement', 0):.1f}%</span>
    </div>
    
    <h2>📋 Detailed Comparisons</h2>
    <table class="table">
        <thead>
            <tr>
                <th>Question (Preview)</th>
                <th>Raw Words</th>
                <th>Enhanced Words</th>
                <th>Improvement</th>
                <th>Quality Score</th>
                <th>Enhancement Tone</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for comp in comparison_data.get('item_comparisons', [])[:50]:  # Limit to 50 items
            html += f"""
            <tr>
                <td>{comp.get('raw_question', '')}</td>
                <td>{comp.get('raw_words', 0)}</td>
                <td>{comp.get('enhanced_words', 0)}</td>
                <td class="improvement">{comp.get('improvement_percent', 0):+.1f}%</td>
                <td>{comp.get('quality_score', 0):.2f}</td>
                <td>{comp.get('enhancement_tone', 'unknown').replace('_', ' ').title()}</td>
            </tr>
"""
        
        html += """
        </tbody>
    </table>
</body>
</html>
"""
        return html
    
    def _generate_html_quality_dashboard(self, quality_data: Dict[str, Any]) -> str:
        """Generate HTML quality dashboard"""
        
        summary = quality_data.get('summary', {})
        distribution = quality_data.get('quality_distribution', {})
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Analysis Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .metric-large {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .quality-excellent {{ border-left-color: #28a745; }}
        .quality-good {{ border-left-color: #ffc107; }}
        .quality-fair {{ border-left-color: #fd7e14; }}
        .quality-poor {{ border-left-color: #dc3545; }}
    </style>
</head>
<body>
    <h1>🎯 Quality Analysis Dashboard</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="dashboard">
        <div class="card">
            <h3>📊 Overall Quality</h3>
            <div class="metric-large">{summary.get('average_quality_score', 0):.2f}</div>
            <p>Average Quality Score</p>
        </div>
        
        <div class="card">
            <h3>💰 Total Cost</h3>
            <div class="metric-large">${summary.get('total_enhancement_cost', 0):.4f}</div>
            <p>Enhancement Investment</p>
        </div>
        
        <div class="card quality-excellent">
            <h3>🟢 Excellent Quality</h3>
            <div class="metric-large">{distribution.get('excellent', 0)}</div>
            <p>Items (Score ≥ 0.8)</p>
        </div>
        
        <div class="card quality-good">
            <h3>🟡 Good Quality</h3>
            <div class="metric-large">{distribution.get('good', 0)}</div>
            <p>Items (Score 0.6-0.8)</p>
        </div>
        
        <div class="card quality-fair">
            <h3>🟠 Fair Quality</h3>
            <div class="metric-large">{distribution.get('fair', 0)}</div>
            <p>Items (Score 0.4-0.6)</p>
        </div>
        
        <div class="card quality-poor">
            <h3>🔴 Poor Quality</h3>
            <div class="metric-large">{distribution.get('poor', 0)}</div>
            <p>Items (Score < 0.4)</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _render_export_summary(self, final_content: List[Dict[str, Any]], 
                              export_formats: List[str], include_raw: bool, 
                              include_enhanced: bool, include_comparison: bool, 
                              include_stats: bool) -> None:
        """Render export summary information"""
        
        st.markdown("#### 📋 **Export Summary**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Content Statistics:**")
            st.write(f"• Final training items: {len(final_content)}")
            st.write(f"• Export formats: {', '.join(export_formats).upper()}")
            st.write(f"• Raw content included: {'✅' if include_raw else '❌'}")
            st.write(f"• Enhanced content included: {'✅' if include_enhanced else '❌'}")
        
        with col2:
            st.markdown("**📦 Package Contents:**")
            st.write(f"• Comparison report: {'✅' if include_comparison else '❌'}")
            st.write(f"• Statistics included: {'✅' if include_stats else '❌'}")
            st.write(f"• Documentation included: ✅")
            st.write(f"• Quality reports: ✅")
        
        # Calculate total words
        total_words = sum(len((item.get('question', '') + ' ' + item.get('answer', '')).split()) 
                         for item in final_content)
        
        st.info(f"📝 **Total training content:** {total_words:,} words across {len(final_content)} examples")

