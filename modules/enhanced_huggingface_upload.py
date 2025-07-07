"""
Enhanced Hugging Face Upload Module
===================================

Implements enhanced Hugging Face dataset upload functionality with comprehensive features.
Allows users to upload their processed spiritual content directly to Hugging Face Hub.

Features:
- Dataset validation and preparation
- Metadata generation
- Progress tracking
- Error handling and recovery
- Dataset versioning
- Privacy and licensing options
"""

import streamlit as st
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedHuggingFaceUploader:
    """Enhanced Hugging Face Dataset Upload - Optional Add-on"""
    
    def __init__(self):
        self.hf_available = self._check_huggingface_availability()
        self.supported_licenses = [
            'apache-2.0', 'mit', 'cc-by-4.0', 'cc-by-sa-4.0', 
            'cc-by-nc-4.0', 'cc0-1.0', 'other'
        ]
        self.supported_tasks = [
            'text-generation', 'question-answering', 'conversational',
            'text-classification', 'other'
        ]
    
    def _check_huggingface_availability(self) -> bool:
        """Check if Hugging Face libraries are available"""
        try:
            import datasets
            import huggingface_hub
            return True
        except ImportError:
            return False
    
    def render_upload_ui(self, final_content: Optional[List[Dict[str, Any]]] = None) -> None:
        """Render enhanced Hugging Face upload UI"""
        
        st.markdown("### 🤗 **Enhanced Hugging Face Dataset Upload**")
        
        if not self.hf_available:
            st.warning("""
            📦 **Hugging Face libraries not installed**
            
            To use this feature, install the required packages:
            ```bash
            pip install datasets huggingface_hub
            ```
            """)
            return
        
        if not final_content:
            st.info("📝 No content available for upload. Process some content first.")
            return
        
        # Dataset validation
        validation_results = self._validate_dataset(final_content)
        self._render_validation_results(validation_results)
        
        if not validation_results['is_valid']:
            st.error("❌ Dataset validation failed. Please fix the issues above before uploading.")
            return
        
        # Upload configuration
        self._render_upload_configuration(final_content)
    
    def _validate_dataset(self, content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate dataset for Hugging Face upload"""
        
        issues = []
        warnings = []
        
        # Check minimum content requirements
        if len(content) < 10:
            issues.append(f"Dataset too small: {len(content)} items (minimum 10 recommended)")
        
        # Check required fields
        required_fields = ['question', 'answer']
        for i, item in enumerate(content[:100]):  # Check first 100 items
            for field in required_fields:
                if not item.get(field) or not item[field].strip():
                    issues.append(f"Item {i+1}: Missing or empty '{field}' field")
        
        # Check content quality
        low_quality_count = sum(1 for item in content if item.get('quality_score', 0) < 0.4)
        if low_quality_count > len(content) * 0.2:  # More than 20% low quality
            warnings.append(f"High number of low-quality items: {low_quality_count} ({low_quality_count/len(content)*100:.1f}%)")
        
        # Check for duplicates
        questions = [item.get('question', '') for item in content]
        unique_questions = set(questions)
        if len(unique_questions) < len(questions):
            duplicate_count = len(questions) - len(unique_questions)
            warnings.append(f"Found {duplicate_count} duplicate questions")
        
        # Check average content length
        avg_length = sum(len((item.get('question', '') + ' ' + item.get('answer', '')).split()) 
                        for item in content) / len(content)
        if avg_length < 10:
            warnings.append(f"Average content length is low: {avg_length:.1f} words per item")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'stats': {
                'total_items': len(content),
                'average_length': avg_length,
                'quality_distribution': self._get_quality_distribution(content),
                'unique_questions': len(unique_questions)
            }
        }
    
    def _render_validation_results(self, validation: Dict[str, Any]) -> None:
        """Render dataset validation results"""
        
        st.markdown("#### 🔍 **Dataset Validation**")
        
        if validation['is_valid']:
            st.success("✅ **Dataset validation passed!** Ready for upload.")
        else:
            st.error("❌ **Dataset validation failed.** Please fix the issues below:")
            for issue in validation['issues']:
                st.error(f"• {issue}")
        
        if validation['warnings']:
            st.warning("⚠️ **Validation warnings:**")
            for warning in validation['warnings']:
                st.warning(f"• {warning}")
        
        # Show validation stats
        stats = validation['stats']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Items", stats['total_items'])
        
        with col2:
            st.metric("📝 Avg Length", f"{stats['average_length']:.1f} words")
        
        with col3:
            st.metric("🔄 Unique Questions", stats['unique_questions'])
        
        with col4:
            quality_dist = stats['quality_distribution']
            high_quality = quality_dist.get('excellent', 0) + quality_dist.get('good', 0)
            st.metric("🎯 High Quality", f"{high_quality} items")
    
    def _render_upload_configuration(self, content: List[Dict[str, Any]]) -> None:
        """Render upload configuration UI"""
        
        st.markdown("#### ⚙️ **Upload Configuration**")
        
        # Authentication
        with st.expander("🔐 **Authentication**", expanded=True):
            hf_token = st.text_input(
                "Hugging Face Token",
                type="password",
                help="Your Hugging Face access token (get it from https://huggingface.co/settings/tokens)"
            )
            
            if not hf_token:
                st.info("💡 You need a Hugging Face token to upload datasets. Get one from your HF settings.")
                return
        
        # Dataset information
        with st.expander("📋 **Dataset Information**", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                dataset_name = st.text_input(
                    "Dataset Name",
                    value="spiritual-qa-dataset",
                    help="Name for your dataset (lowercase, hyphens allowed)"
                )
                
                dataset_description = st.text_area(
                    "Dataset Description",
                    value="AI training dataset for spiritual Q&A conversations",
                    help="Brief description of your dataset"
                )
            
            with col2:
                license_type = st.selectbox(
                    "License",
                    options=self.supported_licenses,
                    index=0,
                    help="Choose an appropriate license for your dataset"
                )
                
                task_type = st.selectbox(
                    "Task Type",
                    options=self.supported_tasks,
                    index=0,
                    help="Primary task this dataset is designed for"
                )
        
        # Privacy and sharing
        with st.expander("🔒 **Privacy & Sharing**", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                is_private = st.checkbox(
                    "Private Dataset",
                    value=True,
                    help="Keep dataset private (only you can access)"
                )
                
                include_metadata = st.checkbox(
                    "Include Processing Metadata",
                    value=True,
                    help="Include quality scores and enhancement information"
                )
            
            with col2:
                create_model_card = st.checkbox(
                    "Generate Model Card",
                    value=True,
                    help="Create comprehensive documentation"
                )
                
                include_examples = st.checkbox(
                    "Include Example Samples",
                    value=True,
                    help="Add sample data to model card"
                )
        
        # Data preparation options
        with st.expander("🛠️ **Data Preparation**", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                min_quality_score = st.slider(
                    "Minimum Quality Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.1,
                    help="Filter out items below this quality threshold"
                )
                
                remove_duplicates = st.checkbox(
                    "Remove Duplicate Questions",
                    value=True,
                    help="Remove items with identical questions"
                )
            
            with col2:
                max_items = st.number_input(
                    "Maximum Items",
                    min_value=10,
                    max_value=len(content),
                    value=len(content),
                    help="Limit the number of items to upload"
                )
                
                shuffle_data = st.checkbox(
                    "Shuffle Data",
                    value=True,
                    help="Randomly shuffle items before upload"
                )
        
        # Upload button and process
        st.markdown("#### 🚀 **Upload Dataset**")
        
        if st.button("🤗 **Upload to Hugging Face**", type="primary"):
            try:
                # Prepare dataset
                prepared_data = self._prepare_dataset(
                    content, min_quality_score, remove_duplicates, 
                    max_items, shuffle_data, include_metadata
                )
                
                # Upload dataset
                with st.spinner("🚀 Uploading dataset to Hugging Face..."):
                    upload_result = self._upload_dataset(
                        prepared_data, dataset_name, dataset_description,
                        license_type, task_type, is_private, hf_token,
                        create_model_card, include_examples
                    )
                
                if upload_result['success']:
                    st.success(f"✅ **Dataset uploaded successfully!**")
                    st.info(f"🔗 **Dataset URL:** {upload_result['url']}")
                    
                    # Show upload summary
                    self._render_upload_summary(upload_result)
                else:
                    st.error(f"❌ **Upload failed:** {upload_result['error']}")
                    
            except Exception as e:
                st.error(f"❌ **Upload error:** {str(e)}")
                logger.error(f"HuggingFace upload error: {e}")
    
    def _prepare_dataset(self, content: List[Dict[str, Any]], min_quality: float,
                        remove_duplicates: bool, max_items: int, shuffle: bool,
                        include_metadata: bool) -> List[Dict[str, Any]]:
        """Prepare dataset for upload"""
        
        import random
        
        # Filter by quality
        filtered_content = [
            item for item in content 
            if item.get('quality_score', 0) >= min_quality
        ]
        
        # Remove duplicates
        if remove_duplicates:
            seen_questions = set()
            unique_content = []
            for item in filtered_content:
                question = item.get('question', '').strip().lower()
                if question not in seen_questions:
                    seen_questions.add(question)
                    unique_content.append(item)
            filtered_content = unique_content
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(filtered_content)
        
        # Limit items
        filtered_content = filtered_content[:max_items]
        
        # Prepare final format
        prepared_data = []
        for item in filtered_content:
            prepared_item = {
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
            }
            
            if include_metadata:
                prepared_item.update({
                    'quality_score': item.get('quality_score', 0),
                    'enhanced': item.get('enhanced', False),
                    'enhancement_tone': item.get('enhancement_tone', 'none'),
                })
                
                if item.get('enhanced', False):
                    prepared_item.update({
                        'original_question': item.get('original_question', ''),
                        'original_answer': item.get('original_answer', ''),
                    })
            
            prepared_data.append(prepared_item)
        
        return prepared_data
    
    def _upload_dataset(self, data: List[Dict[str, Any]], name: str, description: str,
                       license_type: str, task_type: str, is_private: bool, token: str,
                       create_model_card: bool, include_examples: bool) -> Dict[str, Any]:
        """Upload dataset to Hugging Face"""
        
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi, DatasetCard
            
            # Create dataset
            dataset = Dataset.from_list(data)
            
            # Upload dataset
            dataset.push_to_hub(
                name,
                token=token,
                private=is_private
            )
            
            # Create model card if requested
            if create_model_card:
                self._create_model_card(
                    name, description, data, license_type, 
                    task_type, include_examples, token
                )
            
            # Get dataset URL
            username = self._get_username_from_token(token)
            dataset_url = f"https://huggingface.co/datasets/{username}/{name}"
            
            return {
                'success': True,
                'url': dataset_url,
                'items_uploaded': len(data),
                'dataset_name': name
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_model_card(self, name: str, description: str, data: List[Dict[str, Any]],
                          license_type: str, task_type: str, include_examples: bool, token: str) -> None:
        """Create comprehensive model card"""
        
        from huggingface_hub import DatasetCard
        
        # Generate statistics
        stats = self._generate_dataset_stats(data)
        
        # Create model card content
        card_content = f"""---
license: {license_type}
task_categories:
- {task_type}
language:
- en
tags:
- spiritual
- qa
- consciousness
- training-data
size_categories:
- {self._get_size_category(len(data))}
---

# {name.replace('-', ' ').title()}

## Dataset Description

{description}

This dataset contains {len(data)} high-quality question-answer pairs focused on spiritual and consciousness-related topics. The data has been processed and enhanced using AI to ensure consistency and quality.

## Dataset Statistics

- **Total Items**: {stats['total_items']:,}
- **Average Question Length**: {stats['avg_question_length']:.1f} words
- **Average Answer Length**: {stats['avg_answer_length']:.1f} words
- **Quality Score Range**: {stats['min_quality']:.2f} - {stats['max_quality']:.2f}
- **Average Quality Score**: {stats['avg_quality']:.2f}

## Data Fields

- `question`: The training question or prompt
- `answer`: The corresponding answer or response
- `quality_score`: Quality rating from 0.0 to 1.0 (if metadata included)
- `enhanced`: Whether content was AI-enhanced (if metadata included)
- `enhancement_tone`: Applied enhancement style (if metadata included)

## Quality Distribution

- **Excellent (≥0.8)**: {stats['quality_dist']['excellent']} items
- **Good (0.6-0.8)**: {stats['quality_dist']['good']} items
- **Fair (0.4-0.6)**: {stats['quality_dist']['fair']} items
- **Poor (<0.4)**: {stats['quality_dist']['poor']} items

## Usage

This dataset is suitable for:
- Fine-tuning conversational AI models
- Training question-answering systems
- Spiritual and consciousness-focused applications
- Research in AI ethics and wisdom traditions

```python
from datasets import load_dataset

dataset = load_dataset("{name}")
```

## Ethical Considerations

This dataset focuses on spiritual and consciousness-related content. Users should:
- Respect the wisdom traditions represented
- Use the content responsibly and ethically
- Consider the cultural context of spiritual teachings
- Avoid making absolute claims about spiritual matters

## License

This dataset is released under the {license_type.upper()} license.

## Citation

If you use this dataset, please cite:

```
@dataset{{{name.replace('-', '_')},
  title={{{name.replace('-', ' ').title()}}},
  author={{Universal AI Training Data Creator}},
  year={{{datetime.now().year}}},
  url={{https://huggingface.co/datasets/{name}}}
}}
```

---

*Generated by Universal AI Training Data Creator v2.0*
"""

        # Add examples if requested
        if include_examples and data:
            card_content += "\n## Examples\n\n"
            for i, item in enumerate(data[:3]):  # Show first 3 examples
                card_content += f"### Example {i+1}\n\n"
                card_content += f"**Question:** {item['question']}\n\n"
                card_content += f"**Answer:** {item['answer']}\n\n"
                if 'quality_score' in item:
                    card_content += f"**Quality Score:** {item['quality_score']:.2f}\n\n"
                card_content += "---\n\n"
        
        # Create and push card
        card = DatasetCard(card_content)
        card.push_to_hub(name, token=token)
    
    def _generate_dataset_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics"""
        
        question_lengths = [len(item['question'].split()) for item in data]
        answer_lengths = [len(item['answer'].split()) for item in data]
        quality_scores = [item.get('quality_score', 0) for item in data]
        
        # Quality distribution
        excellent = sum(1 for score in quality_scores if score >= 0.8)
        good = sum(1 for score in quality_scores if 0.6 <= score < 0.8)
        fair = sum(1 for score in quality_scores if 0.4 <= score < 0.6)
        poor = sum(1 for score in quality_scores if score < 0.4)
        
        return {
            'total_items': len(data),
            'avg_question_length': sum(question_lengths) / len(question_lengths),
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
            'min_quality': min(quality_scores) if quality_scores else 0,
            'max_quality': max(quality_scores) if quality_scores else 0,
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'quality_dist': {
                'excellent': excellent,
                'good': good,
                'fair': fair,
                'poor': poor
            }
        }
    
    def _get_quality_distribution(self, content: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get quality score distribution"""
        
        quality_scores = [item.get('quality_score', 0) for item in content]
        
        return {
            'excellent': sum(1 for score in quality_scores if score >= 0.8),
            'good': sum(1 for score in quality_scores if 0.6 <= score < 0.8),
            'fair': sum(1 for score in quality_scores if 0.4 <= score < 0.6),
            'poor': sum(1 for score in quality_scores if score < 0.4)
        }
    
    def _get_size_category(self, size: int) -> str:
        """Get Hugging Face size category"""
        
        if size < 1000:
            return "n<1K"
        elif size < 10000:
            return "1K<n<10K"
        elif size < 100000:
            return "10K<n<100K"
        elif size < 1000000:
            return "100K<n<1M"
        else:
            return "n>1M"
    
    def _get_username_from_token(self, token: str) -> str:
        """Get username from Hugging Face token"""
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user_info = api.whoami(token=token)
            return user_info['name']
        except:
            return "user"  # Fallback
    
    def _render_upload_summary(self, result: Dict[str, Any]) -> None:
        """Render upload summary"""
        
        st.markdown("#### 📊 **Upload Summary**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📤 Items Uploaded", result['items_uploaded'])
            st.metric("📋 Dataset Name", result['dataset_name'])
        
        with col2:
            st.info(f"🔗 **Access your dataset:** {result['url']}")
            st.success("✅ **Model card created with comprehensive documentation**")
        
        st.markdown("#### 🎯 **Next Steps**")
        st.markdown("""
        1. **Review your dataset** on Hugging Face
        2. **Test the data** with your training pipeline
        3. **Share responsibly** if making public
        4. **Cite appropriately** when using in research
        """)
        
        st.markdown("#### 💡 **Usage Example**")
        st.code(f"""
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("{result['dataset_name']}")

# Access the data
for item in dataset['train']:
    print(f"Q: {{item['question']}}")
    print(f"A: {{item['answer']}}")
    print(f"Quality: {{item.get('quality_score', 'N/A')}}")
    print("-" * 40)
        """, language="python")

