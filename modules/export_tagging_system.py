"""
Export Tagging System
====================

Tag export purpose (e.g., "Instruction", "QA", "Chat", "Narrative")
to help organize and mix datasets later.

Features:
- Purpose-based tagging system
- Automatic tag suggestions
- Custom tag creation
- Tag-based filtering and organization
- Export with embedded tags
- Dataset mixing recommendations
"""

import streamlit as st
from typing import List, Dict, Any, Set, Optional
from datetime import datetime
import json
from modules.logger import get_logger, log_event, log_user_action
import pandas as pd

class ExportTaggingSystem:
    """
    Comprehensive export tagging system for dataset organization
    
    Features:
    - Predefined purpose tags
    - Custom tag creation
    - Automatic tag suggestions
    - Tag-based filtering
    - Export with embedded metadata
    - Dataset mixing recommendations
    """
    
    def __init__(self):
        self.logger = get_logger("export_tagging")
        
        # Initialize session state
        if 'selected_tags' not in st.session_state:
            st.session_state['selected_tags'] = set()
        
        if 'custom_tags' not in st.session_state:
            st.session_state['custom_tags'] = set()
        
        if 'tag_descriptions' not in st.session_state:
            st.session_state['tag_descriptions'] = {}
        
        # Predefined tag categories
        self.predefined_tags = {
            'purpose': {
                'instruction': {
                    'name': 'Instruction Following',
                    'description': 'Data for training models to follow instructions',
                    'examples': ['Do X', 'Explain Y', 'Create Z'],
                    'use_cases': ['General instruction following', 'Task completion', 'Command execution']
                },
                'qa': {
                    'name': 'Question & Answer',
                    'description': 'Question-answer pairs for factual knowledge',
                    'examples': ['What is X?', 'How does Y work?', 'When did Z happen?'],
                    'use_cases': ['Knowledge retrieval', 'Factual QA', 'Information lookup']
                },
                'chat': {
                    'name': 'Conversational',
                    'description': 'Natural conversation and dialogue',
                    'examples': ['Casual chat', 'Multi-turn dialogue', 'Social interaction'],
                    'use_cases': ['Chatbots', 'Virtual assistants', 'Social AI']
                },
                'narrative': {
                    'name': 'Narrative/Storytelling',
                    'description': 'Stories, creative writing, and narrative content',
                    'examples': ['Short stories', 'Creative writing', 'Plot development'],
                    'use_cases': ['Creative AI', 'Story generation', 'Content creation']
                },
                'reasoning': {
                    'name': 'Reasoning & Logic',
                    'description': 'Logical reasoning and problem-solving',
                    'examples': ['Math problems', 'Logic puzzles', 'Analytical thinking'],
                    'use_cases': ['Problem solving', 'Mathematical reasoning', 'Logical analysis']
                },
                'code': {
                    'name': 'Code Generation',
                    'description': 'Programming and code-related tasks',
                    'examples': ['Write function', 'Debug code', 'Explain algorithm'],
                    'use_cases': ['Code assistants', 'Programming help', 'Technical documentation']
                }
            },
            'domain': {
                'general': {
                    'name': 'General Knowledge',
                    'description': 'Broad, general-purpose content',
                    'examples': ['Common facts', 'General advice', 'Everyday topics'],
                    'use_cases': ['General-purpose models', 'Broad knowledge base']
                },
                'technical': {
                    'name': 'Technical/Scientific',
                    'description': 'Technical, scientific, or specialized content',
                    'examples': ['Scientific concepts', 'Technical procedures', 'Expert knowledge'],
                    'use_cases': ['Specialized models', 'Expert systems', 'Technical assistance']
                },
                'creative': {
                    'name': 'Creative/Artistic',
                    'description': 'Creative, artistic, or imaginative content',
                    'examples': ['Art descriptions', 'Creative writing', 'Design concepts'],
                    'use_cases': ['Creative AI', 'Art generation', 'Design assistance']
                },
                'educational': {
                    'name': 'Educational',
                    'description': 'Teaching, learning, and educational content',
                    'examples': ['Tutorials', 'Explanations', 'Learning materials'],
                    'use_cases': ['Educational AI', 'Tutoring systems', 'Learning platforms']
                }
            },
            'style': {
                'formal': {
                    'name': 'Formal',
                    'description': 'Professional, academic, or formal tone',
                    'examples': ['Business communication', 'Academic writing', 'Official documents'],
                    'use_cases': ['Professional AI', 'Business applications', 'Academic tools']
                },
                'casual': {
                    'name': 'Casual',
                    'description': 'Informal, conversational tone',
                    'examples': ['Friendly chat', 'Casual advice', 'Informal explanations'],
                    'use_cases': ['Casual chatbots', 'Friendly assistants', 'Social AI']
                },
                'technical': {
                    'name': 'Technical',
                    'description': 'Precise, technical language',
                    'examples': ['Technical documentation', 'Specifications', 'Procedures'],
                    'use_cases': ['Technical assistants', 'Documentation tools', 'Expert systems']
                }
            },
            'complexity': {
                'beginner': {
                    'name': 'Beginner',
                    'description': 'Simple, easy-to-understand content',
                    'examples': ['Basic concepts', 'Simple explanations', 'Introductory material'],
                    'use_cases': ['Educational AI', 'Beginner-friendly tools', 'Simplified explanations']
                },
                'intermediate': {
                    'name': 'Intermediate',
                    'description': 'Moderate complexity content',
                    'examples': ['Detailed explanations', 'Multi-step processes', 'Moderate complexity'],
                    'use_cases': ['General-purpose AI', 'Balanced complexity', 'Practical applications']
                },
                'advanced': {
                    'name': 'Advanced',
                    'description': 'Complex, expert-level content',
                    'examples': ['Expert knowledge', 'Complex analysis', 'Advanced concepts'],
                    'use_cases': ['Expert systems', 'Advanced AI', 'Specialized applications']
                }
            }
        }
    
    def render_tagging_interface(self, content_data: List[Dict]):
        """Render the complete export tagging interface"""
        
        st.subheader("🏷️ Export Tagging System")
        
        if not content_data:
            st.warning("⚠️ No content available for tagging")
            return
        
        # Tagging controls
        self.render_tagging_controls()
        
        # Tag suggestions
        self.render_tag_suggestions(content_data)
        
        # Selected tags overview
        self.render_selected_tags_overview()
        
        # Content preview with tags
        self.render_tagged_content_preview(content_data)
        
        # Export options with tags
        self.render_tagged_export_options(content_data)
    
    def render_tagging_controls(self):
        """Render tag selection and management controls"""
        
        st.markdown("**🏷️ Tag Selection:**")
        
        # Tag category tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Purpose", "🌐 Domain", "✍️ Style", "📊 Complexity", "➕ Custom"
        ])
        
        with tab1:
            self.render_tag_category('purpose', 'Purpose Tags')
        
        with tab2:
            self.render_tag_category('domain', 'Domain Tags')
        
        with tab3:
            self.render_tag_category('style', 'Style Tags')
        
        with tab4:
            self.render_tag_category('complexity', 'Complexity Tags')
        
        with tab5:
            self.render_custom_tags_interface()
    
    def render_tag_category(self, category: str, category_name: str):
        """Render tags for a specific category"""
        
        st.markdown(f"**{category_name}:**")
        
        tags = self.predefined_tags[category]
        
        for tag_key, tag_info in tags.items():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                is_selected = tag_key in st.session_state['selected_tags']
                
                if st.checkbox(
                    tag_info['name'],
                    value=is_selected,
                    key=f"tag_{category}_{tag_key}"
                ):
                    st.session_state['selected_tags'].add(tag_key)
                    log_user_action("tag_selected", {"tag": tag_key, "category": category})
                else:
                    st.session_state['selected_tags'].discard(tag_key)
            
            with col2:
                with st.expander(f"ℹ️ About {tag_info['name']}", expanded=False):
                    st.write(f"**Description:** {tag_info['description']}")
                    st.write(f"**Examples:** {', '.join(tag_info['examples'])}")
                    st.write(f"**Use Cases:** {', '.join(tag_info['use_cases'])}")
    
    def render_custom_tags_interface(self):
        """Render custom tag creation interface"""
        
        st.markdown("**➕ Custom Tags:**")
        
        # Add new custom tag
        col1, col2 = st.columns([2, 1])
        
        with col1:
            new_tag = st.text_input(
                "Create Custom Tag",
                placeholder="Enter custom tag name",
                help="Create your own tags for specific use cases"
            )
        
        with col2:
            if st.button("➕ Add Tag") and new_tag:
                if new_tag.lower() not in st.session_state['custom_tags']:
                    st.session_state['custom_tags'].add(new_tag.lower())
                    st.session_state['selected_tags'].add(new_tag.lower())
                    log_user_action("custom_tag_created", {"tag": new_tag.lower()})
                    st.success(f"Added custom tag: {new_tag}")
                    st.rerun()
                else:
                    st.warning("Tag already exists")
        
        # Display existing custom tags
        if st.session_state['custom_tags']:
            st.markdown("**Your Custom Tags:**")
            
            for custom_tag in sorted(st.session_state['custom_tags']):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    is_selected = custom_tag in st.session_state['selected_tags']
                    
                    if st.checkbox(
                        custom_tag.title(),
                        value=is_selected,
                        key=f"custom_tag_{custom_tag}"
                    ):
                        st.session_state['selected_tags'].add(custom_tag)
                    else:
                        st.session_state['selected_tags'].discard(custom_tag)
                
                with col2:
                    if st.button("📝", key=f"edit_{custom_tag}", help="Edit description"):
                        st.session_state[f'edit_desc_{custom_tag}'] = True
                        st.rerun()
                
                with col3:
                    if st.button("🗑️", key=f"delete_{custom_tag}", help="Delete tag"):
                        st.session_state['custom_tags'].discard(custom_tag)
                        st.session_state['selected_tags'].discard(custom_tag)
                        log_user_action("custom_tag_deleted", {"tag": custom_tag})
                        st.rerun()
                
                # Edit description interface
                if st.session_state.get(f'edit_desc_{custom_tag}', False):
                    description = st.text_area(
                        f"Description for '{custom_tag}'",
                        value=st.session_state['tag_descriptions'].get(custom_tag, ''),
                        key=f"desc_input_{custom_tag}"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save", key=f"save_desc_{custom_tag}"):
                            st.session_state['tag_descriptions'][custom_tag] = description
                            st.session_state[f'edit_desc_{custom_tag}'] = False
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_desc_{custom_tag}"):
                            st.session_state[f'edit_desc_{custom_tag}'] = False
                            st.rerun()
    
    def render_tag_suggestions(self, content_data: List[Dict]):
        """Render automatic tag suggestions based on content analysis"""
        
        st.markdown("---")
        st.markdown("**🤖 Suggested Tags:**")
        
        # Analyze content to suggest tags
        suggested_tags = self.analyze_content_for_tags(content_data)
        
        if suggested_tags:
            st.markdown("Based on your content analysis, we suggest these tags:")
            
            cols = st.columns(min(len(suggested_tags), 4))
            
            for i, (tag, confidence) in enumerate(suggested_tags.items()):
                with cols[i % len(cols)]:
                    if st.button(
                        f"➕ {tag.title()} ({confidence:.0%})",
                        key=f"suggest_{tag}",
                        help=f"Confidence: {confidence:.1%}"
                    ):
                        st.session_state['selected_tags'].add(tag)
                        log_user_action("suggested_tag_accepted", {"tag": tag, "confidence": confidence})
                        st.rerun()
        else:
            st.info("No automatic suggestions available. Please select tags manually.")
    
    def analyze_content_for_tags(self, content_data: List[Dict]) -> Dict[str, float]:
        """Analyze content to suggest appropriate tags"""
        
        suggestions = {}
        
        if not content_data:
            return suggestions
        
        # Sample content for analysis
        sample_size = min(10, len(content_data))
        sample_content = content_data[:sample_size]
        
        # Analyze content patterns
        total_text = ""
        question_count = 0
        instruction_count = 0
        code_count = 0
        
        for item in sample_content:
            # Extract text content
            text_content = self.extract_text_content(item)
            total_text += text_content.lower() + " "
            
            # Count patterns
            if any(word in text_content.lower() for word in ['what', 'how', 'when', 'where', 'why', 'which']):
                question_count += 1
            
            if any(word in text_content.lower() for word in ['create', 'write', 'generate', 'make', 'build']):
                instruction_count += 1
            
            if any(word in text_content.lower() for word in ['function', 'code', 'programming', 'algorithm', 'script']):
                code_count += 1
        
        # Calculate suggestions based on patterns
        if question_count / sample_size > 0.5:
            suggestions['qa'] = question_count / sample_size
        
        if instruction_count / sample_size > 0.3:
            suggestions['instruction'] = instruction_count / sample_size
        
        if code_count / sample_size > 0.2:
            suggestions['code'] = code_count / sample_size
        
        # Analyze complexity
        avg_length = len(total_text) / sample_size if sample_size > 0 else 0
        
        if avg_length < 100:
            suggestions['beginner'] = 0.7
        elif avg_length > 300:
            suggestions['advanced'] = 0.6
        else:
            suggestions['intermediate'] = 0.8
        
        # Analyze style
        formal_indicators = ['therefore', 'furthermore', 'consequently', 'however', 'moreover']
        casual_indicators = ['hey', 'cool', 'awesome', 'yeah', 'okay']
        
        formal_count = sum(1 for word in formal_indicators if word in total_text)
        casual_count = sum(1 for word in casual_indicators if word in total_text)
        
        if formal_count > casual_count:
            suggestions['formal'] = 0.6
        elif casual_count > formal_count:
            suggestions['casual'] = 0.6
        
        return suggestions
    
    def render_selected_tags_overview(self):
        """Render overview of selected tags"""
        
        st.markdown("---")
        st.markdown("**📋 Selected Tags:**")
        
        if st.session_state['selected_tags']:
            # Group tags by category
            categorized_tags = self.categorize_selected_tags()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                for category, tags in categorized_tags.items():
                    if tags:
                        st.markdown(f"**{category.title()}:** {', '.join(tag.title() for tag in tags)}")
            
            with col2:
                if st.button("🗑️ Clear All Tags"):
                    st.session_state['selected_tags'].clear()
                    log_user_action("all_tags_cleared")
                    st.rerun()
            
            # Tag combination analysis
            self.render_tag_combination_analysis()
        else:
            st.info("No tags selected. Please select tags to categorize your dataset.")
    
    def categorize_selected_tags(self) -> Dict[str, List[str]]:
        """Categorize selected tags by their type"""
        
        categorized = {
            'purpose': [],
            'domain': [],
            'style': [],
            'complexity': [],
            'custom': []
        }
        
        for tag in st.session_state['selected_tags']:
            # Check predefined categories
            found = False
            for category, tags in self.predefined_tags.items():
                if tag in tags:
                    categorized[category].append(tag)
                    found = True
                    break
            
            # If not found in predefined, it's custom
            if not found:
                categorized['custom'].append(tag)
        
        return categorized
    
    def render_tag_combination_analysis(self):
        """Render analysis of tag combinations"""
        
        with st.expander("📊 Tag Combination Analysis"):
            categorized_tags = self.categorize_selected_tags()
            
            # Check for good combinations
            recommendations = []
            warnings = []
            
            # Purpose + Domain combination
            if categorized_tags['purpose'] and categorized_tags['domain']:
                recommendations.append("✅ Good: Purpose + Domain tags provide clear dataset categorization")
            elif categorized_tags['purpose'] and not categorized_tags['domain']:
                recommendations.append("💡 Consider adding a Domain tag for better organization")
            
            # Style + Complexity combination
            if categorized_tags['style'] and categorized_tags['complexity']:
                recommendations.append("✅ Good: Style + Complexity tags help with model training specificity")
            
            # Check for conflicts
            if 'formal' in st.session_state['selected_tags'] and 'casual' in st.session_state['selected_tags']:
                warnings.append("⚠️ Conflicting style tags: 'formal' and 'casual'")
            
            if 'beginner' in st.session_state['selected_tags'] and 'advanced' in st.session_state['selected_tags']:
                warnings.append("⚠️ Conflicting complexity tags: 'beginner' and 'advanced'")
            
            # Display recommendations
            for rec in recommendations:
                st.write(rec)
            
            for warning in warnings:
                st.write(warning)
            
            if not recommendations and not warnings:
                st.info("Tag combination looks good!")
    
    def render_tagged_content_preview(self, content_data: List[Dict]):
        """Render preview of content with tags applied"""
        
        st.markdown("---")
        st.markdown("**👀 Tagged Content Preview:**")
        
        if not st.session_state['selected_tags']:
            st.info("Select tags to see how they'll be applied to your content")
            return
        
        # Show preview of first few items
        preview_count = min(3, len(content_data))
        
        for i in range(preview_count):
            item = content_data[i]
            tagged_item = self.apply_tags_to_item(item)
            
            with st.expander(f"📄 Preview Item {i + 1}"):
                st.json(tagged_item)
    
    def render_tagged_export_options(self, content_data: List[Dict]):
        """Render export options with tag embedding"""
        
        st.markdown("---")
        st.markdown("**📦 Tagged Export Options:**")
        
        if not st.session_state['selected_tags']:
            st.warning("⚠️ Please select tags before exporting")
            return
        
        # Export settings
        col1, col2 = st.columns(2)
        
        with col1:
            embed_tags_in_content = st.checkbox(
                "Embed Tags in Content",
                value=True,
                help="Include tags as metadata in each item"
            )
            
            create_tag_summary = st.checkbox(
                "Create Tag Summary File",
                value=True,
                help="Generate separate file with tag information"
            )
        
        with col2:
            include_tag_descriptions = st.checkbox(
                "Include Tag Descriptions",
                value=False,
                help="Include detailed tag descriptions in metadata"
            )
            
            tag_format = st.selectbox(
                "Tag Format",
                ["list", "string", "object"],
                help="How to format tags in the exported data"
            )
        
        # Export buttons
        st.markdown("**Export Options:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Tagged JSONL"):
                self.export_tagged_data(
                    content_data, 'jsonl', embed_tags_in_content, 
                    create_tag_summary, include_tag_descriptions, tag_format
                )
        
        with col2:
            if st.button("📥 Export Tagged JSON"):
                self.export_tagged_data(
                    content_data, 'json', embed_tags_in_content, 
                    create_tag_summary, include_tag_descriptions, tag_format
                )
        
        with col3:
            if st.button("📊 Export Tag Report"):
                self.export_tag_report(content_data)
    
    def apply_tags_to_item(self, item: Dict) -> Dict:
        """Apply selected tags to a single content item"""
        
        tagged_item = item.copy()
        
        # Add tags metadata
        tagged_item['_metadata'] = tagged_item.get('_metadata', {})
        tagged_item['_metadata']['tags'] = list(st.session_state['selected_tags'])
        tagged_item['_metadata']['tagged_at'] = datetime.now().isoformat()
        
        # Add tag descriptions if available
        tag_descriptions = {}
        for tag in st.session_state['selected_tags']:
            if tag in st.session_state['tag_descriptions']:
                tag_descriptions[tag] = st.session_state['tag_descriptions'][tag]
            else:
                # Look for predefined descriptions
                for category, tags in self.predefined_tags.items():
                    if tag in tags:
                        tag_descriptions[tag] = tags[tag]['description']
                        break
        
        if tag_descriptions:
            tagged_item['_metadata']['tag_descriptions'] = tag_descriptions
        
        return tagged_item
    
    def export_tagged_data(self, content_data: List[Dict], format_type: str, 
                          embed_tags: bool, create_summary: bool, 
                          include_descriptions: bool, tag_format: str):
        """Export data with tags embedded"""
        
        try:
            # Apply tags to all content
            tagged_data = []
            for item in content_data:
                tagged_item = self.apply_tags_to_item(item)
                
                if not embed_tags:
                    # Remove metadata if not embedding
                    tagged_item.pop('_metadata', None)
                
                tagged_data.append(tagged_item)
            
            # Generate export data
            if format_type == 'jsonl':
                export_data = '\n'.join(json.dumps(item, ensure_ascii=False) for item in tagged_data)
                mime_type = 'application/jsonl'
                file_extension = 'jsonl'
            else:  # json
                export_data = json.dumps(tagged_data, indent=2, ensure_ascii=False)
                mime_type = 'application/json'
                file_extension = 'json'
            
            # Generate filename with tags
            tag_string = '_'.join(sorted(st.session_state['selected_tags']))[:50]  # Limit length
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tagged_data_{tag_string}_{timestamp}.{file_extension}"
            
            # Download button
            st.download_button(
                f"📥 Download Tagged {format_type.upper()}",
                data=export_data,
                file_name=filename,
                mime=mime_type
            )
            
            # Create tag summary if requested
            if create_summary:
                self.create_tag_summary_download(content_data)
            
            # Log export
            log_event("tagged_export", {
                "format": format_type,
                "item_count": len(tagged_data),
                "tags": list(st.session_state['selected_tags']),
                "embed_tags": embed_tags
            }, "export_tagging")
            
            st.success(f"✅ {len(tagged_data)} items exported with tags!")
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            self.logger.error(f"Tagged export failed: {str(e)}")
    
    def create_tag_summary_download(self, content_data: List[Dict]):
        """Create downloadable tag summary file"""
        
        summary = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_items': len(content_data),
                'tags_applied': list(st.session_state['selected_tags'])
            },
            'tag_details': {},
            'tag_statistics': self.calculate_tag_statistics()
        }
        
        # Add tag details
        for tag in st.session_state['selected_tags']:
            tag_info = {'name': tag.title()}
            
            # Add description
            if tag in st.session_state['tag_descriptions']:
                tag_info['description'] = st.session_state['tag_descriptions'][tag]
            else:
                # Look for predefined descriptions
                for category, tags in self.predefined_tags.items():
                    if tag in tags:
                        tag_info.update(tags[tag])
                        tag_info['category'] = category
                        break
            
            summary['tag_details'][tag] = tag_info
        
        # Create download
        summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
        
        st.download_button(
            "📋 Download Tag Summary",
            data=summary_json,
            file_name=f"tag_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def export_tag_report(self, content_data: List[Dict]):
        """Export comprehensive tag report"""
        
        try:
            # Generate report data
            report_data = {
                'dataset_info': {
                    'total_items': len(content_data),
                    'export_timestamp': datetime.now().isoformat(),
                    'tags_applied': list(st.session_state['selected_tags'])
                },
                'tag_analysis': self.calculate_tag_statistics(),
                'recommendations': self.generate_tag_recommendations(),
                'mixing_suggestions': self.generate_mixing_suggestions()
            }
            
            # Create downloadable report
            report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📊 Download Tag Report",
                data=report_json,
                file_name=f"tag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Display report summary
            with st.expander("📊 Tag Report Summary"):
                st.json(report_data)
        
        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")
            self.logger.error(f"Tag report generation failed: {str(e)}")
    
    def calculate_tag_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about selected tags"""
        
        categorized_tags = self.categorize_selected_tags()
        
        return {
            'total_tags': len(st.session_state['selected_tags']),
            'tags_by_category': {k: len(v) for k, v in categorized_tags.items() if v},
            'custom_tags_count': len(categorized_tags['custom']),
            'tag_coverage': {
                'has_purpose': bool(categorized_tags['purpose']),
                'has_domain': bool(categorized_tags['domain']),
                'has_style': bool(categorized_tags['style']),
                'has_complexity': bool(categorized_tags['complexity'])
            }
        }
    
    def generate_tag_recommendations(self) -> List[str]:
        """Generate recommendations for tag usage"""
        
        recommendations = []
        categorized_tags = self.categorize_selected_tags()
        
        if not categorized_tags['purpose']:
            recommendations.append("Consider adding a Purpose tag (instruction, qa, chat, etc.) for better dataset categorization")
        
        if not categorized_tags['domain']:
            recommendations.append("Consider adding a Domain tag (general, technical, creative, etc.) to specify the content area")
        
        if len(st.session_state['selected_tags']) < 2:
            recommendations.append("Consider adding more tags for better dataset organization and mixing capabilities")
        
        if len(st.session_state['selected_tags']) > 8:
            recommendations.append("Consider reducing the number of tags to avoid over-categorization")
        
        return recommendations
    
    def generate_mixing_suggestions(self) -> List[str]:
        """Generate suggestions for mixing with other datasets"""
        
        suggestions = []
        categorized_tags = self.categorize_selected_tags()
        
        if 'instruction' in st.session_state['selected_tags']:
            suggestions.append("Mix with QA datasets to create comprehensive instruction-following training data")
        
        if 'qa' in st.session_state['selected_tags']:
            suggestions.append("Combine with instruction datasets for balanced question-answering capabilities")
        
        if 'chat' in st.session_state['selected_tags']:
            suggestions.append("Mix with narrative datasets to improve conversational storytelling")
        
        if 'beginner' in st.session_state['selected_tags']:
            suggestions.append("Combine with intermediate complexity data for progressive learning")
        
        return suggestions
    
    def extract_text_content(self, item: Dict) -> str:
        """Extract text content from item for analysis"""
        
        text_parts = []
        
        # Common text fields
        for field in ['question', 'answer', 'content', 'text', 'instruction', 'input', 'output']:
            if field in item and isinstance(item[field], str):
                text_parts.append(item[field])
        
        return ' '.join(text_parts)

# Integration function for main app
def render_export_tagging_system(content_data: List[Dict]):
    """
    Render export tagging system in main app
    
    Usage:
    from modules.export_tagging_system import render_export_tagging_system
    
    render_export_tagging_system(enhanced_content)
    """
    
    tagging_system = ExportTaggingSystem()
    tagging_system.render_tagging_interface(content_data)

# Compact tagging interface for sidebar
def render_compact_tagging_interface():
    """
    Render compact tagging interface for sidebar
    
    Usage:
    with st.sidebar:
        render_compact_tagging_interface()
    """
    
    st.markdown("### 🏷️ Quick Tags")
    
    # Quick tag selection
    quick_tags = ['instruction', 'qa', 'chat', 'general', 'formal', 'casual']
    
    for tag in quick_tags:
        is_selected = tag in st.session_state.get('selected_tags', set())
        
        if st.checkbox(tag.title(), value=is_selected, key=f"quick_{tag}"):
            if 'selected_tags' not in st.session_state:
                st.session_state['selected_tags'] = set()
            st.session_state['selected_tags'].add(tag)
        else:
            if 'selected_tags' in st.session_state:
                st.session_state['selected_tags'].discard(tag)
    
    # Show selected count
    selected_count = len(st.session_state.get('selected_tags', set()))
    if selected_count > 0:
        st.caption(f"{selected_count} tags selected")
    
    if st.button("🏷️ Full Tagging"):
        st.session_state['show_tagging_system'] = True

if __name__ == "__main__":
    # Test the export tagging system
    st.set_page_config(page_title="Export Tagging System Test", layout="wide")
    
    st.title("Export Tagging System Test")
    
    # Sample data
    sample_data = [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "instruction": "Write a Python function to calculate factorial",
            "input": "def factorial(n):",
            "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)"
        }
    ]
    
    # Render tagging system
    render_export_tagging_system(sample_data)

