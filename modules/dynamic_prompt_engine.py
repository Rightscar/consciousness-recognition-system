"""
Dynamic Prompt Template Engine
=============================

Provides dynamic loading and management of prompt templates for different spiritual tones.
Implements Core Enhancement 2: Dynamic Prompt Templates Per Tone.

Features:
- Dynamic template loading from prompts/ folder
- Template validation and error handling
- Tone-specific prompt injection
- Template caching for performance
- Custom template support
"""

import streamlit as st
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DynamicPromptEngine:
    """Enhanced prompt engine with dynamic template loading"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.template_cache = {}
        self.available_tones = []
        self._load_available_tones()
        
    def _load_available_tones(self) -> None:
        """Load available spiritual tones from prompts directory"""
        
        try:
            if not self.prompts_dir.exists():
                logger.warning(f"Prompts directory {self.prompts_dir} does not exist")
                self.available_tones = []
                return
                
            # Find all .txt files in prompts directory
            tone_files = list(self.prompts_dir.glob("*.txt"))
            self.available_tones = [f.stem for f in tone_files]
            
            logger.info(f"Loaded {len(self.available_tones)} spiritual tones: {self.available_tones}")
            
        except Exception as e:
            logger.error(f"Error loading available tones: {e}")
            self.available_tones = []
            
    def get_available_tones(self) -> List[str]:
        """Get list of available spiritual tones"""
        return self.available_tones.copy()
        
    def load_prompt_template(self, tone: str) -> Optional[str]:
        """Load prompt template for specified tone - Core Enhancement 2 requirement"""
        
        # Check cache first
        if tone in self.template_cache:
            return self.template_cache[tone]
            
        try:
            template_path = self.prompts_dir / f"{tone}.txt"
            
            if not template_path.exists():
                logger.error(f"Template file not found: {template_path}")
                return None
                
            # Load template content
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read().strip()
                
            if not template_content:
                logger.error(f"Template file is empty: {template_path}")
                return None
                
            # Cache the template
            self.template_cache[tone] = template_content
            
            logger.info(f"Loaded prompt template for tone: {tone}")
            return template_content
            
        except Exception as e:
            logger.error(f"Error loading prompt template for {tone}: {e}")
            return None
            
    def get_system_prompt(self, tone: str, custom_instructions: Optional[str] = None) -> str:
        """Get complete system prompt with tone-specific template"""
        
        # Load base template for the tone
        base_template = self.load_prompt_template(tone)
        
        if not base_template:
            # Fallback to universal wisdom if specific tone not found
            base_template = self.load_prompt_template("universal_wisdom")
            
        if not base_template:
            # Ultimate fallback
            base_template = self._get_fallback_template()
            
        # Add custom instructions if provided
        if custom_instructions:
            system_prompt = f"{base_template}\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions}"
        else:
            system_prompt = base_template
            
        return system_prompt
        
    def _get_fallback_template(self) -> str:
        """Get fallback template when no specific template is available"""
        
        return """You are a wise spiritual teacher helping to enhance Q&A content with depth and clarity.

TONE & STYLE:
- Speak with wisdom and compassion
- Use clear, accessible language
- Maintain spiritual depth while being practical
- Balance intellectual understanding with heart wisdom

CONTENT GUIDELINES:
- Enhance questions to be more precise and meaningful
- Expand answers with spiritual insight and practical wisdom
- Use appropriate spiritual terminology with clear explanations
- Focus on direct experience and understanding
- Maintain authenticity and sincerity

ENHANCEMENT APPROACH:
- Improve clarity and depth of both questions and answers
- Add relevant spiritual context and wisdom
- Use metaphors and analogies to illustrate points
- Emphasize practical application of spiritual principles
- Maintain the original intent while enhancing quality

Remember: You are enhancing existing content with spiritual wisdom and clarity."""

    def validate_template(self, template_content: str) -> Dict[str, Any]:
        """Validate template content and structure"""
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'sections': []
        }
        
        try:
            # Check for required sections
            required_sections = ['TONE & STYLE', 'CONTENT GUIDELINES', 'ENHANCEMENT APPROACH']
            found_sections = []
            
            for section in required_sections:
                if section in template_content:
                    found_sections.append(section)
                else:
                    validation_result['warnings'].append(f"Missing recommended section: {section}")
                    
            validation_result['sections'] = found_sections
            
            # Check template length
            if len(template_content) < 100:
                validation_result['warnings'].append("Template seems very short")
            elif len(template_content) > 5000:
                validation_result['warnings'].append("Template is very long - may impact performance")
                
            # Check for common issues
            if not template_content.strip():
                validation_result['valid'] = False
                validation_result['errors'].append("Template is empty")
                
            # Check encoding
            try:
                template_content.encode('utf-8')
            except UnicodeEncodeError:
                validation_result['warnings'].append("Template contains non-UTF-8 characters")
                
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            
        return validation_result
        
    def create_enhancement_prompt(self, tone: str, question: str, answer: str, 
                                custom_instructions: Optional[str] = None) -> str:
        """Create complete enhancement prompt for GPT API"""
        
        system_prompt = self.get_system_prompt(tone, custom_instructions)
        
        user_prompt = f"""Please enhance this Q&A pair with the specified spiritual tone and approach:

ORIGINAL QUESTION:
{question}

ORIGINAL ANSWER:
{answer}

ENHANCEMENT REQUIREMENTS:
1. Improve the clarity and depth of both question and answer
2. Maintain the original meaning while enhancing spiritual insight
3. Use the specified tone and style consistently
4. Ensure the enhanced content is suitable for AI training data
5. Keep the enhanced content focused and practical

Please provide the enhanced version in this format:

ENHANCED QUESTION:
[Your enhanced question here]

ENHANCED ANSWER:
[Your enhanced answer here]"""

        return system_prompt, user_prompt
        
    def render_tone_selector(self, key_suffix: str = "") -> str:
        """Render Streamlit tone selector widget"""
        
        if not self.available_tones:
            st.error("❌ No spiritual tones available. Please check the prompts/ directory.")
            return "universal_wisdom"
            
        # Create user-friendly tone names
        tone_display_names = {}
        for tone in self.available_tones:
            display_name = tone.replace('_', ' ').title()
            tone_display_names[display_name] = tone
            
        selected_display = st.selectbox(
            "🎭 **Spiritual Tone**",
            options=list(tone_display_names.keys()),
            index=0,
            key=f"tone_selector_{key_suffix}",
            help="Choose the spiritual tradition/tone for content enhancement"
        )
        
        return tone_display_names[selected_display]
        
    def render_template_preview(self, tone: str) -> None:
        """Render template preview in Streamlit"""
        
        template = self.load_prompt_template(tone)
        
        if template:
            with st.expander(f"📜 Preview {tone.replace('_', ' ').title()} Template"):
                st.text_area(
                    "Template Content",
                    value=template,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
                
                # Show validation results
                validation = self.validate_template(template)
                
                if validation['valid']:
                    st.success("✅ Template is valid")
                else:
                    st.error("❌ Template has errors")
                    for error in validation['errors']:
                        st.error(f"Error: {error}")
                        
                if validation['warnings']:
                    for warning in validation['warnings']:
                        st.warning(f"Warning: {warning}")
                        
                # Show template statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Characters", len(template))
                with col2:
                    st.metric("Words", len(template.split()))
                with col3:
                    st.metric("Sections", len(validation['sections']))
        else:
            st.error(f"❌ Could not load template for {tone}")
            
    def render_custom_instructions(self, key_suffix: str = "") -> Optional[str]:
        """Render custom instructions input widget"""
        
        with st.expander("🎯 **Custom Instructions** (Optional)"):
            custom_instructions = st.text_area(
                "Additional Enhancement Instructions",
                placeholder="Add specific instructions for this enhancement session...\n\nExample:\n- Focus on practical applications\n- Use more metaphors\n- Emphasize meditation techniques",
                height=100,
                key=f"custom_instructions_{key_suffix}",
                help="Add specific instructions to customize the enhancement for this session"
            )
            
            if custom_instructions.strip():
                st.info(f"✅ Custom instructions added ({len(custom_instructions)} characters)")
                return custom_instructions.strip()
            else:
                return None
                
    def get_tone_statistics(self) -> Dict[str, Any]:
        """Get statistics about available tones and templates"""
        
        stats = {
            'total_tones': len(self.available_tones),
            'cached_templates': len(self.template_cache),
            'tone_details': {}
        }
        
        for tone in self.available_tones:
            template = self.load_prompt_template(tone)
            if template:
                validation = self.validate_template(template)
                stats['tone_details'][tone] = {
                    'character_count': len(template),
                    'word_count': len(template.split()),
                    'valid': validation['valid'],
                    'sections': len(validation['sections']),
                    'warnings': len(validation['warnings']),
                    'errors': len(validation['errors'])
                }
            else:
                stats['tone_details'][tone] = {
                    'character_count': 0,
                    'word_count': 0,
                    'valid': False,
                    'sections': 0,
                    'warnings': 0,
                    'errors': 1
                }
                
        return stats
        
    def refresh_templates(self) -> None:
        """Refresh template cache and reload available tones"""
        
        self.template_cache.clear()
        self._load_available_tones()
        logger.info("Template cache refreshed")
        
    def export_template(self, tone: str) -> Optional[str]:
        """Export template content for backup or sharing"""
        
        template = self.load_prompt_template(tone)
        if template:
            return template
        return None
        
    def import_template(self, tone: str, template_content: str) -> bool:
        """Import new template content"""
        
        try:
            # Validate template first
            validation = self.validate_template(template_content)
            
            if not validation['valid']:
                logger.error(f"Cannot import invalid template for {tone}")
                return False
                
            # Save to file
            template_path = self.prompts_dir / f"{tone}.txt"
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
                
            # Update cache
            self.template_cache[tone] = template_content
            
            # Refresh available tones
            self._load_available_tones()
            
            logger.info(f"Successfully imported template for {tone}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing template for {tone}: {e}")
            return False

