"""
Enhanced Custom Prompt Engine
============================

Enhanced version with dynamic prompt template loading from prompts/ folder.
Supports all 6 spiritual tones with full system prompts loaded from .txt files.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCustomPromptEngine:
    """
    Enhanced custom prompt engine with dynamic prompt template loading
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.available_tones = [
            "universal_wisdom",
            "advaita_vedanta", 
            "zen_buddhism",
            "sufi_mysticism",
            "christian_mysticism",
            "mindfulness_meditation"
        ]
        self.tone_display_names = {
            "universal_wisdom": "Universal Wisdom",
            "advaita_vedanta": "Advaita Vedanta",
            "zen_buddhism": "Zen Buddhism", 
            "sufi_mysticism": "Sufi Mysticism",
            "christian_mysticism": "Christian Mysticism",
            "mindfulness_meditation": "Mindfulness Meditation"
        }
        self.loaded_prompts = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all prompt templates from the prompts directory"""
        try:
            for tone in self.available_tones:
                prompt_file = self.prompts_dir / f"{tone}.txt"
                if prompt_file.exists():
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        self.loaded_prompts[tone] = f.read().strip()
                    logger.info(f"Loaded prompt template for {tone}")
                else:
                    logger.warning(f"Prompt file not found: {prompt_file}")
                    # Fallback to basic prompt
                    self.loaded_prompts[tone] = self._get_fallback_prompt(tone)
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")
            self._load_fallback_prompts()
    
    def _get_fallback_prompt(self, tone: str) -> str:
        """Get fallback prompt if file is not available"""
        fallback_prompts = {
            "universal_wisdom": "You are a wise spiritual guide. Enhance the following spiritual content with universal wisdom and compassion.",
            "advaita_vedanta": "You are a teacher of Advaita Vedanta. Enhance the following content with non-dual wisdom and direct pointing to the Self.",
            "zen_buddhism": "You are a Zen master. Enhance the following content with direct, immediate wisdom that cuts through conceptual thinking.",
            "sufi_mysticism": "You are a Sufi mystic. Enhance the following content with the passionate love and devotion of the mystical path.",
            "christian_mysticism": "You are a Christian mystic. Enhance the following content with contemplative wisdom centered in Christ.",
            "mindfulness_meditation": "You are a mindfulness teacher. Enhance the following content with present-moment awareness and practical wisdom."
        }
        return fallback_prompts.get(tone, "You are a spiritual guide. Enhance the following content with wisdom and compassion.")
    
    def _load_fallback_prompts(self):
        """Load fallback prompts if directory loading fails"""
        for tone in self.available_tones:
            self.loaded_prompts[tone] = self._get_fallback_prompt(tone)
    
    def get_available_tones(self) -> List[str]:
        """Get list of available spiritual tones"""
        return list(self.available_tones)
    
    def get_tone_display_names(self) -> Dict[str, str]:
        """Get mapping of tone keys to display names"""
        return self.tone_display_names.copy()
    
    def get_prompt_template(self, tone: str) -> str:
        """Get the full prompt template for a specific tone"""
        return self.loaded_prompts.get(tone, self._get_fallback_prompt(tone))
    
    def enhance_content(self, content: List[Dict[str, Any]], tone: str, 
                       enhancement_mode: str = "improve", 
                       api_key: str = None) -> List[Dict[str, Any]]:
        """
        Enhance content using the selected spiritual tone
        
        Args:
            content: List of Q&A examples to enhance
            tone: Selected spiritual tone
            enhancement_mode: Type of enhancement (improve, expand, rephrase)
            api_key: OpenAI API key
            
        Returns:
            List of enhanced examples
        """
        if not api_key:
            logger.error("OpenAI API key required for enhancement")
            return content
        
        try:
            import openai
            openai.api_key = api_key
            
            # Get the full prompt template for the selected tone
            system_prompt = self.get_prompt_template(tone)
            
            enhanced_content = []
            
            for example in content:
                try:
                    enhanced_example = self._enhance_single_example(
                        example, system_prompt, enhancement_mode, openai
                    )
                    enhanced_content.append(enhanced_example)
                except Exception as e:
                    logger.error(f"Failed to enhance example: {e}")
                    # Keep original if enhancement fails
                    enhanced_content.append(example)
            
            return enhanced_content
            
        except ImportError:
            logger.error("OpenAI library not available")
            return content
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return content
    
    def _enhance_single_example(self, example: Dict[str, Any], 
                               system_prompt: str, enhancement_mode: str,
                               openai) -> Dict[str, Any]:
        """Enhance a single Q&A example"""
        
        question = example.get('question', '')
        answer = example.get('answer', '')
        
        # Create enhancement prompt based on mode
        if enhancement_mode == "improve":
            user_prompt = f"""Please enhance this spiritual Q&A pair by improving clarity, depth, and wisdom while maintaining the original essence:

Question: {question}
Answer: {answer}

Enhanced Question:
Enhanced Answer:"""
        
        elif enhancement_mode == "expand":
            user_prompt = f"""Please expand this spiritual Q&A pair with additional depth, context, and practical wisdom:

Question: {question}
Answer: {answer}

Expanded Question:
Expanded Answer:"""
        
        elif enhancement_mode == "rephrase":
            user_prompt = f"""Please rephrase this spiritual Q&A pair in the authentic voice of this tradition while preserving the core meaning:

Question: {question}
Answer: {answer}

Rephrased Question:
Rephrased Answer:"""
        
        else:
            user_prompt = f"""Please enhance this spiritual Q&A pair:

Question: {question}
Answer: {answer}

Enhanced Question:
Enhanced Answer:"""
        
        # Make API call
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            enhanced_text = response.choices[0].message.content.strip()
            
            # Parse the enhanced response
            enhanced_question, enhanced_answer = self._parse_enhanced_response(
                enhanced_text, question, answer
            )
            
            # Create enhanced example
            enhanced_example = example.copy()
            enhanced_example.update({
                'question': enhanced_question,
                'answer': enhanced_answer,
                'original_question': question,
                'original_answer': answer,
                'enhancement_tone': tone,
                'enhancement_mode': enhancement_mode,
                'enhanced': True
            })
            
            return enhanced_example
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def _parse_enhanced_response(self, enhanced_text: str, 
                                original_question: str, 
                                original_answer: str) -> tuple:
        """Parse the enhanced response to extract question and answer"""
        
        try:
            # Look for common patterns in the response
            lines = enhanced_text.split('\n')
            enhanced_question = original_question
            enhanced_answer = original_answer
            
            current_section = None
            question_lines = []
            answer_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect section headers
                if any(marker in line.lower() for marker in ['enhanced question:', 'expanded question:', 'rephrased question:']):
                    current_section = 'question'
                    # Extract content after the colon if present
                    if ':' in line:
                        content = line.split(':', 1)[1].strip()
                        if content:
                            question_lines.append(content)
                elif any(marker in line.lower() for marker in ['enhanced answer:', 'expanded answer:', 'rephrased answer:']):
                    current_section = 'answer'
                    # Extract content after the colon if present
                    if ':' in line:
                        content = line.split(':', 1)[1].strip()
                        if content:
                            answer_lines.append(content)
                elif current_section == 'question':
                    question_lines.append(line)
                elif current_section == 'answer':
                    answer_lines.append(line)
            
            # Reconstruct enhanced content
            if question_lines:
                enhanced_question = ' '.join(question_lines).strip()
            if answer_lines:
                enhanced_answer = ' '.join(answer_lines).strip()
            
            # Fallback: if parsing fails, try simple split
            if not question_lines and not answer_lines:
                parts = enhanced_text.split('\n\n')
                if len(parts) >= 2:
                    enhanced_question = parts[0].strip()
                    enhanced_answer = parts[1].strip()
            
            return enhanced_question, enhanced_answer
            
        except Exception as e:
            logger.error(f"Failed to parse enhanced response: {e}")
            return original_question, original_answer
    
    def estimate_cost(self, content: List[Dict[str, Any]], 
                     enhancement_mode: str = "improve") -> Dict[str, float]:
        """Estimate the cost of enhancing content"""
        
        # Rough token estimation
        total_tokens = 0
        for example in content:
            question_tokens = len(example.get('question', '').split()) * 1.3  # Rough token estimate
            answer_tokens = len(example.get('answer', '').split()) * 1.3
            prompt_tokens = 200  # Estimated system prompt tokens
            total_tokens += question_tokens + answer_tokens + prompt_tokens
        
        # OpenAI pricing (approximate)
        cost_per_1k_tokens = 0.002  # GPT-3.5-turbo pricing
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            'estimated_tokens': int(total_tokens),
            'estimated_cost_usd': round(estimated_cost, 4),
            'examples_count': len(content)
        }
    
    def reload_prompts(self):
        """Reload all prompt templates from files"""
        self.loaded_prompts.clear()
        self._load_all_prompts()
        logger.info("Prompt templates reloaded")
    
    def get_prompt_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded prompts"""
        info = {}
        for tone in self.available_tones:
            prompt = self.loaded_prompts.get(tone, '')
            info[tone] = {
                'display_name': self.tone_display_names.get(tone, tone),
                'loaded': bool(prompt),
                'length': len(prompt),
                'file_path': str(self.prompts_dir / f"{tone}.txt")
            }
        return info

