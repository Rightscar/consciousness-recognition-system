"""
Async Enhancement Processor Module
==================================

Handles large enhancement batches with async processing and intelligent chunking.
Prevents timeouts and provides progress tracking for better user experience.

Features:
- Async OpenAI API calls with rate limiting
- Intelligent chunking (10-20 Q&A pairs per batch)
- Progress tracking and error handling
- Concurrent processing with backoff
- Memory-efficient batch processing
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Generator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available - AI enhancement features will be limited")


class AsyncEnhancementProcessor:
    """Async processor for large enhancement batches"""
    
    def __init__(self, 
                 chunk_size: int = 15,
                 max_concurrent: int = 5,
                 rate_limit_delay: float = 1.0):
        self.chunk_size = chunk_size  # Sweet spot: 10-20 Q&A pairs
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client if available
        self.client = None
        self.async_client = None
        if OPENAI_AVAILABLE:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    self.async_client = openai.AsyncOpenAI(api_key=api_key)
                else:
                    self.logger.warning("OpenAI API key not found in secrets")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def create_chunks(self, content_items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create intelligent chunks for processing"""
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for item in content_items:
            # Estimate token count (rough approximation)
            content = item.get('content', '')
            estimated_tokens = len(content.split()) * 1.3  # Rough token estimate
            
            # Check if adding this item would exceed chunk size
            if (current_size + estimated_tokens > self.chunk_size * 100 and 
                len(current_chunk) >= 5):  # Minimum 5 items per chunk
                
                chunks.append(current_chunk)
                current_chunk = [item]
                current_size = estimated_tokens
            else:
                current_chunk.append(item)
                current_size += estimated_tokens
        
        # Add remaining items
        if current_chunk:
            chunks.append(current_chunk)
        
        self.logger.info(f"Created {len(chunks)} chunks from {len(content_items)} items")
        return chunks
    
    async def enhance_batch_async(self, 
                                  batch: List[Dict[str, Any]], 
                                  prompt_template: str,
                                  tone: str,
                                  batch_id: int) -> Dict[str, Any]:
        """Enhance a batch of content asynchronously"""
        
        if not self.async_client:
            return {
                "batch_id": batch_id,
                "success": False,
                "error": "OpenAI client not available",
                "results": []
            }
        
        try:
            # Prepare batch content
            batch_content = []
            for i, item in enumerate(batch):
                content = item.get('content', '')
                batch_content.append(f"Item {i+1}: {content}")
            
            combined_content = "\n\n".join(batch_content)
            
            # Create prompt
            full_prompt = f"""
{prompt_template}

Tone: {tone}

Please enhance the following content items, maintaining their individual structure:

{combined_content}

Please return the enhanced versions in the same order, clearly separated and labeled as "Enhanced Item 1:", "Enhanced Item 2:", etc.
"""
            
            # Make async API call
            response = await self.async_client.chat.completions.create(
                model="gpt-4o-mini",  # Use cost-effective model
                messages=[
                    {"role": "system", "content": "You are an expert content enhancer specializing in spiritual and consciousness-related content."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            # Parse response
            enhanced_text = response.choices[0].message.content
            enhanced_items = self._parse_batch_response(enhanced_text, len(batch))
            
            # Create results
            results = []
            for i, (original_item, enhanced_content) in enumerate(zip(batch, enhanced_items)):
                results.append({
                    "original": original_item,
                    "enhanced": enhanced_content,
                    "tone": tone,
                    "batch_id": batch_id,
                    "item_index": i,
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "batch_id": batch_id,
                "success": True,
                "results": results,
                "token_usage": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            self.logger.error(f"Batch {batch_id} enhancement failed: {e}")
            return {
                "batch_id": batch_id,
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse batch response into individual enhanced items"""
        
        enhanced_items = []
        
        # Try to split by "Enhanced Item X:" pattern
        import re
        pattern = r"Enhanced Item \d+:"
        parts = re.split(pattern, response_text)
        
        # Remove empty first part
        if parts and not parts[0].strip():
            parts = parts[1:]
        
        # If we got the expected number of parts, use them
        if len(parts) == expected_count:
            enhanced_items = [part.strip() for part in parts]
        else:
            # Fallback: split by double newlines and take first N parts
            fallback_parts = response_text.split('\n\n')
            enhanced_items = fallback_parts[:expected_count]
            
            # If still not enough, pad with original content
            while len(enhanced_items) < expected_count:
                enhanced_items.append("[Enhancement failed - using original content]")
        
        return enhanced_items
    
    async def process_all_batches_async(self, 
                                        chunks: List[List[Dict[str, Any]]], 
                                        prompt_template: str,
                                        tone: str,
                                        progress_callback=None) -> List[Dict[str, Any]]:
        """Process all batches asynchronously with rate limiting"""
        
        all_results = []
        completed = 0
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(batch, batch_id):
            async with semaphore:
                # Add delay for rate limiting
                if batch_id > 0:
                    await asyncio.sleep(self.rate_limit_delay)
                
                result = await self.enhance_batch_async(batch, prompt_template, tone, batch_id)
                
                nonlocal completed
                completed += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(completed, len(chunks))
                
                return result
        
        # Create tasks for all batches
        tasks = [
            process_with_semaphore(batch, i) 
            for i, batch in enumerate(chunks)
        ]
        
        # Execute all tasks
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in batch_results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing exception: {result}")
                continue
            
            if result.get("success", False):
                all_results.extend(result.get("results", []))
            else:
                self.logger.error(f"Batch {result.get('batch_id')} failed: {result.get('error')}")
        
        return all_results
    
    def process_large_batch_sync(self, 
                                 content_items: List[Dict[str, Any]], 
                                 prompt_template: str,
                                 tone: str,
                                 progress_callback=None) -> List[Dict[str, Any]]:
        """Process large batch synchronously (fallback for environments without async support)"""
        
        chunks = self.create_chunks(content_items)
        all_results = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(i, len(chunks))
                
                # Process chunk synchronously
                result = self._enhance_batch_sync(chunk, prompt_template, tone, i)
                
                if result.get("success", False):
                    all_results.extend(result.get("results", []))
                else:
                    self.logger.error(f"Chunk {i} failed: {result.get('error')}")
                
                # Rate limiting delay
                if i < len(chunks) - 1:
                    time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {i}: {e}")
                continue
        
        return all_results
    
    def _enhance_batch_sync(self, 
                           batch: List[Dict[str, Any]], 
                           prompt_template: str,
                           tone: str,
                           batch_id: int) -> Dict[str, Any]:
        """Enhance batch synchronously"""
        
        if not self.client:
            return {
                "batch_id": batch_id,
                "success": False,
                "error": "OpenAI client not available",
                "results": []
            }
        
        try:
            # Prepare batch content (same as async version)
            batch_content = []
            for i, item in enumerate(batch):
                content = item.get('content', '')
                batch_content.append(f"Item {i+1}: {content}")
            
            combined_content = "\n\n".join(batch_content)
            
            # Create prompt
            full_prompt = f"""
{prompt_template}

Tone: {tone}

Please enhance the following content items, maintaining their individual structure:

{combined_content}

Please return the enhanced versions in the same order, clearly separated and labeled as "Enhanced Item 1:", "Enhanced Item 2:", etc.
"""
            
            # Make sync API call
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert content enhancer specializing in spiritual and consciousness-related content."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            # Parse response
            enhanced_text = response.choices[0].message.content
            enhanced_items = self._parse_batch_response(enhanced_text, len(batch))
            
            # Create results
            results = []
            for i, (original_item, enhanced_content) in enumerate(zip(batch, enhanced_items)):
                results.append({
                    "original": original_item,
                    "enhanced": enhanced_content,
                    "tone": tone,
                    "batch_id": batch_id,
                    "item_index": i,
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "batch_id": batch_id,
                "success": True,
                "results": results,
                "token_usage": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            self.logger.error(f"Sync batch {batch_id} enhancement failed: {e}")
            return {
                "batch_id": batch_id,
                "success": False,
                "error": str(e),
                "results": []
            }


class StreamlitAsyncProcessor:
    """Wrapper for running async processing in Streamlit"""
    
    def __init__(self):
        self.processor = AsyncEnhancementProcessor()
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def process_with_progress(self, 
                             content_items: List[Dict[str, Any]], 
                             prompt_template: str,
                             tone: str) -> List[Dict[str, Any]]:
        """Process content with Streamlit progress tracking"""
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(completed: int, total: int):
            progress = completed / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"Processing batch {completed} of {total}")
        
        try:
            # Check if we can use async
            if asyncio.get_event_loop().is_running():
                # Already in async context, use sync version
                results = self.processor.process_large_batch_sync(
                    content_items, prompt_template, tone, update_progress
                )
            else:
                # Can use async
                chunks = self.processor.create_chunks(content_items)
                
                async def run_async():
                    return await self.processor.process_all_batches_async(
                        chunks, prompt_template, tone, update_progress
                    )
                
                results = asyncio.run(run_async())
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return results
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Enhancement processing failed: {e}")
            return []


# Global async processor instance
async_processor = StreamlitAsyncProcessor()


def estimate_processing_time(num_items: int, chunk_size: int = 15) -> str:
    """Estimate processing time for given number of items"""
    
    num_chunks = (num_items + chunk_size - 1) // chunk_size
    
    # Rough estimates based on API response times
    if num_chunks <= 1:
        return "1-2 minutes"
    elif num_chunks <= 5:
        return "3-5 minutes"
    elif num_chunks <= 10:
        return "5-10 minutes"
    elif num_chunks <= 20:
        return "10-20 minutes"
    else:
        return f"{num_chunks * 1.5:.0f}+ minutes"


def get_chunk_statistics(content_items: List[Dict[str, Any]], chunk_size: int = 15) -> Dict[str, Any]:
    """Get statistics about chunking for given content"""
    
    processor = AsyncEnhancementProcessor(chunk_size=chunk_size)
    chunks = processor.create_chunks(content_items)
    
    chunk_sizes = [len(chunk) for chunk in chunks]
    
    return {
        "total_items": len(content_items),
        "num_chunks": len(chunks),
        "chunk_size_setting": chunk_size,
        "actual_chunk_sizes": chunk_sizes,
        "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
        "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
        "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
        "estimated_time": estimate_processing_time(len(content_items), chunk_size)
    }

