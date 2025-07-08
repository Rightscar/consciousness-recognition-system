"""
API Rate Limiter Module
=======================

Comprehensive API rate limiting and usage tracking for OpenAI API calls.
Prevents hitting rate limits and quotas with intelligent retry mechanisms.

Features:
- Per-user/session usage tracking
- Retry with exponential backoff
- Model switching (GPT-4 -> GPT-4o-mini -> GPT-3.5)
- Cost estimation and monitoring
- Rate limit detection and handling
"""

import time
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import streamlit as st

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class APIUsageStats:
    """Track API usage statistics"""
    session_id: str
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    requests_per_minute: int = 0
    last_request_time: Optional[datetime] = None
    rate_limit_hits: int = 0
    model_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.model_usage is None:
            self.model_usage = {}


@dataclass
class ModelConfig:
    """Configuration for different OpenAI models"""
    name: str
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    rate_limit_rpm: int
    fallback_model: Optional[str] = None


class APIRateLimiter:
    """Comprehensive API rate limiter with usage tracking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.models = {
            "gpt-4": ModelConfig(
                name="gpt-4",
                max_tokens=8192,
                cost_per_1k_input=0.03,
                cost_per_1k_output=0.06,
                rate_limit_rpm=60,
                fallback_model="gpt-4o-mini"
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                max_tokens=16384,
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.0006,
                rate_limit_rpm=500,
                fallback_model="gpt-3.5-turbo"
            ),
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                max_tokens=4096,
                cost_per_1k_input=0.0015,
                cost_per_1k_output=0.002,
                rate_limit_rpm=3500,
                fallback_model=None
            )
        }
        
        # Default model preference order
        self.model_preference = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]
        
        # Usage tracking
        self.usage_stats = self._load_usage_stats()
        
        # Rate limiting state
        self.request_times = []
        self.last_rate_limit_reset = datetime.now()
        
        # Initialize OpenAI client
        self.client = None
        self.async_client = None
        if OPENAI_AVAILABLE:
            try:
                api_key = st.secrets.get("openai_api_key") or st.secrets.get("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    self.async_client = openai.AsyncOpenAI(api_key=api_key)
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def _load_usage_stats(self) -> APIUsageStats:
        """Load usage statistics from session state"""
        
        session_id = st.session_state.get('session_id', 'default')
        
        if 'api_usage_stats' in st.session_state:
            stats_dict = st.session_state['api_usage_stats']
            return APIUsageStats(**stats_dict)
        else:
            return APIUsageStats(session_id=session_id)
    
    def _save_usage_stats(self):
        """Save usage statistics to session state"""
        st.session_state['api_usage_stats'] = asdict(self.usage_stats)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Rough estimation: 1 token ≈ 4 characters for English
        return len(text) // 4
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for API call"""
        
        if model not in self.models:
            model = "gpt-4o-mini"  # Default fallback
        
        config = self.models[model]
        
        input_cost = (input_tokens / 1000) * config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * config.cost_per_1k_output
        
        return input_cost + output_cost
    
    def check_rate_limit(self, model: str) -> Tuple[bool, int]:
        """Check if we're within rate limits for the model"""
        
        config = self.models.get(model, self.models["gpt-4o-mini"])
        current_time = datetime.now()
        
        # Clean old request times (older than 1 minute)
        cutoff_time = current_time - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check if we're under the rate limit
        requests_in_last_minute = len(self.request_times)
        can_make_request = requests_in_last_minute < config.rate_limit_rpm
        
        wait_time = 0
        if not can_make_request and self.request_times:
            # Calculate wait time until oldest request expires
            oldest_request = min(self.request_times)
            wait_time = int((oldest_request + timedelta(minutes=1) - current_time).total_seconds())
        
        return can_make_request, wait_time
    
    def select_best_model(self, estimated_tokens: int, preferred_model: Optional[str] = None) -> str:
        """Select the best model based on token count and availability"""
        
        # If a specific model is preferred and can handle the tokens, use it
        if preferred_model and preferred_model in self.models:
            config = self.models[preferred_model]
            if estimated_tokens <= config.max_tokens * 0.8:  # Leave 20% buffer
                can_request, _ = self.check_rate_limit(preferred_model)
                if can_request:
                    return preferred_model
        
        # Otherwise, find the best available model
        for model_name in self.model_preference:
            config = self.models[model_name]
            
            # Check token limit
            if estimated_tokens > config.max_tokens * 0.8:
                continue
            
            # Check rate limit
            can_request, _ = self.check_rate_limit(model_name)
            if can_request:
                return model_name
        
        # If all models are rate limited, return the one with shortest wait
        best_model = None
        shortest_wait = float('inf')
        
        for model_name in self.model_preference:
            config = self.models[model_name]
            if estimated_tokens <= config.max_tokens * 0.8:
                _, wait_time = self.check_rate_limit(model_name)
                if wait_time < shortest_wait:
                    shortest_wait = wait_time
                    best_model = model_name
        
        return best_model or "gpt-4o-mini"
    
    async def make_api_call_with_retry(self, 
                                       messages: List[Dict[str, str]], 
                                       preferred_model: Optional[str] = None,
                                       max_retries: int = 3,
                                       **kwargs) -> Dict[str, Any]:
        """Make API call with retry logic and rate limiting"""
        
        if not self.async_client:
            raise Exception("OpenAI client not available")
        
        # Estimate tokens
        total_text = " ".join([msg.get("content", "") for msg in messages])
        estimated_tokens = self.estimate_tokens(total_text)
        
        # Select best model
        selected_model = self.select_best_model(estimated_tokens, preferred_model)
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Check rate limit
                can_request, wait_time = self.check_rate_limit(selected_model)
                
                if not can_request:
                    self.logger.warning(f"Rate limit hit for {selected_model}, waiting {wait_time}s")
                    await asyncio.sleep(wait_time + 1)
                    continue
                
                # Record request time
                self.request_times.append(datetime.now())
                
                # Make API call
                response = await self.async_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    **kwargs
                )
                
                # Update usage statistics
                self._update_usage_stats(selected_model, response)
                
                return {
                    "response": response,
                    "model_used": selected_model,
                    "attempt": attempt + 1,
                    "success": True
                }
                
            except openai.RateLimitError as e:
                self.logger.warning(f"Rate limit error on attempt {attempt + 1}: {e}")
                self.usage_stats.rate_limit_hits += 1
                
                # Try fallback model
                config = self.models.get(selected_model)
                if config and config.fallback_model:
                    selected_model = config.fallback_model
                    self.logger.info(f"Switching to fallback model: {selected_model}")
                
                # Exponential backoff
                wait_time = (2 ** attempt) * 60  # 1, 2, 4 minutes
                await asyncio.sleep(wait_time)
                last_exception = e
                
            except openai.APIError as e:
                self.logger.error(f"API error on attempt {attempt + 1}: {e}")
                
                if "maximum context length" in str(e).lower():
                    # Token limit exceeded, try smaller model
                    if selected_model == "gpt-4":
                        selected_model = "gpt-4o-mini"
                    elif selected_model == "gpt-4o-mini":
                        selected_model = "gpt-3.5-turbo"
                    else:
                        raise e  # No smaller model available
                    
                    self.logger.info(f"Token limit exceeded, switching to: {selected_model}")
                    continue
                
                last_exception = e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                last_exception = e
                await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        self._save_usage_stats()
        return {
            "response": None,
            "model_used": selected_model,
            "attempt": max_retries,
            "success": False,
            "error": str(last_exception)
        }
    
    def make_api_call_sync(self, 
                          messages: List[Dict[str, str]], 
                          preferred_model: Optional[str] = None,
                          max_retries: int = 3,
                          **kwargs) -> Dict[str, Any]:
        """Synchronous version of API call with retry logic"""
        
        if not self.client:
            raise Exception("OpenAI client not available")
        
        # Estimate tokens
        total_text = " ".join([msg.get("content", "") for msg in messages])
        estimated_tokens = self.estimate_tokens(total_text)
        
        # Select best model
        selected_model = self.select_best_model(estimated_tokens, preferred_model)
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Check rate limit
                can_request, wait_time = self.check_rate_limit(selected_model)
                
                if not can_request:
                    self.logger.warning(f"Rate limit hit for {selected_model}, waiting {wait_time}s")
                    time.sleep(wait_time + 1)
                    continue
                
                # Record request time
                self.request_times.append(datetime.now())
                
                # Make API call
                response = self.client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    **kwargs
                )
                
                # Update usage statistics
                self._update_usage_stats(selected_model, response)
                
                return {
                    "response": response,
                    "model_used": selected_model,
                    "attempt": attempt + 1,
                    "success": True
                }
                
            except openai.RateLimitError as e:
                self.logger.warning(f"Rate limit error on attempt {attempt + 1}: {e}")
                self.usage_stats.rate_limit_hits += 1
                
                # Try fallback model
                config = self.models.get(selected_model)
                if config and config.fallback_model:
                    selected_model = config.fallback_model
                    self.logger.info(f"Switching to fallback model: {selected_model}")
                
                # Exponential backoff
                wait_time = (2 ** attempt) * 60  # 1, 2, 4 minutes
                time.sleep(wait_time)
                last_exception = e
                
            except openai.APIError as e:
                self.logger.error(f"API error on attempt {attempt + 1}: {e}")
                
                if "maximum context length" in str(e).lower():
                    # Token limit exceeded, try smaller model
                    if selected_model == "gpt-4":
                        selected_model = "gpt-4o-mini"
                    elif selected_model == "gpt-4o-mini":
                        selected_model = "gpt-3.5-turbo"
                    else:
                        raise e  # No smaller model available
                    
                    self.logger.info(f"Token limit exceeded, switching to: {selected_model}")
                    continue
                
                last_exception = e
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                last_exception = e
                time.sleep(2 ** attempt)
        
        # All retries failed
        self._save_usage_stats()
        return {
            "response": None,
            "model_used": selected_model,
            "attempt": max_retries,
            "success": False,
            "error": str(last_exception)
        }
    
    def _update_usage_stats(self, model: str, response):
        """Update usage statistics after successful API call"""
        
        self.usage_stats.total_requests += 1
        self.usage_stats.last_request_time = datetime.now()
        
        # Update model usage
        if model not in self.usage_stats.model_usage:
            self.usage_stats.model_usage[model] = 0
        self.usage_stats.model_usage[model] += 1
        
        # Update token and cost tracking
        if hasattr(response, 'usage') and response.usage:
            total_tokens = response.usage.total_tokens
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            self.usage_stats.total_tokens += total_tokens
            
            # Calculate cost
            cost = self.estimate_cost(model, input_tokens, output_tokens)
            self.usage_stats.total_cost += cost
        
        # Save updated stats
        self._save_usage_stats()
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary"""
        
        return {
            "session_id": self.usage_stats.session_id,
            "total_requests": self.usage_stats.total_requests,
            "total_tokens": self.usage_stats.total_tokens,
            "total_cost": round(self.usage_stats.total_cost, 4),
            "rate_limit_hits": self.usage_stats.rate_limit_hits,
            "model_usage": self.usage_stats.model_usage,
            "last_request": self.usage_stats.last_request_time.isoformat() if self.usage_stats.last_request_time else None,
            "current_rpm": len([t for t in self.request_times if t > datetime.now() - timedelta(minutes=1)])
        }
    
    def render_usage_dashboard(self):
        """Render usage dashboard in Streamlit sidebar"""
        
        with st.sidebar.expander("📊 API Usage Dashboard", expanded=False):
            summary = self.get_usage_summary()
            
            # Current session stats
            st.write("**Current Session:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Requests", summary["total_requests"])
                st.metric("Tokens", f"{summary['total_tokens']:,}")
            with col2:
                st.metric("Cost", f"${summary['total_cost']:.4f}")
                st.metric("Rate Limits", summary["rate_limit_hits"])
            
            # Current rate limiting status
            st.write("**Rate Limit Status:**")
            for model_name in self.model_preference:
                can_request, wait_time = self.check_rate_limit(model_name)
                status = "✅ Available" if can_request else f"⏳ Wait {wait_time}s"
                st.write(f"- {model_name}: {status}")
            
            # Model usage distribution
            if summary["model_usage"]:
                st.write("**Model Usage:**")
                for model, count in summary["model_usage"].items():
                    percentage = (count / summary["total_requests"]) * 100
                    st.write(f"- {model}: {count} ({percentage:.1f}%)")
            
            # Cost warnings
            if summary["total_cost"] > 1.0:
                st.warning(f"⚠️ High usage: ${summary['total_cost']:.2f}")
            elif summary["total_cost"] > 0.1:
                st.info(f"💡 Current cost: ${summary['total_cost']:.4f}")
            
            # Reset button
            if st.button("🔄 Reset Usage Stats"):
                self.usage_stats = APIUsageStats(session_id=self.usage_stats.session_id)
                self._save_usage_stats()
                st.success("Usage stats reset!")
                st.rerun()


# Global rate limiter instance
rate_limiter = APIRateLimiter()


def get_rate_limiter() -> APIRateLimiter:
    """Get the global rate limiter instance"""
    return rate_limiter


def estimate_batch_cost(content_items: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Estimate cost for processing a batch of content items"""
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for item in content_items:
        content = item.get('content', '')
        input_tokens = rate_limiter.estimate_tokens(content)
        output_tokens = input_tokens * 1.5  # Estimate 50% expansion
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
    
    estimated_cost = rate_limiter.estimate_cost(model, total_input_tokens, total_output_tokens)
    
    return {
        "total_items": len(content_items),
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_cost": estimated_cost,
        "model": model,
        "cost_per_item": estimated_cost / len(content_items) if content_items else 0
    }

