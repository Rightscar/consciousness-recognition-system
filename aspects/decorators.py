"""
Aspect-Oriented Decorators
==========================

Cross-cutting concerns decorators for logging, security, error handling, and metrics.
Eliminates code duplication across modules by centralizing common functionality.

Features:
- Unified logging with automatic trace information
- Security validation and access control
- Comprehensive error handling with recovery
- Performance metrics and timing
- Audit trails and compliance logging
- Configurable aspect behavior
"""

import functools
import time
import traceback
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import threading
from modules.logger import get_logger
from modules.comprehensive_security_manager import SecurityManager

# Global configuration for aspects
ASPECT_CONFIG = {
    "logging": {
        "enabled": True,
        "log_args": False,
        "log_return": False,
        "log_timing": True
    },
    "security": {
        "enabled": True,
        "validate_inputs": True,
        "sanitize_outputs": True,
        "audit_calls": True
    },
    "metrics": {
        "enabled": True,
        "track_timing": True,
        "track_memory": False,
        "track_calls": True
    },
    "error_handling": {
        "enabled": True,
        "auto_retry": False,
        "max_retries": 3,
        "log_errors": True
    }
}

# Thread-local storage for context
_context = threading.local()

def get_call_context() -> Dict[str, Any]:
    """Get current call context."""
    if not hasattr(_context, 'stack'):
        _context.stack = []
    return {
        "call_stack": _context.stack.copy(),
        "timestamp": datetime.now().isoformat(),
        "thread_id": threading.get_ident()
    }

def push_context(function_name: str, module_name: str):
    """Push function to call context stack."""
    if not hasattr(_context, 'stack'):
        _context.stack = []
    _context.stack.append(f"{module_name}.{function_name}")

def pop_context():
    """Pop function from call context stack."""
    if hasattr(_context, 'stack') and _context.stack:
        _context.stack.pop()

class MetricsCollector:
    """Collect and store performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record_call(self, function_name: str, duration: float, success: bool):
        """Record function call metrics."""
        with self.lock:
            if function_name not in self.metrics:
                self.metrics[function_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_duration": 0.0,
                    "min_duration": float('inf'),
                    "max_duration": 0.0,
                    "avg_duration": 0.0
                }
            
            stats = self.metrics[function_name]
            stats["total_calls"] += 1
            
            if success:
                stats["successful_calls"] += 1
            else:
                stats["failed_calls"] += 1
            
            stats["total_duration"] += duration
            stats["min_duration"] = min(stats["min_duration"], duration)
            stats["max_duration"] = max(stats["max_duration"], duration)
            stats["avg_duration"] = stats["total_duration"] / stats["total_calls"]
    
    def get_metrics(self, function_name: str = None) -> Dict[str, Any]:
        """Get metrics for specific function or all functions."""
        with self.lock:
            if function_name:
                return self.metrics.get(function_name, {})
            return self.metrics.copy()
    
    def reset_metrics(self, function_name: str = None):
        """Reset metrics for specific function or all functions."""
        with self.lock:
            if function_name:
                self.metrics.pop(function_name, None)
            else:
                self.metrics.clear()

# Global metrics collector
_metrics_collector = MetricsCollector()

def trace_and_secure(
    log_level: str = "INFO",
    validate_inputs: bool = True,
    sanitize_outputs: bool = True,
    track_metrics: bool = True,
    auto_retry: bool = False,
    max_retries: int = 3
):
    """
    Comprehensive decorator for logging, security, and metrics.
    
    Args:
        log_level: Logging level for function calls
        validate_inputs: Whether to validate input parameters
        sanitize_outputs: Whether to sanitize output data
        track_metrics: Whether to track performance metrics
        auto_retry: Whether to automatically retry on failure
        max_retries: Maximum number of retry attempts
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function and module information
            module_name = func.__module__.split('.')[-1] if func.__module__ else "unknown"
            function_name = func.__name__
            full_name = f"{module_name}.{function_name}"
            
            # Initialize logger and security manager
            logger = get_logger(module_name)
            security_manager = SecurityManager()
            
            # Push to call context
            push_context(function_name, module_name)
            
            start_time = time.time()
            success = False
            result = None
            error = None
            
            try:
                # Pre-execution logging
                if ASPECT_CONFIG["logging"]["enabled"]:
                    context = get_call_context()
                    logger.info(f"Entering {full_name}", extra={
                        "function": function_name,
                        "module": module_name,
                        "context": context,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    })
                
                # Input validation and security
                if ASPECT_CONFIG["security"]["enabled"] and validate_inputs:
                    # Validate input parameters
                    for i, arg in enumerate(args):
                        if isinstance(arg, str):
                            if not security_manager.validate_input_text(arg):
                                raise ValueError(f"Security validation failed for argument {i}")
                    
                    for key, value in kwargs.items():
                        if isinstance(value, str):
                            if not security_manager.validate_input_text(value):
                                raise ValueError(f"Security validation failed for parameter {key}")
                
                # Execute function with retry logic
                retry_count = 0
                while retry_count <= (max_retries if auto_retry else 0):
                    try:
                        result = func(*args, **kwargs)
                        success = True
                        break
                    except Exception as e:
                        if retry_count < max_retries and auto_retry:
                            retry_count += 1
                            logger.warning(f"Retry {retry_count}/{max_retries} for {full_name}: {str(e)}")
                            time.sleep(0.1 * retry_count)  # Exponential backoff
                        else:
                            raise
                
                # Output sanitization
                if ASPECT_CONFIG["security"]["enabled"] and sanitize_outputs and result:
                    if isinstance(result, str):
                        result = security_manager.sanitize_output_text(result)
                    elif isinstance(result, dict):
                        result = security_manager.sanitize_output_data(result)
                
                return result
                
            except Exception as e:
                error = e
                success = False
                
                # Error logging
                if ASPECT_CONFIG["error_handling"]["enabled"]:
                    logger.error(f"Error in {full_name}: {str(e)}", extra={
                        "function": function_name,
                        "module": module_name,
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                        "context": get_call_context()
                    })
                
                raise
                
            finally:
                # Calculate execution time
                duration = time.time() - start_time
                
                # Pop from call context
                pop_context()
                
                # Post-execution logging
                if ASPECT_CONFIG["logging"]["enabled"]:
                    log_method = logger.info if success else logger.error
                    log_method(f"Exiting {full_name} ({'success' if success else 'failure'}) in {duration:.3f}s", extra={
                        "function": function_name,
                        "module": module_name,
                        "duration": duration,
                        "success": success,
                        "error": str(error) if error else None
                    })
                
                # Metrics collection
                if ASPECT_CONFIG["metrics"]["enabled"] and track_metrics:
                    _metrics_collector.record_call(full_name, duration, success)
                
                # Security audit logging
                if ASPECT_CONFIG["security"]["enabled"]:
                    security_manager.log_function_call(
                        function_name=full_name,
                        success=success,
                        duration=duration,
                        context=get_call_context()
                    )
        
        return wrapper
    return decorator

def log_only(log_level: str = "INFO", log_args: bool = False, log_return: bool = False):
    """Lightweight logging-only decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            module_name = func.__module__.split('.')[-1] if func.__module__ else "unknown"
            function_name = func.__name__
            logger = get_logger(module_name)
            
            start_time = time.time()
            
            # Pre-execution log
            log_data = {"function": function_name, "module": module_name}
            if log_args:
                log_data["args"] = str(args)[:200]  # Truncate for safety
                log_data["kwargs"] = str(kwargs)[:200]
            
            logger.log(getattr(logger, log_level.lower(), logger.info), 
                      f"Calling {function_name}", extra=log_data)
            
            try:
                result = func(*args, **kwargs)
                
                # Post-execution log
                duration = time.time() - start_time
                log_data.update({"duration": duration, "success": True})
                if log_return and result is not None:
                    log_data["return"] = str(result)[:200]  # Truncate for safety
                
                logger.log(getattr(logger, log_level.lower(), logger.info),
                          f"Completed {function_name} in {duration:.3f}s", extra=log_data)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {function_name} in {duration:.3f}s: {str(e)}", extra={
                    "function": function_name,
                    "module": module_name,
                    "duration": duration,
                    "error": str(e),
                    "success": False
                })
                raise
        
        return wrapper
    return decorator

def secure_only(validate_inputs: bool = True, sanitize_outputs: bool = True):
    """Security-only decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            security_manager = SecurityManager()
            
            # Input validation
            if validate_inputs:
                for i, arg in enumerate(args):
                    if isinstance(arg, str) and not security_manager.validate_input_text(arg):
                        raise ValueError(f"Security validation failed for argument {i}")
                
                for key, value in kwargs.items():
                    if isinstance(value, str) and not security_manager.validate_input_text(value):
                        raise ValueError(f"Security validation failed for parameter {key}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Output sanitization
            if sanitize_outputs and result:
                if isinstance(result, str):
                    result = security_manager.sanitize_output_text(result)
                elif isinstance(result, dict):
                    result = security_manager.sanitize_output_data(result)
            
            return result
        
        return wrapper
    return decorator

def metrics_only(track_timing: bool = True, track_calls: bool = True):
    """Metrics-only decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            function_name = f"{func.__module__.split('.')[-1]}.{func.__name__}"
            start_time = time.time() if track_timing else None
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception:
                success = False
                raise
            finally:
                if track_timing and track_calls:
                    duration = time.time() - start_time
                    _metrics_collector.record_call(function_name, duration, success)
        
        return wrapper
    return decorator

def retry_on_failure(max_retries: int = 3, delay: float = 0.1, backoff: float = 2.0):
    """Retry decorator for handling transient failures."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__.split('.')[-1] if func.__module__ else "unknown")
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Final attempt failed for {func.__name__}: {str(e)}")
                        raise
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {wait_time:.2f}s: {str(e)}")
                    time.sleep(wait_time)
        
        return wrapper
    return decorator

# Utility functions for aspect management
def configure_aspects(**config):
    """Configure global aspect behavior."""
    global ASPECT_CONFIG
    for category, settings in config.items():
        if category in ASPECT_CONFIG:
            ASPECT_CONFIG[category].update(settings)

def get_aspect_config() -> Dict[str, Any]:
    """Get current aspect configuration."""
    return ASPECT_CONFIG.copy()

def get_function_metrics(function_name: str = None) -> Dict[str, Any]:
    """Get performance metrics for functions."""
    return _metrics_collector.get_metrics(function_name)

def reset_function_metrics(function_name: str = None):
    """Reset performance metrics."""
    _metrics_collector.reset_metrics(function_name)

def disable_aspects(*aspect_names):
    """Temporarily disable specific aspects."""
    for aspect_name in aspect_names:
        if aspect_name in ASPECT_CONFIG:
            ASPECT_CONFIG[aspect_name]["enabled"] = False

def enable_aspects(*aspect_names):
    """Re-enable specific aspects."""
    for aspect_name in aspect_names:
        if aspect_name in ASPECT_CONFIG:
            ASPECT_CONFIG[aspect_name]["enabled"] = True

# Context managers for temporary aspect configuration
class AspectContext:
    """Context manager for temporary aspect configuration."""
    
    def __init__(self, **config):
        self.config = config
        self.original_config = {}
    
    def __enter__(self):
        global ASPECT_CONFIG
        # Save original configuration
        for category, settings in self.config.items():
            if category in ASPECT_CONFIG:
                self.original_config[category] = ASPECT_CONFIG[category].copy()
                ASPECT_CONFIG[category].update(settings)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global ASPECT_CONFIG
        # Restore original configuration
        for category, settings in self.original_config.items():
            ASPECT_CONFIG[category] = settings

# Convenience context managers
def with_logging_disabled():
    """Context manager to temporarily disable logging."""
    return AspectContext(logging={"enabled": False})

def with_security_disabled():
    """Context manager to temporarily disable security."""
    return AspectContext(security={"enabled": False})

def with_metrics_disabled():
    """Context manager to temporarily disable metrics."""
    return AspectContext(metrics={"enabled": False})

# Example usage and testing
if __name__ == "__main__":
    # Example of using the decorators
    
    @trace_and_secure(log_level="DEBUG", track_metrics=True)
    def example_function(text: str, number: int = 42) -> str:
        """Example function with full aspect coverage."""
        time.sleep(0.1)  # Simulate work
        return f"Processed: {text} with {number}"
    
    @log_only(log_args=True, log_return=True)
    def simple_function(x: int) -> int:
        """Example function with logging only."""
        return x * 2
    
    @retry_on_failure(max_retries=2)
    def unreliable_function(fail_rate: float = 0.5) -> str:
        """Example function that sometimes fails."""
        import random
        if random.random() < fail_rate:
            raise Exception("Random failure")
        return "Success!"
    
    # Test the functions
    try:
        result1 = example_function("test data", 123)
        print(f"Result 1: {result1}")
        
        result2 = simple_function(21)
        print(f"Result 2: {result2}")
        
        result3 = unreliable_function(0.3)
        print(f"Result 3: {result3}")
        
        # Show metrics
        metrics = get_function_metrics()
        print(f"Metrics: {metrics}")
        
    except Exception as e:
        print(f"Error: {e}")

