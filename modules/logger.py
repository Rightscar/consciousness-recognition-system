"""
Centralized Logging System
==========================

Comprehensive logging system for error and event tracking with timestamps.
Provides centralized configuration and integration for all application modules.

Based on user's requirement:
import logging
logging.basicConfig(filename='app.log', level=logging.INFO)
logging.info("Enhancement completed at %s", datetime.now())
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import traceback
from pathlib import Path
import threading
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import streamlit as st

class CentralizedLogger:
    """
    Centralized logging system for the Enhanced Universal AI Training Data Creator
    
    Features:
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - File and console logging with rotation
    - Structured logging with timestamps
    - Module-specific loggers
    - Performance tracking
    - Error tracking with stack traces
    - Event correlation and tracking
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for centralized logger"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CentralizedLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.log_dir = "logs"
        self.log_file = "app.log"
        self.error_log_file = "errors.log"
        self.performance_log_file = "performance.log"
        self.event_log_file = "events.log"
        
        # Create logs directory
        self.setup_log_directory()
        
        # Setup logging configuration
        self.setup_logging()
        
        # Initialize module loggers
        self.module_loggers = {}
        
        # Initialize event tracking
        self.event_counter = 0
        self.session_id = self.generate_session_id()
        
        # Log system startup
        self.log_system_startup()
    
    def setup_log_directory(self):
        """Create logs directory structure"""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Create subdirectories for different log types
            os.makedirs(os.path.join(self.log_dir, "modules"), exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "performance"), exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "errors"), exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "events"), exist_ok=True)
            
        except Exception as e:
            print(f"Failed to create log directory: {str(e)}")
    
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        
        # Create custom formatter with timestamps
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Main application log file with rotation
        main_log_path = os.path.join(self.log_dir, self.log_file)
        file_handler = RotatingFileHandler(
            main_log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Error-specific log file
        error_log_path = os.path.join(self.log_dir, self.error_log_file)
        error_handler = RotatingFileHandler(
            error_log_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Performance log file
        performance_log_path = os.path.join(self.log_dir, self.performance_log_file)
        performance_handler = RotatingFileHandler(
            performance_log_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(formatter)
        
        # Event log file
        event_log_path = os.path.join(self.log_dir, self.event_log_file)
        event_handler = RotatingFileHandler(
            event_log_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        event_handler.setLevel(logging.INFO)
        event_handler.setFormatter(formatter)
        
        # Store handlers for specific logging
        self.performance_handler = performance_handler
        self.event_handler = event_handler
    
    def generate_session_id(self) -> str:
        """Generate unique session ID for tracking"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    
    def get_module_logger(self, module_name: str) -> logging.Logger:
        """Get or create module-specific logger"""
        
        if module_name not in self.module_loggers:
            # Create module-specific logger
            logger = logging.getLogger(f"app.{module_name}")
            
            # Create module-specific log file
            module_log_path = os.path.join(self.log_dir, "modules", f"{module_name}.log")
            module_handler = RotatingFileHandler(
                module_log_path,
                maxBytes=2*1024*1024,  # 2MB
                backupCount=2
            )
            
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            module_handler.setFormatter(formatter)
            logger.addHandler(module_handler)
            
            self.module_loggers[module_name] = logger
        
        return self.module_loggers[module_name]
    
    def log_system_startup(self):
        """Log system startup information"""
        logger = logging.getLogger("system")
        
        logger.info("="*60)
        logger.info("ENHANCED UNIVERSAL AI TRAINING DATA CREATOR - STARTUP")
        logger.info("="*60)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Startup Time: {datetime.now()}")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Working Directory: {os.getcwd()}")
        logger.info(f"Log Directory: {os.path.abspath(self.log_dir)}")
        logger.info("="*60)
    
    def log_event(self, event_type: str, event_data: Dict[str, Any], module_name: str = "app"):
        """Log structured event with tracking"""
        
        self.event_counter += 1
        
        event_record = {
            'event_id': self.event_counter,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'module': module_name,
            'data': event_data
        }
        
        # Log to main logger
        logger = self.get_module_logger(module_name)
        logger.info(f"EVENT: {event_type} | {json.dumps(event_data)}")
        
        # Log to event-specific handler
        event_logger = logging.getLogger("events")
        event_logger.addHandler(self.event_handler)
        event_logger.info(json.dumps(event_record))
    
    def log_performance(self, operation: str, duration: float, details: Dict[str, Any] = None, module_name: str = "app"):
        """Log performance metrics"""
        
        performance_data = {
            'operation': operation,
            'duration_seconds': duration,
            'module': module_name,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        
        if details:
            performance_data.update(details)
        
        # Log to main logger
        logger = self.get_module_logger(module_name)
        logger.info(f"PERFORMANCE: {operation} completed in {duration:.2f}s")
        
        # Log to performance-specific handler
        perf_logger = logging.getLogger("performance")
        perf_logger.addHandler(self.performance_handler)
        perf_logger.info(json.dumps(performance_data))
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, module_name: str = "app"):
        """Log error with full context and stack trace"""
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': traceback.format_exc(),
            'module': module_name,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        
        if context:
            error_data['context'] = context
        
        # Log to main logger
        logger = self.get_module_logger(module_name)
        logger.error(f"ERROR: {type(error).__name__}: {str(error)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # Log structured error data
        error_logger = logging.getLogger("errors")
        error_logger.error(json.dumps(error_data))
    
    def log_user_action(self, action: str, details: Dict[str, Any] = None):
        """Log user actions for analytics"""
        
        action_data = {
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        
        if details:
            action_data.update(details)
        
        self.log_event("user_action", action_data, "ui")
    
    def log_ai_operation(self, operation: str, input_size: int, output_size: int, duration: float, success: bool):
        """Log AI operations for monitoring"""
        
        ai_data = {
            'operation': operation,
            'input_size': input_size,
            'output_size': output_size,
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        
        self.log_event("ai_operation", ai_data, "ai")
        self.log_performance(f"ai_{operation}", duration, ai_data, "ai")
    
    def get_log_files(self) -> Dict[str, str]:
        """Get paths to all log files"""
        return {
            'main': os.path.join(self.log_dir, self.log_file),
            'errors': os.path.join(self.log_dir, self.error_log_file),
            'performance': os.path.join(self.log_dir, self.performance_log_file),
            'events': os.path.join(self.log_dir, self.event_log_file)
        }
    
    def get_recent_logs(self, log_type: str = "main", lines: int = 100) -> List[str]:
        """Get recent log entries"""
        
        log_files = self.get_log_files()
        log_file_path = log_files.get(log_type)
        
        if not log_file_path or not os.path.exists(log_file_path):
            return []
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        except Exception as e:
            logging.error(f"Failed to read log file {log_file_path}: {str(e)}")
            return []
    
    def search_logs(self, query: str, log_type: str = "main", max_results: int = 50) -> List[str]:
        """Search logs for specific content"""
        
        log_files = self.get_log_files()
        log_file_path = log_files.get(log_type)
        
        if not log_file_path or not os.path.exists(log_file_path):
            return []
        
        try:
            matching_lines = []
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if query.lower() in line.lower():
                        matching_lines.append(line.strip())
                        if len(matching_lines) >= max_results:
                            break
            
            return matching_lines
        
        except Exception as e:
            logging.error(f"Failed to search log file {log_file_path}: {str(e)}")
            return []
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        
        stats = {
            'session_id': self.session_id,
            'events_logged': self.event_counter,
            'log_files': {},
            'startup_time': datetime.now().isoformat()
        }
        
        # Get file sizes and line counts
        log_files = self.get_log_files()
        for log_type, log_path in log_files.items():
            if os.path.exists(log_path):
                try:
                    file_size = os.path.getsize(log_path)
                    with open(log_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    
                    stats['log_files'][log_type] = {
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024*1024), 2),
                        'line_count': line_count,
                        'path': log_path
                    }
                except Exception as e:
                    stats['log_files'][log_type] = {'error': str(e)}
        
        return stats

# Global logger instance
_central_logger = None

def get_logger(module_name: str = "app") -> logging.Logger:
    """
    Get module-specific logger
    
    Usage:
    from modules.logger import get_logger
    logger = get_logger("extractor")
    logger.info("Content extraction started")
    """
    global _central_logger
    
    if _central_logger is None:
        _central_logger = CentralizedLogger()
    
    return _central_logger.get_module_logger(module_name)

def log_event(event_type: str, event_data: Dict[str, Any], module_name: str = "app"):
    """
    Log structured event
    
    Usage:
    from modules.logger import log_event
    log_event("file_uploaded", {"filename": "data.txt", "size": 1024}, "uploader")
    """
    global _central_logger
    
    if _central_logger is None:
        _central_logger = CentralizedLogger()
    
    _central_logger.log_event(event_type, event_data, module_name)

def log_performance(operation: str, duration: float, details: Dict[str, Any] = None, module_name: str = "app"):
    """
    Log performance metrics
    
    Usage:
    from modules.logger import log_performance
    log_performance("content_enhancement", 5.2, {"items": 100}, "enhancer")
    """
    global _central_logger
    
    if _central_logger is None:
        _central_logger = CentralizedLogger()
    
    _central_logger.log_performance(operation, duration, details, module_name)

def log_error(error: Exception, context: Dict[str, Any] = None, module_name: str = "app"):
    """
    Log error with context
    
    Usage:
    from modules.logger import log_error
    try:
        risky_operation()
    except Exception as e:
        log_error(e, {"operation": "file_processing"}, "processor")
    """
    global _central_logger
    
    if _central_logger is None:
        _central_logger = CentralizedLogger()
    
    _central_logger.log_error(error, context, module_name)

def log_user_action(action: str, details: Dict[str, Any] = None):
    """
    Log user actions
    
    Usage:
    from modules.logger import log_user_action
    log_user_action("button_clicked", {"button": "enhance_content"})
    """
    global _central_logger
    
    if _central_logger is None:
        _central_logger = CentralizedLogger()
    
    _central_logger.log_user_action(action, details)

def log_ai_operation(operation: str, input_size: int, output_size: int, duration: float, success: bool):
    """
    Log AI operations
    
    Usage:
    from modules.logger import log_ai_operation
    log_ai_operation("content_enhancement", 1000, 1500, 3.2, True)
    """
    global _central_logger
    
    if _central_logger is None:
        _central_logger = CentralizedLogger()
    
    _central_logger.log_ai_operation(operation, input_size, output_size, duration, success)

# Performance timing decorator
def log_timing(operation_name: str = None, module_name: str = "app"):
    """
    Decorator to automatically log function execution time
    
    Usage:
    from modules.logger import log_timing
    
    @log_timing("content_extraction", "extractor")
    def extract_content(file_path):
        # function implementation
        pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            start_time = time.time()
            operation = operation_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log_performance(operation, duration, {"success": True}, module_name)
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                log_performance(operation, duration, {"success": False, "error": str(e)}, module_name)
                log_error(e, {"operation": operation}, module_name)
                raise
        
        return wrapper
    return decorator

# Context manager for operation logging
class LoggedOperation:
    """
    Context manager for logging operations
    
    Usage:
    from modules.logger import LoggedOperation
    
    with LoggedOperation("file_processing", "processor") as op:
        # do work
        op.add_detail("files_processed", 5)
    """
    
    def __init__(self, operation: str, module_name: str = "app"):
        self.operation = operation
        self.module_name = module_name
        self.start_time = None
        self.details = {}
        self.logger = get_logger(module_name)
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            # Success
            self.logger.info(f"Completed operation: {self.operation} in {duration:.2f}s")
            log_performance(self.operation, duration, self.details, self.module_name)
        else:
            # Error
            self.logger.error(f"Failed operation: {self.operation} after {duration:.2f}s")
            log_error(exc_val, {"operation": self.operation, **self.details}, self.module_name)
    
    def add_detail(self, key: str, value: Any):
        """Add detail to operation logging"""
        self.details[key] = value
    
    def log_progress(self, message: str):
        """Log progress during operation"""
        self.logger.info(f"{self.operation} progress: {message}")

# Initialize logging system
def initialize_logging():
    """Initialize the centralized logging system"""
    global _central_logger
    
    if _central_logger is None:
        _central_logger = CentralizedLogger()
    
    return _central_logger

# Example integration patterns for modules
class ModuleLoggerMixin:
    """
    Mixin class to add logging capabilities to any module
    
    Usage:
    class ContentExtractor(ModuleLoggerMixin):
        def __init__(self):
            super().__init__()
            self.setup_module_logging("extractor")
        
        def extract_content(self, file_path):
            self.logger.info(f"Extracting content from {file_path}")
            # implementation
    """
    
    def setup_module_logging(self, module_name: str):
        """Setup logging for the module"""
        self.module_name = module_name
        self.logger = get_logger(module_name)
    
    def log_operation_start(self, operation: str, details: Dict[str, Any] = None):
        """Log operation start"""
        self.logger.info(f"Starting {operation}")
        if details:
            log_event(f"{operation}_started", details, self.module_name)
    
    def log_operation_success(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """Log successful operation"""
        self.logger.info(f"Completed {operation} in {duration:.2f}s")
        log_performance(operation, duration, details, self.module_name)
    
    def log_operation_error(self, operation: str, error: Exception, context: Dict[str, Any] = None):
        """Log operation error"""
        self.logger.error(f"Failed {operation}: {str(error)}")
        log_error(error, {"operation": operation, **(context or {})}, self.module_name)

if __name__ == "__main__":
    # Test the logging system
    logger = get_logger("test")
    
    logger.info("Testing centralized logging system")
    log_event("test_event", {"test": True}, "test")
    log_performance("test_operation", 1.5, {"items": 10}, "test")
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error(e, {"test_context": True}, "test")
    
    print("Logging test completed. Check logs/ directory for output.")

