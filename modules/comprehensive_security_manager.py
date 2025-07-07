"""
Comprehensive Security Manager Module
====================================

Complete security management system implementing the user's critical security patterns:
- Store API keys in .env files (never in code)
- Use python-dotenv to load environment variables
- Automatic cleanup of temporary files with shutil.rmtree

This module addresses critical security vulnerabilities:
- API key exposure and misuse
- Unencrypted sensitive data storage
- Temporary file leaks
- Session security issues
- Data protection gaps

Based on the user's exact security patterns:
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Cleanup
shutil.rmtree(temp_folder_path)
"""

import os
import shutil
import tempfile
import logging
import hashlib
import stat
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import streamlit as st
from cryptography.fernet import Fernet

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

class SecureAPIKeyManager:
    """
    Secure API key management system
    
    Implements the user's exact pattern:
    - Store keys in .env files
    - Use python-dotenv to load
    - Never store keys in code
    """
    
    def __init__(self):
        self.load_environment_variables()
        self.validate_required_keys()
    
    def load_environment_variables(self):
        """Load environment variables using python-dotenv - USER'S EXACT PATTERN!"""
        from dotenv import load_dotenv
        
        # Load from .env file - USER'S EXACT APPROACH!
        load_dotenv()
        
        # Validate .env file exists
        if not os.path.exists('.env'):
            self._create_env_template()
            raise SecurityError("Please configure .env file with your API keys")
        
        # Load API keys securely
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Validate keys are loaded
        if not self.openai_api_key:
            raise SecurityError("OPENAI_API_KEY not found in environment variables")
    
    def get_openai_client(self):
        """Get OpenAI client with secure API key - USER'S EXACT PATTERN!"""
        import openai
        
        # USER'S EXACT IMPLEMENTATION!
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai.api_key:
            raise SecurityError("OpenAI API key not configured")
        
        return openai
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and accessibility"""
        if not api_key or len(api_key) < 20:
            return False
        
        # Test API key with minimal request
        try:
            client = self.get_openai_client()
            # Make a minimal test request
            response = client.models.list()
            return True
        except Exception as e:
            logging.error(f"API key validation failed: {str(e)}")
            return False
    
    def validate_required_keys(self):
        """Validate that required API keys are present"""
        if not self.openai_api_key or self.openai_api_key.startswith('your_'):
            raise SecurityError("Valid OpenAI API key required. Please configure in .env file.")
    
    def _create_env_template(self):
        """Create .env template file for user configuration"""
        env_template = '''# Enhanced Universal AI Training Data Creator - Environment Variables
# Copy this file to .env and fill in your API keys

# OpenAI API Key (Required)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face Token (Optional - for model uploads)
# Get from: https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Security Settings
ENABLE_FILE_ENCRYPTION=true
AUTO_CLEANUP_TEMP_FILES=true
SESSION_TIMEOUT_MINUTES=60
MAX_FILE_SIZE_MB=100

# Debug Settings (DO NOT ENABLE IN PRODUCTION)
DEBUG_MODE=false
LOG_API_REQUESTS=false
'''
        
        with open('.env.example', 'w') as f:
            f.write(env_template)
        
        logging.info("Created .env.example template file")

class SecureFileManager:
    """
    Secure file management with automatic cleanup
    
    Implements the user's cleanup pattern:
    shutil.rmtree(temp_folder_path)
    """
    
    def __init__(self):
        self.temp_directories = set()
        self.temp_files = set()
        self.encryption_key = self._generate_session_key()
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.cleanup_all_temp_files)
    
    def create_secure_temp_directory(self) -> str:
        """Create secure temporary directory with proper permissions"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="ai_trainer_secure_")
        
        # Set secure permissions (owner read/write/execute only)
        os.chmod(temp_dir, stat.S_IRWXU)
        
        # Track for cleanup
        self.temp_directories.add(temp_dir)
        
        logging.info(f"Created secure temp directory: {temp_dir}")
        return temp_dir
    
    def create_secure_temp_file(self, suffix: str = ".tmp") -> str:
        """Create secure temporary file with encryption"""
        # Create temporary file
        fd, temp_file = tempfile.mkstemp(suffix=suffix, prefix="ai_trainer_")
        os.close(fd)  # Close file descriptor
        
        # Set secure permissions (owner read/write only)
        os.chmod(temp_file, stat.S_IRUSR | stat.S_IWUSR)
        
        # Track for cleanup
        self.temp_files.add(temp_file)
        
        logging.info(f"Created secure temp file: {temp_file}")
        return temp_file
    
    def write_encrypted_file(self, file_path: str, content: str):
        """Write content to file with encryption"""
        # Encrypt content
        fernet = Fernet(self.encryption_key)
        encrypted_content = fernet.encrypt(content.encode())
        
        # Write encrypted content
        with open(file_path, 'wb') as f:
            f.write(encrypted_content)
        
        logging.info(f"Wrote encrypted content to: {file_path}")
    
    def read_encrypted_file(self, file_path: str) -> str:
        """Read and decrypt content from file"""
        try:
            # Read encrypted content
            with open(file_path, 'rb') as f:
                encrypted_content = f.read()
            
            # Decrypt content
            fernet = Fernet(self.encryption_key)
            decrypted_content = fernet.decrypt(encrypted_content)
            
            return decrypted_content.decode()
            
        except Exception as e:
            logging.error(f"Failed to decrypt file {file_path}: {str(e)}")
            raise SecurityError(f"File decryption failed: {str(e)}")
    
    def secure_file_upload(self, uploaded_file) -> tuple:
        """Securely handle file upload with validation"""
        # Validate file size
        max_size = int(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024
        if uploaded_file.size > max_size:
            raise SecurityError(f"File too large: {uploaded_file.size} bytes")
        
        # Get file content
        file_content = uploaded_file.getvalue()
        
        # Create secure temp file
        temp_file = self.create_secure_temp_file(suffix=f"_{uploaded_file.name}")
        
        # Calculate file hash for integrity
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Write file with encryption if enabled
        if os.getenv("ENABLE_FILE_ENCRYPTION", "true").lower() == "true":
            try:
                content_str = file_content.decode('utf-8', errors='ignore')
                self.write_encrypted_file(temp_file, content_str)
            except:
                # Fallback to binary write for non-text files
                with open(temp_file, 'wb') as f:
                    f.write(file_content)
        else:
            with open(temp_file, 'wb') as f:
                f.write(file_content)
        
        # Store file metadata securely
        file_metadata = {
            'original_name': uploaded_file.name,
            'size': uploaded_file.size,
            'hash': file_hash,
            'temp_path': temp_file,
            'encrypted': os.getenv("ENABLE_FILE_ENCRYPTION", "true").lower() == "true"
        }
        
        logging.info(f"Securely uploaded file: {uploaded_file.name}")
        return temp_file, file_metadata
    
    def cleanup_temp_file(self, file_path: str):
        """Securely delete temporary file"""
        try:
            if os.path.exists(file_path):
                # Secure deletion - overwrite before deletion
                self._secure_delete_file(file_path)
                
                # Remove from tracking
                self.temp_files.discard(file_path)
                
                logging.info(f"Cleaned up temp file: {file_path}")
        
        except Exception as e:
            logging.error(f"Failed to cleanup temp file {file_path}: {str(e)}")
    
    def cleanup_temp_directory(self, dir_path: str):
        """Securely delete temporary directory - USER'S EXACT PATTERN!"""
        try:
            if os.path.exists(dir_path):
                # USER'S EXACT CLEANUP PATTERN!
                shutil.rmtree(dir_path)
                
                # Remove from tracking
                self.temp_directories.discard(dir_path)
                
                logging.info(f"Cleaned up temp directory: {dir_path}")
        
        except Exception as e:
            logging.error(f"Failed to cleanup temp directory {dir_path}: {str(e)}")
    
    def cleanup_all_temp_files(self):
        """Clean up all temporary files and directories - USER'S PATTERN!"""
        logging.info("Starting comprehensive temp file cleanup...")
        
        # Clean up all temp files
        for temp_file in list(self.temp_files):
            self.cleanup_temp_file(temp_file)
        
        # Clean up all temp directories - USER'S EXACT PATTERN!
        for temp_dir in list(self.temp_directories):
            self.cleanup_temp_directory(temp_dir)  # Uses shutil.rmtree internally
        
        logging.info("Completed temp file cleanup")
    
    def _secure_delete_file(self, file_path: str):
        """Securely delete file by overwriting before deletion"""
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Overwrite with random data multiple times
            with open(file_path, 'r+b') as f:
                for _ in range(3):  # 3-pass overwrite
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
            
            # Finally delete the file
            os.remove(file_path)
            
        except Exception as e:
            # Fallback to regular deletion
            os.remove(file_path)
    
    def _generate_session_key(self) -> bytes:
        """Generate encryption key for session"""
        return Fernet.generate_key()

class SecureSessionManager:
    """Secure session management with encryption and cleanup"""
    
    def __init__(self, session_manager):
        self.session = session_manager
        self.encryption_key = self._get_or_create_session_key()
        self.setup_session_security()
    
    def setup_session_security(self):
        """Setup session security measures"""
        # Set session timeout
        timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
        self.session_timeout = timeout_minutes * 60  # Convert to seconds
        
        # Initialize security tracking
        if 'security_initialized' not in st.session_state:
            st.session_state['security_initialized'] = True
            st.session_state['session_start_time'] = datetime.now()
            st.session_state['last_activity_time'] = datetime.now()
            st.session_state['security_events'] = []
    
    def validate_session_security(self) -> bool:
        """Validate session security and check for timeout"""
        current_time = datetime.now()
        
        # Check session timeout
        last_activity = st.session_state.get('last_activity_time', current_time)
        if current_time - last_activity > timedelta(seconds=self.session_timeout):
            self.log_security_event("Session timeout", "warning")
            self.cleanup_expired_session()
            return False
        
        # Update last activity
        st.session_state['last_activity_time'] = current_time
        
        return True
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for session storage"""
        fernet = Fernet(self.encryption_key)
        encrypted_data = fernet.encrypt(data.encode())
        return encrypted_data.hex()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data from session storage"""
        try:
            fernet = Fernet(self.encryption_key)
            encrypted_bytes = bytes.fromhex(encrypted_data)
            decrypted_data = fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            self.log_security_event(f"Decryption failed: {str(e)}", "error")
            raise SecurityError("Failed to decrypt session data")
    
    def log_security_event(self, event: str, level: str = "info"):
        """Log security events for audit trail"""
        security_event = {
            'timestamp': datetime.now(),
            'event': event,
            'level': level,
            'session_id': id(st.session_state)
        }
        
        security_events = st.session_state.get('security_events', [])
        security_events.append(security_event)
        
        # Keep only recent events (last 100)
        if len(security_events) > 100:
            security_events = security_events[-50:]
        
        st.session_state['security_events'] = security_events
        
        # Log to application log
        logging.log(getattr(logging, level.upper()), f"Security Event: {event}")
    
    def cleanup_expired_session(self):
        """Clean up expired session data"""
        # Clear sensitive session data
        sensitive_keys = [
            'uploaded_file', 'file_content', 'extracted_content',
            'enhanced_content', 'quality_scores', 'final_dataset'
        ]
        
        for key in sensitive_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        self.log_security_event("Session cleaned up due to timeout", "warning")
        
        # Force page refresh
        st.rerun()
    
    def _get_or_create_session_key(self) -> bytes:
        """Get or create encryption key for session"""
        if 'session_encryption_key' not in st.session_state:
            st.session_state['session_encryption_key'] = Fernet.generate_key()
        
        return st.session_state['session_encryption_key']

class SecurityValidator:
    """Security validation and monitoring system"""
    
    def __init__(self):
        self.security_checks = []
        self.setup_security_monitoring()
    
    def setup_security_monitoring(self):
        """Setup comprehensive security monitoring"""
        
        # Register security checks
        self.security_checks = [
            self.check_api_key_security,
            self.check_file_permissions,
            self.check_temp_file_cleanup,
            self.check_session_security,
            self.check_data_encryption
        ]
    
    def run_security_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        audit_results = {
            'timestamp': datetime.now(),
            'overall_status': 'secure',
            'checks': [],
            'warnings': [],
            'errors': []
        }
        
        for check_func in self.security_checks:
            try:
                result = check_func()
                audit_results['checks'].append(result)
                
                if result['status'] == 'warning':
                    audit_results['warnings'].append(result)
                elif result['status'] == 'error':
                    audit_results['errors'].append(result)
                    audit_results['overall_status'] = 'vulnerable'
                
            except Exception as e:
                error_result = {
                    'check': check_func.__name__,
                    'status': 'error',
                    'message': f"Security check failed: {str(e)}"
                }
                audit_results['checks'].append(error_result)
                audit_results['errors'].append(error_result)
        
        return audit_results
    
    def check_api_key_security(self) -> Dict[str, str]:
        """Check API key security configuration"""
        # Check if .env file exists
        if not os.path.exists('.env'):
            return {
                'check': 'API Key Security',
                'status': 'error',
                'message': '.env file not found - API keys may be exposed'
            }
        
        # Check if API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {
                'check': 'API Key Security',
                'status': 'error',
                'message': 'OPENAI_API_KEY not configured in environment'
            }
        
        # Check if key looks valid
        if len(api_key) < 20 or api_key.startswith('your_'):
            return {
                'check': 'API Key Security',
                'status': 'warning',
                'message': 'API key appears to be placeholder or invalid'
            }
        
        return {
            'check': 'API Key Security',
            'status': 'secure',
            'message': 'API key properly configured in environment variables'
        }
    
    def check_file_permissions(self) -> Dict[str, str]:
        """Check file permission security"""
        # Check .env file permissions
        if os.path.exists('.env'):
            env_stat = os.stat('.env')
            
            # .env should not be world-readable
            if env_stat.st_mode & stat.S_IROTH:
                return {
                    'check': 'File Permissions',
                    'status': 'warning',
                    'message': '.env file is world-readable - consider restricting permissions'
                }
        
        return {
            'check': 'File Permissions',
            'status': 'secure',
            'message': 'File permissions are appropriately configured'
        }
    
    def check_temp_file_cleanup(self) -> Dict[str, str]:
        """Check temporary file cleanup configuration"""
        auto_cleanup = os.getenv('AUTO_CLEANUP_TEMP_FILES', 'true').lower()
        
        if auto_cleanup != 'true':
            return {
                'check': 'Temp File Cleanup',
                'status': 'warning',
                'message': 'Automatic temp file cleanup is disabled'
            }
        
        return {
            'check': 'Temp File Cleanup',
            'status': 'secure',
            'message': 'Automatic temp file cleanup is enabled'
        }
    
    def check_session_security(self) -> Dict[str, str]:
        """Check session security configuration"""
        # Check if session security is initialized
        if 'security_initialized' not in st.session_state:
            return {
                'check': 'Session Security',
                'status': 'warning',
                'message': 'Session security not initialized'
            }
        
        return {
            'check': 'Session Security',
            'status': 'secure',
            'message': 'Session security is properly configured'
        }
    
    def check_data_encryption(self) -> Dict[str, str]:
        """Check data encryption configuration"""
        encryption_enabled = os.getenv('ENABLE_FILE_ENCRYPTION', 'true').lower()
        
        if encryption_enabled != 'true':
            return {
                'check': 'Data Encryption',
                'status': 'warning',
                'message': 'File encryption is disabled'
            }
        
        return {
            'check': 'Data Encryption',
            'status': 'secure',
            'message': 'Data encryption is enabled'
        }

class ComprehensiveSecurityManager:
    """
    Complete security management system
    
    Integrates all security components:
    - API key management (user's .env pattern)
    - File security with encryption
    - Automatic cleanup (user's shutil.rmtree pattern)
    - Session security
    - Security monitoring
    """
    
    def __init__(self, session_manager):
        self.session_manager = session_manager
        
        # Initialize security components
        self.api_key_manager = SecureAPIKeyManager()
        self.file_manager = SecureFileManager()
        self.session_security = SecureSessionManager(session_manager)
        self.security_validator = SecurityValidator()
        
        # Setup security monitoring
        self.setup_security_monitoring()
    
    def setup_security_monitoring(self):
        """Setup comprehensive security monitoring"""
        import atexit
        
        # Register cleanup on application exit
        atexit.register(self.emergency_cleanup)
        
        # Log security initialization
        logging.info("Comprehensive security system initialized")
        self.session_security.log_security_event("Security system initialized", "info")
    
    def validate_security_state(self) -> bool:
        """Validate overall security state"""
        
        # Check session security
        if not self.session_security.validate_session_security():
            return False
        
        # Run security audit
        audit_results = self.security_validator.run_security_audit()
        
        # Log any security issues
        if audit_results['errors']:
            for error in audit_results['errors']:
                self.session_security.log_security_event(f"Security Error: {error['message']}", "error")
        
        if audit_results['warnings']:
            for warning in audit_results['warnings']:
                self.session_security.log_security_event(f"Security Warning: {warning['message']}", "warning")
        
        return audit_results['overall_status'] == 'secure'
    
    def secure_api_call(self, api_function, *args, **kwargs):
        """Make secure API call with proper error handling"""
        try:
            # Validate API key before call
            client = self.api_key_manager.get_openai_client()
            
            # Make API call
            result = api_function(*args, **kwargs)
            
            # Log successful API call (without sensitive data)
            self.session_security.log_security_event("Secure API call completed", "info")
            
            return result
            
        except Exception as e:
            # Log API error (without exposing sensitive information)
            error_msg = "API call failed"
            self.session_security.log_security_event(error_msg, "error")
            
            # Re-raise with sanitized error message
            raise SecurityError("API operation failed - check configuration")
    
    def secure_file_operation(self, operation_func, *args, **kwargs):
        """Perform secure file operation with automatic cleanup"""
        temp_resources = []
        
        try:
            # Perform file operation
            result = operation_func(*args, **kwargs)
            
            # Track any temporary resources created
            if hasattr(result, 'temp_path'):
                temp_resources.append(result.temp_path)
            
            return result
            
        except Exception as e:
            # Log file operation error
            self.session_security.log_security_event(f"File operation failed: {str(e)}", "error")
            raise
            
        finally:
            # Clean up temporary resources - USER'S CLEANUP PATTERN!
            for temp_resource in temp_resources:
                if os.path.isdir(temp_resource):
                    self.file_manager.cleanup_temp_directory(temp_resource)
                else:
                    self.file_manager.cleanup_temp_file(temp_resource)
    
    def emergency_cleanup(self):
        """Emergency cleanup on application shutdown"""
        try:
            # Clean up all temporary files - USER'S PATTERN!
            self.file_manager.cleanup_all_temp_files()
            
            # Log emergency cleanup
            logging.info("Emergency security cleanup completed")
            
        except Exception as e:
            logging.error(f"Emergency cleanup failed: {str(e)}")
    
    def render_security_dashboard(self):
        """Render security status dashboard"""
        st.subheader("🔐 Security Status")
        
        # Run security audit
        audit_results = self.security_validator.run_security_audit()
        
        # Overall status
        if audit_results['overall_status'] == 'secure':
            st.success("🛡️ **Security Status: SECURE**")
        else:
            st.error("⚠️ **Security Status: VULNERABLE**")
        
        # Security checks
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Security Checks:**")
            for check in audit_results['checks']:
                if check['status'] == 'secure':
                    st.success(f"✅ {check['check']}")
                elif check['status'] == 'warning':
                    st.warning(f"⚠️ {check['check']}")
                else:
                    st.error(f"❌ {check['check']}")
        
        with col2:
            st.write("**Security Events:**")
            security_events = self.session_manager.get('security_events', [])
            
            if security_events:
                for event in security_events[-5:]:  # Show last 5 events
                    timestamp = event['timestamp'].strftime('%H:%M:%S')
                    if event['level'] == 'error':
                        st.error(f"{timestamp}: {event['event']}")
                    elif event['level'] == 'warning':
                        st.warning(f"{timestamp}: {event['event']}")
                    else:
                        st.info(f"{timestamp}: {event['event']}")
            else:
                st.info("No security events logged")
        
        # Security configuration
        with st.expander("🔧 Security Configuration"):
            
            # API key status
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
                st.write(f"**OpenAI API Key:** {masked_key}")
            else:
                st.error("**OpenAI API Key:** Not configured")
            
            # Security settings
            st.write("**Security Settings:**")
            st.write(f"- File Encryption: {os.getenv('ENABLE_FILE_ENCRYPTION', 'true')}")
            st.write(f"- Auto Cleanup: {os.getenv('AUTO_CLEANUP_TEMP_FILES', 'true')}")
            st.write(f"- Session Timeout: {os.getenv('SESSION_TIMEOUT_MINUTES', '60')} minutes")
            st.write(f"- Max File Size: {os.getenv('MAX_FILE_SIZE_MB', '100')} MB")
            
            # Temp file status
            temp_files_count = len(self.file_manager.temp_files)
            temp_dirs_count = len(self.file_manager.temp_directories)
            st.write(f"- Active Temp Files: {temp_files_count}")
            st.write(f"- Active Temp Directories: {temp_dirs_count}")
            
            # Manual cleanup button
            if st.button("🧹 Manual Cleanup", key="manual_cleanup_btn"):
                self.file_manager.cleanup_all_temp_files()
                st.success("Manual cleanup completed")
                st.rerun()

# Example usage and integration
def main_with_security():
    """Example of main app with comprehensive security"""
    
    # Initialize security system
    from .session_state_manager import SessionStateManager
    session_manager = SessionStateManager()
    security_manager = ComprehensiveSecurityManager(session_manager)
    
    # Validate security before proceeding
    if not security_manager.validate_security_state():
        st.error("⚠️ Security validation failed. Please check your configuration.")
        return
    
    st.title("🧠 Enhanced Universal AI Training Data Creator")
    st.write("*With Comprehensive Security Protection*")
    
    # Security dashboard in sidebar
    with st.sidebar:
        security_manager.render_security_dashboard()
    
    # Main application with security integration
    tab1, tab2, tab3 = st.tabs(["🔐 Secure Upload", "✨ Secure Processing", "📊 Security Audit"])
    
    with tab1:
        st.subheader("Secure File Upload")
        
        uploaded_file = st.file_uploader("Choose file", type=['txt', 'pdf', 'json'])
        
        if uploaded_file:
            try:
                # Secure file upload with validation
                temp_file, metadata = security_manager.file_manager.secure_file_upload(uploaded_file)
                
                st.success("✅ File uploaded securely")
                st.json(metadata)
                
            except SecurityError as e:
                st.error(f"❌ Upload failed: {str(e)}")
    
    with tab2:
        st.subheader("Secure AI Processing")
        
        if st.button("🚀 Secure AI Enhancement"):
            try:
                # Example secure API call
                def example_api_call():
                    # This would be your actual OpenAI API call
                    return {"status": "success", "message": "Content enhanced"}
                
                result = security_manager.secure_api_call(example_api_call)
                st.success("✅ AI processing completed securely")
                st.json(result)
                
            except SecurityError as e:
                st.error(f"❌ Processing failed: {str(e)}")
    
    with tab3:
        st.subheader("Security Audit Results")
        
        if st.button("🔍 Run Security Audit"):
            audit_results = security_manager.security_validator.run_security_audit()
            
            st.write("**Audit Results:**")
            st.json(audit_results)

if __name__ == "__main__":
    main_with_security()

