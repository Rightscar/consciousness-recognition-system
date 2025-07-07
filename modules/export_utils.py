"""
Export Utilities Module
======================

Isolated export utilities with comprehensive error handling and validation.
Implements the user's exact reliability patterns:
- Isolated export_utils.py module
- Try/except wrapping every export block
- File size validation before uploads
- Comprehensive logging of failures

Based on the user's exact pattern:
try:
    generate_zip()
    if os.path.getsize("file.zip") < 50_000_000:
        upload_to_hf()
except Exception as e:
    st.error(f"Export failed: {e}")
"""

import os
import json
import shutil
import tempfile
import logging
import time
import zipfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import streamlit as st

class ExportError(Exception):
    """Custom exception for export-related errors"""
    pass

class RobustExportManager:
    """
    Robust export manager with comprehensive error handling
    
    Implements the user's exact pattern:
    - Try/except wrapping every export block
    - File size validation before uploads
    - Comprehensive logging of failures
    """
    
    def __init__(self):
        self.setup_export_logging()
        self.max_file_size = 50_000_000  # USER'S EXACT SIZE LIMIT!
        self.temp_files = set()
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.cleanup_temp_files)
    
    def setup_export_logging(self):
        """Setup export-specific logging"""
        self.logger = logging.getLogger('export_utils')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def export_with_validation(self, data: List[Dict], format_type: str, output_path: str) -> bool:
        """
        Export data with comprehensive validation - USER'S EXACT PATTERN!
        
        Implements user's try/except pattern:
        try:
            # export operations
        except Exception as e:
            st.error(f"Export failed: {e}")
        """
        try:
            # Pre-export validation
            self._validate_export_data(data)
            self._validate_output_path(output_path)
            
            # Generate export based on format
            if format_type == 'jsonl':
                self._generate_jsonl_export(data, output_path)
            elif format_type == 'json':
                self._generate_json_export(data, output_path)
            elif format_type == 'csv':
                self._generate_csv_export(data, output_path)
            elif format_type == 'xlsx':
                self._generate_xlsx_export(data, output_path)
            else:
                raise ExportError(f"Unsupported format: {format_type}")
            
            # Post-export validation
            self._validate_generated_file(output_path, format_type)
            
            # Track temp file for cleanup
            self.temp_files.add(output_path)
            
            self.logger.info(f"Export successful: {output_path}")
            return True
            
        except Exception as e:
            # USER'S EXACT ERROR HANDLING PATTERN!
            error_msg = f"Export failed: {e}"
            self.logger.error(error_msg)
            st.error(error_msg)  # USER'S EXACT ERROR DISPLAY!
            return False
    
    def generate_zip_with_validation(self, files_dict: Dict[str, str], zip_path: str) -> bool:
        """
        Generate ZIP with validation - USER'S EXACT PATTERN!
        
        Implements user's exact pattern:
        try:
            generate_zip()
            if os.path.getsize("file.zip") < 50_000_000:
                # proceed
        except Exception as e:
            st.error(f"Export failed: {e}")
        """
        try:
            # USER'S EXACT PATTERN IMPLEMENTATION!
            self._generate_zip(files_dict, zip_path)
            
            # USER'S EXACT SIZE VALIDATION!
            if os.path.getsize(zip_path) < self.max_file_size:  # USER'S EXACT THRESHOLD!
                self.logger.info(f"ZIP generated successfully: {zip_path}")
                self.temp_files.add(zip_path)
                return True
            else:
                raise ExportError(f"ZIP file too large: {os.path.getsize(zip_path)} bytes")
                
        except Exception as e:
            # USER'S EXACT ERROR HANDLING!
            error_msg = f"ZIP generation failed: {e}"
            self.logger.error(error_msg)
            st.error(error_msg)
            return False
    
    def upload_to_huggingface_with_validation(self, file_path: str, repo_name: str) -> bool:
        """
        Upload to Hugging Face with validation - USER'S EXACT PATTERN!
        
        Implements user's exact size check and upload pattern:
        if os.path.getsize("file.zip") < 50_000_000:
            upload_to_hf()
        """
        try:
            # USER'S EXACT SIZE CHECK PATTERN!
            if os.path.getsize(file_path) >= self.max_file_size:
                raise ExportError(f"File too large for upload: {os.path.getsize(file_path)} bytes")
            
            # Validate file integrity
            self._validate_file_integrity(file_path)
            
            # Perform upload with retry mechanism
            self._upload_to_hf_with_retry(file_path, repo_name)
            
            self.logger.info(f"Upload successful: {repo_name}")
            return True
            
        except Exception as e:
            # USER'S EXACT ERROR HANDLING!
            error_msg = f"Upload failed: {e}"
            self.logger.error(error_msg)
            st.error(error_msg)
            return False
    
    def _validate_export_data(self, data: List[Dict]):
        """Validate data before export"""
        if not data:
            raise ExportError("No data provided for export")
        
        if not isinstance(data, list):
            raise ExportError("Data must be a list of dictionaries")
        
        # Validate data structure
        for i, item in enumerate(data[:10]):  # Check first 10 items
            if not isinstance(item, dict):
                raise ExportError(f"Item {i} is not a dictionary")
            
            if 'question' not in item or 'answer' not in item:
                raise ExportError(f"Item {i} missing required fields (question, answer)")
    
    def _validate_output_path(self, output_path: str):
        """Validate output path"""
        output_dir = os.path.dirname(output_path)
        
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                raise ExportError(f"Cannot create output directory: {e}")
        
        if not os.access(output_dir, os.W_OK):
            raise ExportError(f"No write permission for directory: {output_dir}")
    
    def _generate_jsonl_export(self, data: List[Dict], output_path: str):
        """Generate JSONL export with robust error handling"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    # Sanitize item before writing
                    sanitized_item = self._sanitize_json_item(item)
                    json_line = json.dumps(sanitized_item, ensure_ascii=False)
                    f.write(json_line + '\n')
            
        except Exception as e:
            raise ExportError(f"JSONL export failed: {str(e)}")
    
    def _generate_json_export(self, data: List[Dict], output_path: str):
        """Generate JSON export with robust error handling"""
        try:
            # Sanitize data before export
            sanitized_data = [self._sanitize_json_item(item) for item in data]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            raise ExportError(f"JSON export failed: {str(e)}")
    
    def _generate_csv_export(self, data: List[Dict], output_path: str):
        """Generate CSV export with robust error handling"""
        try:
            import pandas as pd
            
            # Convert to DataFrame with error handling
            df = pd.DataFrame(data)
            
            # Sanitize DataFrame for CSV export
            df = self._sanitize_dataframe_for_csv(df)
            
            # Export with proper encoding and escaping
            df.to_csv(output_path, index=False, encoding='utf-8', escapechar='\\')
            
        except ImportError:
            raise ExportError("pandas is required for CSV export")
        except Exception as e:
            raise ExportError(f"CSV export failed: {str(e)}")
    
    def _generate_xlsx_export(self, data: List[Dict], output_path: str):
        """Generate Excel export with robust error handling"""
        try:
            import pandas as pd
            
            # Convert to DataFrame with error handling
            df = pd.DataFrame(data)
            
            # Sanitize DataFrame for Excel export
            df = self._sanitize_dataframe_for_excel(df)
            
            # Export with proper formatting
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Training_Data')
            
        except ImportError:
            raise ExportError("pandas and openpyxl are required for Excel export")
        except Exception as e:
            raise ExportError(f"Excel export failed: {str(e)}")
    
    def _sanitize_json_item(self, item: Dict) -> Dict:
        """Sanitize item for JSON export"""
        sanitized = {}
        
        for key, value in item.items():
            # Sanitize key
            clean_key = str(key).replace('\n', ' ').replace('\r', ' ').strip()
            
            # Sanitize value
            if isinstance(value, str):
                clean_value = value.replace('\x00', '').strip()
            elif isinstance(value, (int, float, bool)):
                clean_value = value
            elif value is None:
                clean_value = ""
            else:
                clean_value = str(value)
            
            sanitized[clean_key] = clean_value
        
        return sanitized
    
    def _sanitize_dataframe_for_csv(self, df):
        """Sanitize DataFrame for CSV export"""
        import pandas as pd
        
        # Replace problematic characters
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
            df[col] = df[col].str.replace('\x00', '').str.strip()
        
        return df
    
    def _sanitize_dataframe_for_excel(self, df):
        """Sanitize DataFrame for Excel export"""
        import pandas as pd
        
        # Excel has specific limitations
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.replace('\x00', '')
            # Truncate very long cells (Excel limit is ~32,767 characters)
            df[col] = df[col].str[:32000]
        
        return df
    
    def _generate_zip(self, files_dict: Dict[str, str], zip_path: str):
        """Generate ZIP file with compression"""
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for archive_name, file_path in files_dict.items():
                    if os.path.exists(file_path):
                        zipf.write(file_path, archive_name)
                    else:
                        self.logger.warning(f"File not found for ZIP: {file_path}")
            
            # Validate ZIP integrity
            self._validate_zip_integrity(zip_path)
            
        except Exception as e:
            raise ExportError(f"ZIP generation failed: {str(e)}")
    
    def _validate_zip_integrity(self, zip_path: str):
        """Validate ZIP file integrity"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # Test ZIP integrity
                bad_file = zipf.testzip()
                if bad_file:
                    raise ExportError(f"Corrupted file in ZIP: {bad_file}")
                
                # Verify all files are present
                file_list = zipf.namelist()
                if not file_list:
                    raise ExportError("ZIP file is empty")
                
        except zipfile.BadZipFile:
            raise ExportError("Generated ZIP file is corrupted")
        except Exception as e:
            raise ExportError(f"ZIP validation failed: {str(e)}")
    
    def _validate_generated_file(self, file_path: str, format_type: str):
        """Validate generated file"""
        if not os.path.exists(file_path):
            raise ExportError(f"Generated file not found: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ExportError(f"Generated file is empty: {file_path}")
        
        # Format-specific validation
        if format_type == 'jsonl':
            self._validate_jsonl_file(file_path)
        elif format_type == 'json':
            self._validate_json_file(file_path)
        elif format_type == 'csv':
            self._validate_csv_file(file_path)
        elif format_type == 'xlsx':
            self._validate_xlsx_file(file_path)
    
    def _validate_jsonl_file(self, file_path: str):
        """Validate generated JSONL file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # Skip empty lines
                        json.loads(line)  # Validate JSON
        except Exception as e:
            raise ExportError(f"Invalid JSONL file generated: {str(e)}")
    
    def _validate_json_file(self, file_path: str):
        """Validate generated JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)  # Validate JSON
        except Exception as e:
            raise ExportError(f"Invalid JSON file generated: {str(e)}")
    
    def _validate_csv_file(self, file_path: str):
        """Validate generated CSV file"""
        try:
            import pandas as pd
            pd.read_csv(file_path)  # Validate CSV
        except ImportError:
            # Basic validation without pandas
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line.strip():
                    raise ExportError("CSV file appears to be empty")
        except Exception as e:
            raise ExportError(f"Invalid CSV file generated: {str(e)}")
    
    def _validate_xlsx_file(self, file_path: str):
        """Validate generated Excel file"""
        try:
            import pandas as pd
            pd.read_excel(file_path)  # Validate Excel
        except ImportError:
            # Basic validation without pandas
            try:
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as zipf:
                    # Excel files are ZIP archives
                    if 'xl/workbook.xml' not in zipf.namelist():
                        raise ExportError("Invalid Excel file structure")
            except:
                raise ExportError("Cannot validate Excel file without pandas")
        except Exception as e:
            raise ExportError(f"Invalid Excel file generated: {str(e)}")
    
    def _validate_file_integrity(self, file_path: str):
        """Validate file integrity before upload"""
        if not os.path.exists(file_path):
            raise ExportError(f"File not found: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ExportError(f"File is empty: {file_path}")
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Read first 1KB to test
        except Exception as e:
            raise ExportError(f"File is not readable: {str(e)}")
    
    def _upload_to_hf_with_retry(self, file_path: str, repo_name: str):
        """Upload to Hugging Face with retry mechanism"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Validate authentication first
                self._validate_hf_authentication()
                
                # Perform upload
                from huggingface_hub import HfApi
                api = HfApi()
                
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=repo_name,
                    repo_type="dataset"
                )
                
                return  # Success
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Upload attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    raise ExportError(f"Upload failed after {max_retries} attempts: {str(e)}")
    
    def _validate_hf_authentication(self):
        """Validate Hugging Face authentication"""
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            # Test authentication with a simple API call
            user_info = api.whoami()
            
            if not user_info:
                raise ExportError("Hugging Face authentication failed")
                
        except ImportError:
            raise ExportError("huggingface_hub is required for uploads")
        except Exception as e:
            raise ExportError(f"Hugging Face authentication error: {str(e)}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in list(self.temp_files):
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.info(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp file {temp_file}: {str(e)}")
        
        self.temp_files.clear()

class ExportMetrics:
    """Export performance and reliability metrics"""
    
    def __init__(self):
        self.export_stats = {
            'total_exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'format_stats': {},
            'error_types': {},
            'average_export_time': 0,
            'largest_export_size': 0
        }
    
    def record_export_attempt(self, format_type: str, file_size: int, duration: float, success: bool, error: Exception = None):
        """Record export attempt for metrics"""
        self.export_stats['total_exports'] += 1
        
        if success:
            self.export_stats['successful_exports'] += 1
        else:
            self.export_stats['failed_exports'] += 1
            
            if error:
                error_type = type(error).__name__
                self.export_stats['error_types'][error_type] = \
                    self.export_stats['error_types'].get(error_type, 0) + 1
        
        # Format statistics
        if format_type not in self.export_stats['format_stats']:
            self.export_stats['format_stats'][format_type] = {
                'attempts': 0, 'successes': 0, 'failures': 0
            }
        
        self.export_stats['format_stats'][format_type]['attempts'] += 1
        if success:
            self.export_stats['format_stats'][format_type]['successes'] += 1
        else:
            self.export_stats['format_stats'][format_type]['failures'] += 1
        
        # Update averages and maximums
        if file_size > self.export_stats['largest_export_size']:
            self.export_stats['largest_export_size'] = file_size
        
        # Update average export time
        if self.export_stats['total_exports'] > 0:
            total_time = self.export_stats['average_export_time'] * (self.export_stats['total_exports'] - 1)
            self.export_stats['average_export_time'] = (total_time + duration) / self.export_stats['total_exports']
    
    def get_success_rate(self) -> float:
        """Get overall export success rate"""
        if self.export_stats['total_exports'] == 0:
            return 0
        
        return (self.export_stats['successful_exports'] / self.export_stats['total_exports']) * 100
    
    def get_format_success_rate(self, format_type: str) -> float:
        """Get success rate for specific format"""
        if format_type not in self.export_stats['format_stats']:
            return 0
        
        stats = self.export_stats['format_stats'][format_type]
        if stats['attempts'] == 0:
            return 0
        
        return (stats['successes'] / stats['attempts']) * 100
    
    def render_metrics_dashboard(self):
        """Render export metrics dashboard"""
        st.subheader("📊 Export Reliability Metrics")
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Exports", self.export_stats['total_exports'])
        
        with col2:
            success_rate = self.get_success_rate()
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col3:
            st.metric("Failed Exports", self.export_stats['failed_exports'])
        
        with col4:
            avg_time = self.export_stats['average_export_time']
            st.metric("Avg Export Time", f"{avg_time:.1f}s")
        
        # Format-specific statistics
        if self.export_stats['format_stats']:
            st.write("**Format Statistics:**")
            
            format_data = []
            for format_type, stats in self.export_stats['format_stats'].items():
                success_rate = self.get_format_success_rate(format_type)
                format_data.append({
                    'Format': format_type.upper(),
                    'Attempts': stats['attempts'],
                    'Successes': stats['successes'],
                    'Failures': stats['failures'],
                    'Success Rate': f"{success_rate:.1f}%"
                })
            
            import pandas as pd
            df = pd.DataFrame(format_data)
            st.dataframe(df, use_container_width=True)
        
        # Error analysis
        if self.export_stats['error_types']:
            st.write("**Common Error Types:**")
            
            error_data = []
            for error_type, count in self.export_stats['error_types'].items():
                error_data.append({
                    'Error Type': error_type,
                    'Occurrences': count
                })
            
            import pandas as pd
            df = pd.DataFrame(error_data)
            st.dataframe(df, use_container_width=True)

class BulletproofExportSystem:
    """
    Complete bulletproof export system
    
    Integrates all export components:
    - Isolated export utilities (user's requirement)
    - Comprehensive error handling (user's try/except pattern)
    - File size validation (user's size check pattern)
    """
    
    def __init__(self, session_manager):
        self.session_manager = session_manager
        self.export_manager = RobustExportManager()
        self.metrics = ExportMetrics()
    
    def export_dataset(self, data: List[Dict], format_type: str, include_zip: bool = True, 
                      upload_to_hf: bool = False, hf_repo_name: str = None) -> Dict[str, Any]:
        """
        Export dataset with comprehensive error handling - USER'S EXACT PATTERN!
        
        Implements user's exact approach:
        try:
            generate_zip()
            if os.path.getsize("file.zip") < 50_000_000:
                upload_to_hf()
        except Exception as e:
            st.error(f"Export failed: {e}")
        """
        
        start_time = time.time()
        export_results = {
            'success': False,
            'files_generated': [],
            'zip_path': None,
            'upload_success': False,
            'errors': []
        }
        
        try:
            # USER'S EXACT TRY/EXCEPT PATTERN!
            
            # Generate primary export file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_training_dataset_{timestamp}.{format_type}"
            output_path = os.path.join(tempfile.gettempdir(), filename)
            
            # Export with validation
            if self.export_manager.export_with_validation(data, format_type, output_path):
                export_results['files_generated'].append(output_path)
                self.export_manager.logger.info(f"Primary export successful: {output_path}")
            else:
                raise ExportError("Primary export failed")
            
            # Generate ZIP if requested - USER'S EXACT PATTERN!
            if include_zip:
                try:
                    zip_filename = f"ai_training_dataset_{timestamp}.zip"
                    zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
                    
                    # Prepare files for ZIP
                    files_dict = {}
                    for file_path in export_results['files_generated']:
                        archive_name = os.path.basename(file_path)
                        files_dict[archive_name] = file_path
                    
                    # Add metadata file
                    metadata_path = self._generate_metadata_file(data, timestamp)
                    files_dict['metadata.json'] = metadata_path
                    export_results['files_generated'].append(metadata_path)
                    
                    # USER'S EXACT ZIP GENERATION PATTERN!
                    if self.export_manager.generate_zip_with_validation(files_dict, zip_path):
                        export_results['zip_path'] = zip_path
                        self.export_manager.logger.info(f"ZIP generation successful: {zip_path}")
                    
                except Exception as e:
                    error_msg = f"ZIP generation failed: {e}"
                    self.export_manager.logger.error(error_msg)
                    export_results['errors'].append(error_msg)
                    # Don't fail entire export for ZIP issues
            
            # Upload to Hugging Face if requested - USER'S EXACT PATTERN!
            if upload_to_hf and hf_repo_name:
                try:
                    upload_file = export_results['zip_path'] if export_results['zip_path'] else output_path
                    
                    # USER'S EXACT SIZE CHECK AND UPLOAD PATTERN!
                    if os.path.getsize(upload_file) < 50_000_000:  # USER'S EXACT SIZE LIMIT!
                        if self.export_manager.upload_to_huggingface_with_validation(upload_file, hf_repo_name):
                            export_results['upload_success'] = True
                            self.export_manager.logger.info(f"Hugging Face upload successful: {hf_repo_name}")
                    else:
                        error_msg = "File too large for Hugging Face upload"
                        self.export_manager.logger.warning(error_msg)
                        export_results['errors'].append(error_msg)
                
                except Exception as e:
                    error_msg = f"Hugging Face upload failed: {e}"
                    self.export_manager.logger.error(error_msg)
                    export_results['errors'].append(error_msg)
                    # Don't fail entire export for upload issues
            
            # Mark as successful if primary export worked
            export_results['success'] = True
            
            # Record metrics
            duration = time.time() - start_time
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            self.metrics.record_export_attempt(format_type, file_size, duration, True)
            
            return export_results
            
        except Exception as e:
            # USER'S EXACT ERROR HANDLING PATTERN!
            error_msg = f"Export failed: {e}"
            self.export_manager.logger.error(error_msg)
            st.error(error_msg)  # USER'S EXACT ERROR DISPLAY!
            
            export_results['errors'].append(error_msg)
            
            # Record metrics for failed export
            duration = time.time() - start_time
            self.metrics.record_export_attempt(format_type, 0, duration, False, e)
            
            return export_results
    
    def _generate_metadata_file(self, data: List[Dict], timestamp: str) -> str:
        """Generate metadata file for export package"""
        metadata = {
            'export_timestamp': timestamp,
            'total_items': len(data),
            'export_version': '1.0',
            'generator': 'Enhanced Universal AI Training Data Creator',
            'data_statistics': self._calculate_data_statistics(data)
        }
        
        metadata_path = os.path.join(tempfile.gettempdir(), f"metadata_{timestamp}.json")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata_path
    
    def _calculate_data_statistics(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for exported data"""
        if not data:
            return {}
        
        stats = {
            'total_items': len(data),
            'average_question_length': 0,
            'average_answer_length': 0,
            'total_characters': 0
        }
        
        try:
            question_lengths = []
            answer_lengths = []
            total_chars = 0
            
            for item in data:
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                question_lengths.append(len(question))
                answer_lengths.append(len(answer))
                total_chars += len(question) + len(answer)
            
            if question_lengths:
                stats['average_question_length'] = sum(question_lengths) / len(question_lengths)
            
            if answer_lengths:
                stats['average_answer_length'] = sum(answer_lengths) / len(answer_lengths)
            
            stats['total_characters'] = total_chars
            
        except Exception as e:
            self.export_manager.logger.warning(f"Statistics calculation failed: {str(e)}")
        
        return stats
    
    def render_export_interface(self):
        """Render bulletproof export interface"""
        st.subheader("📦 Bulletproof Export System")
        
        # Export configuration
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format:",
                ["jsonl", "json", "csv", "xlsx"],
                help="Choose the output format for your training dataset"
            )
            
            include_zip = st.checkbox(
                "Create ZIP Package",
                value=True,
                help="Include metadata and create a comprehensive ZIP package"
            )
        
        with col2:
            upload_to_hf = st.checkbox(
                "Upload to Hugging Face",
                value=False,
                help="Upload dataset to Hugging Face Hub (requires authentication)"
            )
            
            hf_repo_name = st.text_input(
                "Hugging Face Repository:",
                placeholder="username/dataset-name",
                help="Repository name for Hugging Face upload",
                disabled=not upload_to_hf
            )
        
        # Export button with validation
        if st.button("🚀 Export Dataset", type="primary"):
            
            # Validate prerequisites
            data = self.session_manager.get('final_dataset', [])
            
            if not data:
                st.error("❌ No dataset available for export. Please complete the workflow first.")
                return
            
            if upload_to_hf and not hf_repo_name:
                st.error("❌ Please specify a Hugging Face repository name.")
                return
            
            # Perform export with progress tracking
            with st.spinner("🔄 Exporting dataset..."):
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Preparing export...")
                progress_bar.progress(10)
                
                # Perform export
                results = self.export_dataset(
                    data=data,
                    format_type=export_format,
                    include_zip=include_zip,
                    upload_to_hf=upload_to_hf,
                    hf_repo_name=hf_repo_name
                )
                
                progress_bar.progress(100)
                status_text.text("Export completed!")
                
                # Display results
                if results['success']:
                    st.success("✅ **Export completed successfully!**")
                    
                    # Show generated files
                    if results['files_generated']:
                        st.write("**Generated Files:**")
                        for file_path in results['files_generated']:
                            if os.path.exists(file_path):
                                file_size = os.path.getsize(file_path)
                                st.write(f"- {os.path.basename(file_path)} ({file_size:,} bytes)")
                    
                    # Show ZIP information
                    if results['zip_path'] and os.path.exists(results['zip_path']):
                        zip_size = os.path.getsize(results['zip_path'])
                        st.write(f"**ZIP Package:** {os.path.basename(results['zip_path'])} ({zip_size:,} bytes)")
                        
                        # Provide download button
                        with open(results['zip_path'], 'rb') as f:
                            st.download_button(
                                "📥 Download ZIP Package",
                                data=f.read(),
                                file_name=os.path.basename(results['zip_path']),
                                mime="application/zip"
                            )
                    
                    # Show upload status
                    if upload_to_hf:
                        if results['upload_success']:
                            st.success(f"✅ **Successfully uploaded to Hugging Face:** {hf_repo_name}")
                        else:
                            st.warning("⚠️ **Upload to Hugging Face failed.** Dataset files are still available for download.")
                
                else:
                    st.error("❌ **Export failed.** Please check the error details below.")
                
                # Show any errors or warnings
                if results['errors']:
                    st.write("**Issues encountered:**")
                    for error in results['errors']:
                        st.warning(f"⚠️ {error}")
        
        # Export metrics dashboard
        st.markdown("---")
        self.metrics.render_metrics_dashboard()

# Example usage
def main():
    """Example usage of bulletproof export system"""
    
    # Mock session manager for testing
    class MockSessionManager:
        def __init__(self):
            self.data = {
                'final_dataset': [
                    {'question': 'What is AI?', 'answer': 'Artificial Intelligence is...'},
                    {'question': 'How does ML work?', 'answer': 'Machine Learning works by...'}
                ]
            }
        
        def get(self, key, default=None):
            return self.data.get(key, default)
    
    session_manager = MockSessionManager()
    export_system = BulletproofExportSystem(session_manager)
    
    st.title("🧠 Enhanced Universal AI Training Data Creator")
    st.write("*With Bulletproof Export System*")
    
    # Render export interface
    export_system.render_export_interface()

if __name__ == "__main__":
    main()

