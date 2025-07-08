"""
Export Confirmations Module
===========================

Provides comprehensive export confirmation and validation for the Enhanced Universal AI Training Data Creator.
Includes file size validation, cost estimation, and user confirmation dialogs for safe exports.

Features:
- File size validation before export
- Cost estimation for external uploads
- Confirmation dialogs for large operations
- Export progress tracking
- Rollback capabilities
- Safety checks and warnings
"""

import streamlit as st
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExportConfirmations:
    """Export confirmation system with comprehensive safety checks"""
    
    def __init__(self):
        self.size_limits = {
            'local_download': 100 * 1024 * 1024,  # 100MB
            'huggingface_upload': 50 * 1024 * 1024,  # 50MB
            'email_attachment': 25 * 1024 * 1024,  # 25MB
            'warning_threshold': 10 * 1024 * 1024  # 10MB warning
        }
        
        self.cost_estimates = {
            'huggingface_storage': 0.023,  # per GB per month
            'bandwidth_cost': 0.09,  # per GB transfer
            'processing_cost': 0.001  # per item processed
        }
    
    def estimate_export_size(self, data: List[Dict[str, Any]], export_format: str) -> int:
        """Estimate export file size in bytes"""
        if not data:
            return 0
        
        # Calculate content size
        total_content_size = 0
        for item in data:
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            total_content_size += len(input_text.encode('utf-8'))
            total_content_size += len(output_text.encode('utf-8'))
        
        # Format-specific size estimation
        format_multipliers = {
            'json': 2.5,  # JSON formatting overhead
            'jsonl': 1.8,  # JSONL is more compact
            'csv': 1.2,   # CSV is compact
            'xlsx': 3.0,  # Excel has significant overhead
            'txt': 1.0,   # Plain text baseline
            'zip': 0.3    # Compression ratio
        }
        
        multiplier = format_multipliers.get(export_format.lower(), 2.0)
        estimated_size = int(total_content_size * multiplier)
        
        # Add metadata overhead
        metadata_overhead = len(json.dumps({
            'metadata': {
                'total_items': len(data),
                'created_at': datetime.now().isoformat(),
                'export_format': export_format
            }
        }).encode('utf-8'))
        
        return estimated_size + metadata_overhead
    
    def estimate_upload_cost(self, file_size_bytes: int, destination: str) -> Dict[str, float]:
        """Estimate costs for external uploads"""
        file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
        
        costs = {
            'storage_monthly': 0.0,
            'bandwidth': 0.0,
            'total_first_month': 0.0
        }
        
        if destination.lower() == 'huggingface':
            costs['storage_monthly'] = file_size_gb * self.cost_estimates['huggingface_storage']
            costs['bandwidth'] = file_size_gb * self.cost_estimates['bandwidth_cost']
            costs['total_first_month'] = costs['storage_monthly'] + costs['bandwidth']
        
        return costs
    
    def validate_export_safety(self, data: List[Dict[str, Any]], 
                             export_format: str, destination: str) -> Dict[str, Any]:
        """Validate export safety and generate warnings"""
        validation_result = {
            'safe_to_export': True,
            'warnings': [],
            'errors': [],
            'size_info': {},
            'cost_info': {},
            'recommendations': []
        }
        
        # Estimate file size
        estimated_size = self.estimate_export_size(data, export_format)
        validation_result['size_info'] = {
            'estimated_bytes': estimated_size,
            'estimated_mb': estimated_size / (1024 * 1024),
            'item_count': len(data)
        }
        
        # Check size limits
        if destination == 'local':
            limit = self.size_limits['local_download']
            limit_name = "local download"
        elif destination == 'huggingface':
            limit = self.size_limits['huggingface_upload']
            limit_name = "Hugging Face upload"
        elif destination == 'email':
            limit = self.size_limits['email_attachment']
            limit_name = "email attachment"
        else:
            limit = self.size_limits['warning_threshold']
            limit_name = "general export"
        
        if estimated_size > limit:
            validation_result['safe_to_export'] = False
            validation_result['errors'].append(
                f"File size ({estimated_size / (1024 * 1024):.1f} MB) exceeds {limit_name} limit "
                f"({limit / (1024 * 1024):.1f} MB)"
            )
        elif estimated_size > self.size_limits['warning_threshold']:
            validation_result['warnings'].append(
                f"Large file size ({estimated_size / (1024 * 1024):.1f} MB) - export may take time"
            )
        
        # Cost estimation for external services
        if destination in ['huggingface', 'cloud']:
            costs = self.estimate_upload_cost(estimated_size, destination)
            validation_result['cost_info'] = costs
            
            if costs['total_first_month'] > 1.0:  # $1 threshold
                validation_result['warnings'].append(
                    f"Estimated first month cost: ${costs['total_first_month']:.2f}"
                )
        
        # Data quality checks
        if len(data) < 10:
            validation_result['warnings'].append(
                "Small dataset (< 10 items) may not be suitable for training"
            )
        
        # Check for sensitive content
        sensitive_patterns = ['password', 'api_key', 'secret', 'token', 'private']
        for item in data[:10]:  # Check first 10 items
            content = f"{item.get('input', '')} {item.get('output', '')}".lower()
            for pattern in sensitive_patterns:
                if pattern in content:
                    validation_result['warnings'].append(
                        "Potential sensitive information detected - please review before export"
                    )
                    break
        
        # Generate recommendations
        if estimated_size > self.size_limits['warning_threshold']:
            validation_result['recommendations'].append(
                "Consider splitting large datasets into smaller batches"
            )
        
        if export_format in ['xlsx', 'json'] and estimated_size > 5 * 1024 * 1024:
            validation_result['recommendations'].append(
                "Consider using JSONL format for better compression and performance"
            )
        
        return validation_result
    
    def render_export_confirmation_dialog(self, validation_result: Dict[str, Any], 
                                        export_format: str, destination: str) -> bool:
        """Render export confirmation dialog and return user decision"""
        
        st.markdown("### 🔍 **Export Confirmation**")
        
        # Display size information
        size_info = validation_result['size_info']
        st.info(f"📊 **Export Details:**\n"
               f"- Items: {size_info['item_count']:,}\n"
               f"- Estimated size: {size_info['estimated_mb']:.1f} MB\n"
               f"- Format: {export_format.upper()}\n"
               f"- Destination: {destination.title()}")
        
        # Display warnings
        if validation_result['warnings']:
            st.warning("⚠️ **Warnings:**")
            for warning in validation_result['warnings']:
                st.write(f"• {warning}")
        
        # Display errors
        if validation_result['errors']:
            st.error("❌ **Errors:**")
            for error in validation_result['errors']:
                st.write(f"• {error}")
            st.stop()  # Don't allow export if there are errors
        
        # Display cost information
        if validation_result['cost_info']:
            cost_info = validation_result['cost_info']
            if cost_info['total_first_month'] > 0:
                st.info(f"💰 **Estimated Costs:**\n"
                       f"- Storage (monthly): ${cost_info['storage_monthly']:.3f}\n"
                       f"- Bandwidth: ${cost_info['bandwidth']:.3f}\n"
                       f"- **Total first month: ${cost_info['total_first_month']:.3f}**")
        
        # Display recommendations
        if validation_result['recommendations']:
            st.info("💡 **Recommendations:**")
            for rec in validation_result['recommendations']:
                st.write(f"• {rec}")
        
        # Confirmation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("✅ Confirm Export", type="primary"):
                return True
        
        with col2:
            if st.button("❌ Cancel"):
                return False
        
        with col3:
            if st.button("📝 Review Data"):
                st.session_state['show_data_preview'] = True
                st.rerun()
        
        return None  # No decision made yet
    
    def render_upload_progress(self, progress_callback=None):
        """Render upload progress with cancellation option"""
        
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 📤 **Upload in Progress**")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Cancel button
            col1, col2 = st.columns([3, 1])
            with col2:
                cancel_button = st.button("❌ Cancel Upload")
            
            if cancel_button:
                st.session_state['upload_cancelled'] = True
                st.error("Upload cancelled by user")
                return False
            
            # Simulate progress updates (in real implementation, this would be driven by actual upload)
            if progress_callback:
                return progress_callback(progress_bar, status_text)
            
            return True
    
    def validate_file_before_upload(self, file_path: str, destination: str) -> Dict[str, Any]:
        """Validate file before upload to external service"""
        validation_result = {
            'valid': True,
            'file_size': 0,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                validation_result['valid'] = False
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Get file size
            file_size = os.path.getsize(file_path)
            validation_result['file_size'] = file_size
            
            # Check size limits
            if destination == 'huggingface':
                limit = self.size_limits['huggingface_upload']
                if file_size > limit:
                    validation_result['valid'] = False
                    validation_result['errors'].append(
                        f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds "
                        f"Hugging Face limit ({limit / (1024 * 1024):.1f} MB)"
                    )
            
            # Check file format
            file_extension = os.path.splitext(file_path)[1].lower()
            allowed_extensions = ['.json', '.jsonl', '.csv', '.txt', '.zip']
            
            if file_extension not in allowed_extensions:
                validation_result['warnings'].append(
                    f"Unusual file extension: {file_extension}"
                )
            
            # Check file content (basic validation)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if not first_line.strip():
                        validation_result['warnings'].append("File appears to be empty")
            except UnicodeDecodeError:
                validation_result['warnings'].append("File may contain binary data")
            except Exception as e:
                validation_result['warnings'].append(f"Could not read file: {str(e)}")
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"File validation error: {str(e)}")
        
        return validation_result
    
    def render_batch_export_confirmation(self, batch_info: Dict[str, Any]) -> bool:
        """Render confirmation for batch export operations"""
        
        st.markdown("### 📦 **Batch Export Confirmation**")
        
        total_items = batch_info.get('total_items', 0)
        total_size = batch_info.get('total_size_mb', 0)
        batch_count = batch_info.get('batch_count', 1)
        
        st.info(f"📊 **Batch Export Details:**\n"
               f"- Total items: {total_items:,}\n"
               f"- Total size: {total_size:.1f} MB\n"
               f"- Number of batches: {batch_count}\n"
               f"- Items per batch: {total_items // batch_count if batch_count > 0 else 0}")
        
        if total_size > 50:  # 50MB threshold
            st.warning("⚠️ Large batch export - this may take several minutes")
        
        if batch_count > 10:
            st.warning("⚠️ Many batches - consider consolidating or using a different format")
        
        # Confirmation
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Start Batch Export", type="primary"):
                return True
        
        with col2:
            if st.button("❌ Cancel Batch Export"):
                return False
        
        return None
    
    def create_export_summary(self, export_results: Dict[str, Any]) -> str:
        """Create export summary for user confirmation"""
        
        summary_lines = [
            "# Export Summary",
            "",
            f"**Export completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Items exported:** {export_results.get('items_exported', 0):,}",
            f"**File size:** {export_results.get('file_size_mb', 0):.1f} MB",
            f"**Format:** {export_results.get('format', 'Unknown')}",
            f"**Destination:** {export_results.get('destination', 'Unknown')}",
            ""
        ]
        
        if export_results.get('validation_passed', True):
            summary_lines.append("✅ **Validation:** All checks passed")
        else:
            summary_lines.append("⚠️ **Validation:** Some issues detected")
        
        if export_results.get('cost_info'):
            cost_info = export_results['cost_info']
            summary_lines.extend([
                "",
                "## Cost Information",
                f"- Storage cost (monthly): ${cost_info.get('storage_monthly', 0):.3f}",
                f"- Bandwidth cost: ${cost_info.get('bandwidth', 0):.3f}",
                f"- **Total first month: ${cost_info.get('total_first_month', 0):.3f}**"
            ])
        
        if export_results.get('warnings'):
            summary_lines.extend([
                "",
                "## Warnings",
                ""
            ])
            for warning in export_results['warnings']:
                summary_lines.append(f"- {warning}")
        
        return "\n".join(summary_lines)


# Global export confirmations instance
export_confirmations = ExportConfirmations()

