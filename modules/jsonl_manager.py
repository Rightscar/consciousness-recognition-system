"""
JSONL File Manager for Consciousness Recognition System

Handles loading, saving, and managing JSONL training data files.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class JSONLManager:
    """Manages JSONL files for consciousness recognition training data."""
    
    def __init__(self):
        """Initialize the JSONL manager."""
        pass
    
    def load_jsonl(self, file_path: str) -> Dict[str, Any]:
        """
        Load dialogues from JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            Loading results with data or error
        """
        try:
            dialogues = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Extract dialogue from OpenAI format
                        if 'messages' in data:
                            # OpenAI fine-tuning format
                            messages = data['messages']
                            user_msg = next((m for m in messages if m['role'] == 'user'), None)
                            assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
                            
                            if user_msg and assistant_msg:
                                dialogue = {
                                    'question': user_msg['content'],
                                    'answer': assistant_msg['content'],
                                    'overall_score': data.get('metadata', {}).get('score', 0.0),
                                    'mode': data.get('metadata', {}).get('mode', 'unknown'),
                                    'source': data.get('metadata', {}).get('source', 'unknown'),
                                    'timestamp': data.get('metadata', {}).get('timestamp', ''),
                                    'metadata': data.get('metadata', {})
                                }
                                dialogues.append(dialogue)
                        
                        elif 'question' in data and 'answer' in data:
                            # Direct format
                            dialogues.append(data)
                    
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
            
            return {
                'success': True,
                'data': dialogues,
                'count': len(dialogues),
                'file_path': file_path
            }
            
        except FileNotFoundError:
            return {
                'success': False,
                'error': f"File not found: {file_path}",
                'data': []
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    def save_jsonl(self, dialogues: List[Dict[str, Any]], file_path: str) -> Dict[str, Any]:
        """
        Save dialogues to JSONL file.
        
        Args:
            dialogues: List of dialogue dictionaries
            file_path: Path to save JSONL file
            
        Returns:
            Save results
        """
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for dialogue in dialogues:
                    # Convert to OpenAI format
                    training_example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a consciousness recognition guide. Point directly to awareness without concepts or seeking."
                            },
                            {
                                "role": "user",
                                "content": dialogue.get('question', '')
                            },
                            {
                                "role": "assistant",
                                "content": dialogue.get('answer', '')
                            }
                        ],
                        "metadata": {
                            "score": dialogue.get('overall_score', 0.0),
                            "mode": dialogue.get('mode', 'unknown'),
                            "source": dialogue.get('source', 'unknown'),
                            "timestamp": dialogue.get('timestamp', datetime.now().isoformat())
                        }
                    }
                    
                    f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
            
            return {
                'success': True,
                'count': len(dialogues),
                'file_path': file_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'count': 0
            }
    
    def validate_jsonl(self, file_path: str) -> Dict[str, Any]:
        """
        Validate JSONL file format.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            Validation results
        """
        try:
            valid_lines = 0
            invalid_lines = 0
            errors = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        json.loads(line)
                        valid_lines += 1
                    except json.JSONDecodeError as e:
                        invalid_lines += 1
                        errors.append(f"Line {line_num}: {e}")
            
            return {
                'valid': invalid_lines == 0,
                'valid_lines': valid_lines,
                'invalid_lines': invalid_lines,
                'errors': errors[:10]  # Limit to first 10 errors
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


class FileManager:
    """Manages files in the output directory."""
    
    def __init__(self, base_directory: str = "./output"):
        """
        Initialize file manager.
        
        Args:
            base_directory: Base directory for file operations
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
    
    def list_files(self, extension: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in the base directory.
        
        Args:
            extension: Filter by file extension (e.g., '.jsonl')
            
        Returns:
            List of file information dictionaries
        """
        try:
            files = []
            
            for file_path in self.base_directory.iterdir():
                if file_path.is_file():
                    if extension and not file_path.suffix == extension:
                        continue
                    
                    stat = file_path.stat()
                    
                    file_info = {
                        'name': file_path.name,
                        'path': str(file_path),
                        'size': stat.st_size,
                        'size_human': self._format_size(stat.st_size),
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'modified_human': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                        'extension': file_path.suffix
                    }
                    
                    # Add JSONL-specific info
                    if file_path.suffix == '.jsonl':
                        jsonl_info = self._analyze_jsonl(file_path)
                        file_info.update(jsonl_info)
                    
                    files.append(file_info)
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x['modified'], reverse=True)
            
            return files
            
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            Deletion results
        """
        try:
            Path(file_path).unlink()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def rename_file(self, old_path: str, new_name: str) -> Dict[str, Any]:
        """
        Rename a file.
        
        Args:
            old_path: Current file path
            new_name: New file name
            
        Returns:
            Rename results
        """
        try:
            old_file = Path(old_path)
            new_file = old_file.parent / new_name
            old_file.rename(new_file)
            return {'success': True, 'new_path': str(new_file)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def _analyze_jsonl(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JSONL file for additional metadata."""
        try:
            entry_count = 0
            modes = set()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            entry_count += 1
                            
                            # Extract mode from metadata
                            if 'metadata' in data and 'mode' in data['metadata']:
                                modes.add(data['metadata']['mode'])
                        except:
                            continue
            
            return {
                'entry_count': entry_count,
                'modes': list(modes)
            }
            
        except Exception:
            return {
                'entry_count': 'Unknown',
                'modes': []
            }

