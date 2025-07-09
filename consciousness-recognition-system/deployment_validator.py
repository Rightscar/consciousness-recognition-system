#!/usr/bin/env python3
"""
Comprehensive Deployment Validator
Identifies ALL potential silent deployment issues before they break production
"""

import os
import sys
import ast
import re
import importlib.util
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Comprehensive validator for deployment issues"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues = []
        self.warnings = []
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks"""
        
        results = {
            'syntax_errors': self.check_syntax_errors(),
            'import_issues': self.check_import_issues(),
            'string_issues': self.check_string_issues(),
            'indentation_issues': self.check_indentation_issues(),
            'circular_imports': self.check_circular_imports(),
            'missing_files': self.check_missing_files(),
            'encoding_issues': self.check_encoding_issues(),
            'requirements_issues': self.check_requirements_issues(),
            'streamlit_issues': self.check_streamlit_issues(),
            'render_deployment_issues': self.check_render_deployment_issues(),
            'summary': {
                'total_issues': len(self.issues),
                'total_warnings': len(self.warnings),
                'critical_issues': len([i for i in self.issues if i.get('severity') == 'critical']),
                'deployment_ready': len(self.issues) == 0
            }
        }
        
        return results
    
    def check_syntax_errors(self) -> List[Dict[str, Any]]:
        """Check for syntax errors in all Python files"""
        syntax_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Check for compilation errors
                try:
                    compile(source, str(py_file), 'exec')
                except SyntaxError as e:
                    issue = {
                        'type': 'syntax_error',
                        'file': str(py_file),
                        'line': e.lineno,
                        'message': str(e),
                        'severity': 'critical'
                    }
                    syntax_issues.append(issue)
                    self.issues.append(issue)
                
                # Check for AST parsing issues
                try:
                    ast.parse(source)
                except SyntaxError as e:
                    issue = {
                        'type': 'ast_parse_error',
                        'file': str(py_file),
                        'line': e.lineno,
                        'message': str(e),
                        'severity': 'critical'
                    }
                    syntax_issues.append(issue)
                    self.issues.append(issue)
                    
            except Exception as e:
                issue = {
                    'type': 'file_read_error',
                    'file': str(py_file),
                    'message': str(e),
                    'severity': 'high'
                }
                syntax_issues.append(issue)
                self.issues.append(issue)
        
        return syntax_issues
    
    def check_string_issues(self) -> List[Dict[str, Any]]:
        """Check for malformed strings that could cause issues"""
        string_issues = []
        
        problematic_patterns = [
            (r"r''[^']*'[^']*$", "Malformed raw string with mixed quotes"),
            (r'r""[^"]*"[^"]*$', "Malformed raw string with mixed quotes"),
            (r"f'[^']*\{[^}]*$", "Unterminated f-string"),
            (r'f"[^"]*\{[^}]*$', "Unterminated f-string"),
            (r"'''[^']*$", "Unterminated triple-quoted string"),
            (r'"""[^"]*$', "Unterminated triple-quoted string"),
            (r"\\$", "Line ending with backslash (potential continuation issue)"),
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, description in problematic_patterns:
                        if re.search(pattern, line.strip()):
                            issue = {
                                'type': 'string_issue',
                                'file': str(py_file),
                                'line': line_num,
                                'message': f"{description}: {line.strip()}",
                                'severity': 'high'
                            }
                            string_issues.append(issue)
                            self.issues.append(issue)
                            
            except Exception as e:
                logger.warning(f"Could not check strings in {py_file}: {e}")
        
        return string_issues
    
    def check_import_issues(self) -> List[Dict[str, Any]]:
        """Check for import-related issues"""
        import_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for conditional imports used in type hints
                lines = content.split('\n')
                in_try_block = False
                imported_names = set()
                
                for line_num, line in enumerate(lines, 1):
                    stripped = line.strip()
                    
                    # Track try blocks
                    if stripped.startswith('try:'):
                        in_try_block = True
                        imported_names.clear()
                    elif stripped.startswith('except'):
                        in_try_block = False
                    
                    # Track imports in try blocks
                    if in_try_block and ('import ' in stripped):
                        # Extract imported names
                        if 'from ' in stripped and ' import ' in stripped:
                            parts = stripped.split(' import ')
                            if len(parts) > 1:
                                names = [n.strip() for n in parts[1].split(',')]
                                imported_names.update(names)
                        elif stripped.startswith('import '):
                            name = stripped.replace('import ', '').split('.')[0]
                            imported_names.add(name)
                    
                    # Check for usage of conditionally imported names in type hints
                    if not in_try_block and ':' in stripped:
                        for name in imported_names:
                            if f': {name}' in stripped or f'-> {name}' in stripped:
                                issue = {
                                    'type': 'conditional_import_in_type_hint',
                                    'file': str(py_file),
                                    'line': line_num,
                                    'message': f"Type hint uses conditionally imported '{name}': {stripped}",
                                    'severity': 'critical'
                                }
                                import_issues.append(issue)
                                self.issues.append(issue)
                
                # Check for circular import patterns
                if 'from modules.' in content and py_file.name != '__init__.py':
                    # Check if this module is imported by modules it imports
                    module_name = py_file.stem
                    imported_modules = re.findall(r'from modules\.(\w+)', content)
                    
                    for imported_module in imported_modules:
                        imported_file = py_file.parent / f"{imported_module}.py"
                        if imported_file.exists():
                            try:
                                with open(imported_file, 'r') as f:
                                    imported_content = f.read()
                                if f'from modules.{module_name}' in imported_content:
                                    issue = {
                                        'type': 'potential_circular_import',
                                        'file': str(py_file),
                                        'message': f"Potential circular import between {module_name} and {imported_module}",
                                        'severity': 'high'
                                    }
                                    import_issues.append(issue)
                                    self.issues.append(issue)
                            except Exception:
                                pass
                                
            except Exception as e:
                logger.warning(f"Could not check imports in {py_file}: {e}")
        
        return import_issues
    
    def check_indentation_issues(self) -> List[Dict[str, Any]]:
        """Check for indentation issues"""
        indentation_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'tabnanny', str(py_file)],
                    capture_output=True, text=True
                )
                
                if result.returncode != 0:
                    issue = {
                        'type': 'indentation_error',
                        'file': str(py_file),
                        'message': result.stdout or result.stderr,
                        'severity': 'high'
                    }
                    indentation_issues.append(issue)
                    self.issues.append(issue)
                    
            except Exception as e:
                logger.warning(f"Could not check indentation in {py_file}: {e}")
        
        return indentation_issues
    
    def check_circular_imports(self) -> List[Dict[str, Any]]:
        """Check for circular import issues by attempting imports"""
        circular_issues = []
        
        # Try importing main modules
        main_modules = ['app', 'healthcheck']
        
        for module_name in main_modules:
            module_file = self.project_root / f"{module_name}.py"
            if module_file.exists():
                try:
                    # Add project root to path
                    sys.path.insert(0, str(self.project_root))
                    
                    spec = importlib.util.spec_from_file_location(module_name, module_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                except Exception as e:
                    if 'circular import' in str(e).lower():
                        issue = {
                            'type': 'circular_import',
                            'file': str(module_file),
                            'message': str(e),
                            'severity': 'critical'
                        }
                        circular_issues.append(issue)
                        self.issues.append(issue)
                    else:
                        # Other import errors might be dependency issues
                        warning = {
                            'type': 'import_warning',
                            'file': str(module_file),
                            'message': str(e),
                            'severity': 'medium'
                        }
                        circular_issues.append(warning)
                        self.warnings.append(warning)
                finally:
                    # Clean up sys.path
                    if str(self.project_root) in sys.path:
                        sys.path.remove(str(self.project_root))
        
        return circular_issues
    
    def check_missing_files(self) -> List[Dict[str, Any]]:
        """Check for missing critical files"""
        missing_issues = []
        
        critical_files = [
            'app.py',
            'requirements.txt',
            'render.yaml',
            '.env.example'
        ]
        
        for file_name in critical_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                issue = {
                    'type': 'missing_file',
                    'file': file_name,
                    'message': f"Critical file {file_name} is missing",
                    'severity': 'high'
                }
                missing_issues.append(issue)
                self.issues.append(issue)
        
        return missing_issues
    
    def check_encoding_issues(self) -> List[Dict[str, Any]]:
        """Check for encoding issues"""
        encoding_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                # Try reading with UTF-8
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for non-ASCII characters that might cause issues
                try:
                    content.encode('ascii')
                except UnicodeEncodeError:
                    # Check if file has proper encoding declaration
                    lines = content.split('\n')[:2]
                    has_encoding = any('coding' in line or 'encoding' in line for line in lines)
                    
                    if not has_encoding:
                        warning = {
                            'type': 'encoding_warning',
                            'file': str(py_file),
                            'message': "File contains non-ASCII characters but no encoding declaration",
                            'severity': 'medium'
                        }
                        encoding_issues.append(warning)
                        self.warnings.append(warning)
                        
            except UnicodeDecodeError:
                issue = {
                    'type': 'encoding_error',
                    'file': str(py_file),
                    'message': "File cannot be decoded as UTF-8",
                    'severity': 'high'
                }
                encoding_issues.append(issue)
                self.issues.append(issue)
            except Exception as e:
                logger.warning(f"Could not check encoding in {py_file}: {e}")
        
        return encoding_issues
    
    def check_requirements_issues(self) -> List[Dict[str, Any]]:
        """Check requirements.txt for potential issues"""
        requirements_issues = []
        
        req_file = self.project_root / 'requirements.txt'
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Check for version conflicts
                        if '>=' in line and '<=' in line:
                            warning = {
                                'type': 'version_constraint_warning',
                                'file': 'requirements.txt',
                                'line': line_num,
                                'message': f"Complex version constraint might cause conflicts: {line}",
                                'severity': 'low'
                            }
                            requirements_issues.append(warning)
                            self.warnings.append(warning)
                        
                        # Check for development dependencies in production
                        dev_packages = ['pytest', 'black', 'flake8', 'mypy', 'jupyter']
                        package_name = line.split('==')[0].split('>=')[0].split('<=')[0]
                        if package_name.lower() in dev_packages:
                            warning = {
                                'type': 'dev_dependency_warning',
                                'file': 'requirements.txt',
                                'line': line_num,
                                'message': f"Development dependency in production requirements: {line}",
                                'severity': 'low'
                            }
                            requirements_issues.append(warning)
                            self.warnings.append(warning)
                            
            except Exception as e:
                issue = {
                    'type': 'requirements_read_error',
                    'file': 'requirements.txt',
                    'message': str(e),
                    'severity': 'medium'
                }
                requirements_issues.append(issue)
                self.issues.append(issue)
        
        return requirements_issues
    
    def check_streamlit_issues(self) -> List[Dict[str, Any]]:
        """Check for Streamlit-specific issues"""
        streamlit_issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    stripped = line.strip()
                    
                    # Check for unsafe session state access
                    if 'st.session_state[' in stripped and '.get(' not in stripped:
                        warning = {
                            'type': 'unsafe_session_state',
                            'file': str(py_file),
                            'line': line_num,
                            'message': f"Unsafe session state access (use .get() for safety): {stripped}",
                            'severity': 'medium'
                        }
                        streamlit_issues.append(warning)
                        self.warnings.append(warning)
                    
                    # Check for missing st.set_page_config
                    if py_file.name == 'app.py' and 'st.set_page_config' not in content:
                        warning = {
                            'type': 'missing_page_config',
                            'file': str(py_file),
                            'message': "Missing st.set_page_config() in main app",
                            'severity': 'low'
                        }
                        streamlit_issues.append(warning)
                        self.warnings.append(warning)
                        
            except Exception as e:
                logger.warning(f"Could not check Streamlit issues in {py_file}: {e}")
        
        return streamlit_issues
    
    def check_render_deployment_issues(self) -> List[Dict[str, Any]]:
        """Check for Render.com specific deployment issues"""
        render_issues = []
        
        # Check render.yaml
        render_file = self.project_root / 'render.yaml'
        if render_file.exists():
            try:
                with open(render_file, 'r') as f:
                    content = f.read()
                
                # Check for missing critical configurations
                if 'buildCommand' not in content:
                    issue = {
                        'type': 'missing_build_command',
                        'file': 'render.yaml',
                        'message': "Missing buildCommand in render.yaml",
                        'severity': 'high'
                    }
                    render_issues.append(issue)
                    self.issues.append(issue)
                
                if 'startCommand' not in content:
                    issue = {
                        'type': 'missing_start_command',
                        'file': 'render.yaml',
                        'message': "Missing startCommand in render.yaml",
                        'severity': 'high'
                    }
                    render_issues.append(issue)
                    self.issues.append(issue)
                
                # Check for memory optimization settings
                if 'LOW_MEM_MODE' not in content:
                    warning = {
                        'type': 'missing_memory_optimization',
                        'file': 'render.yaml',
                        'message': "Consider adding LOW_MEM_MODE environment variable",
                        'severity': 'low'
                    }
                    render_issues.append(warning)
                    self.warnings.append(warning)
                    
            except Exception as e:
                issue = {
                    'type': 'render_config_error',
                    'file': 'render.yaml',
                    'message': str(e),
                    'severity': 'medium'
                }
                render_issues.append(issue)
                self.issues.append(issue)
        
        return render_issues
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report"""
        results = self.validate_all()
        
        report = ["# 🔍 DEPLOYMENT VALIDATION REPORT\n"]
        
        # Summary
        summary = results['summary']
        if summary['deployment_ready']:
            report.append("## ✅ **DEPLOYMENT STATUS: READY**\n")
        else:
            report.append("## ❌ **DEPLOYMENT STATUS: ISSUES FOUND**\n")
        
        report.append(f"- **Total Issues:** {summary['total_issues']}")
        report.append(f"- **Critical Issues:** {summary['critical_issues']}")
        report.append(f"- **Warnings:** {summary['total_warnings']}\n")
        
        # Detailed results
        for category, issues in results.items():
            if category == 'summary' or not issues:
                continue
                
            report.append(f"## {category.replace('_', ' ').title()}\n")
            
            for issue in issues:
                severity_icon = {
                    'critical': '🔴',
                    'high': '🟠', 
                    'medium': '🟡',
                    'low': '🟢'
                }.get(issue.get('severity', 'medium'), '⚪')
                
                report.append(f"{severity_icon} **{issue['type']}**")
                report.append(f"   - File: `{issue['file']}`")
                if 'line' in issue:
                    report.append(f"   - Line: {issue['line']}")
                report.append(f"   - Message: {issue['message']}\n")
        
        return "\n".join(report)

def main():
    """Main validation function"""
    project_root = "/home/ubuntu/consciousness-recognition-system"
    
    validator = DeploymentValidator(project_root)
    results = validator.validate_all()
    
    # Print summary
    summary = results['summary']
    print(f"\n🔍 DEPLOYMENT VALIDATION COMPLETE")
    print(f"Total Issues: {summary['total_issues']}")
    print(f"Critical Issues: {summary['critical_issues']}")
    print(f"Warnings: {summary['total_warnings']}")
    print(f"Deployment Ready: {'✅ YES' if summary['deployment_ready'] else '❌ NO'}")
    
    # Save detailed report
    report = validator.generate_report()
    with open(f"{project_root}/DEPLOYMENT_VALIDATION_REPORT.md", 'w') as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved to: DEPLOYMENT_VALIDATION_REPORT.md")
    
    return results

if __name__ == "__main__":
    main()

