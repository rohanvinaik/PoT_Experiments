#!/usr/bin/env python3
"""
Simple tail-chasing pattern analyzer for PoT codebase.
Detects common LLM-induced anti-patterns.
"""

import ast
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
import hashlib

class TailChasingAnalyzer:
    """Detect tail-chasing anti-patterns in Python code."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.issues = []
        self.function_signatures = defaultdict(list)
        self.imports_by_file = {}
        self.stub_functions = []
        self.duplicate_logic = []
        self.circular_imports = []
        
    def analyze(self) -> Dict[str, Any]:
        """Run full analysis on project."""
        print(f"Analyzing {self.project_path}...")
        
        # Collect all Python files
        py_files = list(self.project_path.rglob("*.py"))
        py_files = [f for f in py_files if not any(
            part.startswith('.') or part == '__pycache__' 
            for part in f.parts
        )]
        
        print(f"Found {len(py_files)} Python files")
        
        # First pass: collect all functions and imports
        for file_path in py_files:
            self._analyze_file(file_path)
        
        # Second pass: detect patterns
        self._detect_phantom_functions()
        self._detect_duplicate_implementations()
        self._detect_circular_dependencies()
        self._detect_unused_imports()
        self._detect_context_thrashing()
        
        return self._generate_report()
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Collect imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            self.imports_by_file[str(file_path)] = imports
            
            # Collect functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    sig = self._get_function_signature(node)
                    self.function_signatures[sig].append({
                        'file': str(file_path),
                        'name': node.name,
                        'line': node.lineno,
                        'body': ast.dump(node),
                        'is_stub': self._is_stub_function(node)
                    })
                    
        except Exception as e:
            pass  # Skip files that can't be parsed
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Generate a signature for function comparison."""
        params = [arg.arg for arg in node.args.args]
        return f"{node.name}({','.join(params)})"
    
    def _is_stub_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is a stub/placeholder."""
        if not node.body:
            return True
        
        # Check for common stub patterns
        for stmt in node.body:
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Call):
                    if hasattr(stmt.exc.func, 'id'):
                        if stmt.exc.func.id in ['NotImplementedError', 'TODO']:
                            return True
            if isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Constant):
                    if stmt.value.value == "...":
                        return True
        
        # Check if function only has docstring
        if len(node.body) == 1:
            if isinstance(node.body[0], ast.Expr):
                if isinstance(node.body[0].value, ast.Constant):
                    if isinstance(node.body[0].value.value, str):
                        return True
        
        return False
    
    def _detect_phantom_functions(self):
        """Detect stub functions that were never implemented."""
        for sig, funcs in self.function_signatures.items():
            for func in funcs:
                if func['is_stub']:
                    self.stub_functions.append({
                        'type': 'phantom_function',
                        'file': func['file'],
                        'line': func['line'],
                        'name': func['name'],
                        'severity': 'medium',
                        'message': f"Stub function '{func['name']}' never implemented"
                    })
    
    def _detect_duplicate_implementations(self):
        """Detect semantically similar functions."""
        for sig, funcs in self.function_signatures.items():
            if len(funcs) > 1:
                # Group by similar implementation
                groups = defaultdict(list)
                for func in funcs:
                    # Simple hash of AST structure
                    body_hash = hashlib.md5(func['body'].encode()).hexdigest()[:8]
                    groups[body_hash].append(func)
                
                for body_hash, group in groups.items():
                    if len(group) > 1:
                        self.duplicate_logic.append({
                            'type': 'duplicate_implementation',
                            'severity': 'high',
                            'functions': [
                                {'file': f['file'], 'name': f['name'], 'line': f['line']}
                                for f in group
                            ],
                            'message': f"Found {len(group)} duplicate implementations of {sig}"
                        })
    
    def _detect_circular_dependencies(self):
        """Detect circular import patterns."""
        # Build import graph
        for file1, imports1 in self.imports_by_file.items():
            for file2, imports2 in self.imports_by_file.items():
                if file1 != file2:
                    # Check if files import each other
                    f1_base = Path(file1).stem
                    f2_base = Path(file2).stem
                    
                    if any(f2_base in imp for imp in imports1) and \
                       any(f1_base in imp for imp in imports2):
                        self.circular_imports.append({
                            'type': 'circular_dependency',
                            'severity': 'high',
                            'files': [file1, file2],
                            'message': f"Circular dependency between {Path(file1).name} and {Path(file2).name}"
                        })
    
    def _detect_unused_imports(self):
        """Detect imports that are never used."""
        for file_path, imports in self.imports_by_file.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content)
                
                # Collect all names used in the file
                used_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        used_names.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        if isinstance(node.value, ast.Name):
                            used_names.add(node.value.id)
                
                # Check each import
                for imp in imports:
                    # Get the base module name
                    base_name = imp.split('.')[0]
                    
                    # Check if import is used
                    if base_name not in used_names:
                        # Also check if it's used as a string (common in dynamic imports)
                        if f'"{base_name}"' not in content and f"'{base_name}'" not in content:
                            self.issues.append({
                                'type': 'unused_import',
                                'severity': 'low',
                                'file': file_path,
                                'import': imp,
                                'message': f"Import '{imp}' is never used in {Path(file_path).name}"
                            })
            except Exception:
                pass  # Skip files that can't be analyzed
    
    def _detect_context_thrashing(self):
        """Detect reimplementation after context loss."""
        # Look for similar functions with significant line separation
        for sig, funcs in self.function_signatures.items():
            if len(funcs) > 1:
                # Sort by file and line
                sorted_funcs = sorted(funcs, key=lambda x: (x['file'], x['line']))
                
                for i in range(len(sorted_funcs) - 1):
                    f1 = sorted_funcs[i]
                    f2 = sorted_funcs[i + 1]
                    
                    if f1['file'] == f2['file']:
                        line_diff = f2['line'] - f1['line']
                        if line_diff > 500:  # Significant separation
                            self.issues.append({
                                'type': 'context_window_thrashing',
                                'severity': 'high',
                                'file': f1['file'],
                                'lines': [f1['line'], f2['line']],
                                'message': f"Function '{f1['name']}' reimplemented after {line_diff} lines"
                            })
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate analysis report."""
        all_issues = (
            self.stub_functions + 
            self.duplicate_logic + 
            self.circular_imports + 
            self.issues
        )
        
        # Remove duplicate circular import entries
        seen_circulars = set()
        unique_circulars = []
        for issue in all_issues:
            if issue['type'] == 'circular_dependency':
                files_key = tuple(sorted(issue['files']))
                if files_key not in seen_circulars:
                    seen_circulars.add(files_key)
                    unique_circulars.append(issue)
            else:
                unique_circulars.append(issue)
        
        # Categorize by severity
        by_severity = defaultdict(list)
        for issue in unique_circulars:
            by_severity[issue['severity']].append(issue)
        
        report = {
            'project': str(self.project_path),
            'timestamp': str(Path.cwd()),
            'summary': {
                'total_files': len(self.imports_by_file),
                'total_issues': len(unique_circulars),
                'by_severity': {
                    'high': len(by_severity['high']),
                    'medium': len(by_severity['medium']),
                    'low': len(by_severity['low'])
                },
                'by_type': {
                    'phantom_functions': len(self.stub_functions),
                    'duplicate_implementations': len(self.duplicate_logic),
                    'circular_dependencies': len([i for i in unique_circulars if i['type'] == 'circular_dependency']),
                    'context_thrashing': len([i for i in self.issues if i['type'] == 'context_window_thrashing'])
                }
            },
            'issues': unique_circulars
        }
        
        return report

def main():
    """Run analysis on PoT codebase."""
    analyzer = TailChasingAnalyzer('/Users/rohanvinaik/PoT_Experiments')
    report = analyzer.analyze()
    
    # Save report
    output_file = '/Users/rohanvinaik/PoT_Experiments/tail_chasing_report.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TAIL-CHASING ANALYSIS REPORT")
    print("="*60)
    print(f"Project: {report['project']}")
    print(f"Total Files Analyzed: {report['summary']['total_files']}")
    print(f"Total Issues Found: {report['summary']['total_issues']}")
    print("\nIssues by Severity:")
    for severity, count in report['summary']['by_severity'].items():
        print(f"  - {severity.upper()}: {count}")
    print("\nIssues by Type:")
    for issue_type, count in report['summary']['by_type'].items():
        print(f"  - {issue_type}: {count}")
    
    # Show top issues
    if report['issues']:
        print("\nTop Issues:")
        for issue in report['issues'][:5]:
            print(f"\n  [{issue['severity'].upper()}] {issue['type']}")
            print(f"  {issue['message']}")
            if 'file' in issue:
                print(f"  File: {Path(issue['file']).name}")
            if 'files' in issue:
                print(f"  Files: {', '.join(Path(f).name for f in issue['files'])}")
    
    print(f"\nFull report saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()