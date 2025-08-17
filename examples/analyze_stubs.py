#!/usr/bin/env python3
"""
Analyze and catalog all stubs in the PoT codebase
"""

import ast
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

class StubAnalyzer(ast.NodeVisitor):
    """Analyze Python AST to find stubs and placeholders"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.stubs = []
        self.current_class = None
        
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node):
        self.check_function(node)
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        self.check_function(node)
        self.generic_visit(node)
        
    def check_function(self, node):
        """Check if function is a stub"""
        stub_info = {
            'file': self.filepath,
            'function': node.name,
            'class': self.current_class,
            'line': node.lineno,
            'type': None,
            'priority': 'LOW',
            'description': ''
        }
        
        # Get function body
        if not node.body:
            return
            
        # Check for various stub patterns
        body_str = ast.unparse(node.body) if hasattr(ast, 'unparse') else str(node.body)
        
        # Pattern 1: Only pass statement
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            stub_info['type'] = 'pass_only'
            stub_info['description'] = 'Function contains only pass statement'
            self.stubs.append(stub_info)
            return
            
        # Pattern 2: Only docstring and pass
        if len(node.body) <= 2:
            has_docstring = False
            has_pass = False
            
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    has_docstring = True
                elif isinstance(stmt, ast.Pass):
                    has_pass = True
                    
            if has_docstring and has_pass:
                stub_info['type'] = 'docstring_pass'
                stub_info['description'] = 'Function has only docstring and pass'
                self.stubs.append(stub_info)
                return
                
        # Pattern 3: Raises NotImplementedError
        for stmt in node.body:
            if isinstance(stmt, ast.Raise):
                if (isinstance(stmt.exc, ast.Call) and 
                    isinstance(stmt.exc.func, ast.Name) and 
                    stmt.exc.func.id == 'NotImplementedError'):
                    stub_info['type'] = 'not_implemented'
                    stub_info['description'] = 'Raises NotImplementedError'
                    stub_info['priority'] = 'HIGH'
                    self.stubs.append(stub_info)
                    return
                    
        # Pattern 4: Returns placeholder value with comment
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                # Check for placeholder comments
                if 'placeholder' in body_str.lower() or 'todo' in body_str.lower():
                    stub_info['type'] = 'placeholder_return'
                    stub_info['description'] = 'Returns placeholder value'
                    stub_info['priority'] = 'MEDIUM'
                    self.stubs.append(stub_info)
                    return
                    
                # Check for trivial returns
                if isinstance(stmt.value, ast.Constant):
                    if stmt.value.value in [True, False, None, "", 0, []]:
                        # Check if it's the only statement (besides docstring)
                        non_doc_stmts = [s for s in node.body 
                                        if not (isinstance(s, ast.Expr) and 
                                               isinstance(s.value, ast.Constant))]
                        if len(non_doc_stmts) == 1:
                            stub_info['type'] = 'trivial_return'
                            stub_info['description'] = f'Returns trivial value: {stmt.value.value}'
                            self.stubs.append(stub_info)
                            return


def analyze_file(filepath: Path) -> List[Dict]:
    """Analyze a single Python file for stubs"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        tree = ast.parse(content)
        analyzer = StubAnalyzer(str(filepath.relative_to(Path.cwd())))
        analyzer.visit(tree)
        
        # Also check for comment-based markers
        for i, line in enumerate(content.split('\n'), 1):
            if any(marker in line for marker in ['TODO:', 'FIXME:', 'XXX:', 'HACK:']):
                analyzer.stubs.append({
                    'file': str(filepath.relative_to(Path.cwd())),
                    'function': None,
                    'class': None,
                    'line': i,
                    'type': 'comment_marker',
                    'priority': 'MEDIUM',
                    'description': line.strip()
                })
                
        return analyzer.stubs
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return []


def categorize_stubs(stubs: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize stubs by area and priority"""
    categories = {
        'security': [],
        'vision': [],
        'language_model': [],
        'core': [],
        'evaluation': [],
        'scripts': [],
        'other': []
    }
    
    for stub in stubs:
        filepath = stub['file']
        
        # Categorize by file path
        if 'security' in filepath:
            categories['security'].append(stub)
        elif 'vision' in filepath:
            categories['vision'].append(stub)
        elif 'lm' in filepath or 'language' in filepath:
            categories['language_model'].append(stub)
        elif 'core' in filepath:
            categories['core'].append(stub)
        elif 'eval' in filepath:
            categories['evaluation'].append(stub)
        elif 'scripts' in filepath:
            categories['scripts'].append(stub)
        else:
            categories['other'].append(stub)
            
    return categories


def main():
    """Main analysis function"""
    print("="*80)
    print("COMPREHENSIVE STUB ANALYSIS FOR PoT EXPERIMENTS")
    print("="*80)
    
    # Directories to analyze
    dirs_to_analyze = ['pot', 'scripts']
    
    all_stubs = []
    
    for dir_name in dirs_to_analyze:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            continue
            
        for py_file in dir_path.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue
            stubs = analyze_file(py_file)
            all_stubs.extend(stubs)
    
    # Categorize stubs
    categorized = categorize_stubs(all_stubs)
    
    # Print report
    print(f"\nTotal stubs found: {len(all_stubs)}")
    print("\n" + "="*80)
    
    # Priority summary
    priority_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for stub in all_stubs:
        priority_counts[stub['priority']] += 1
    
    print("\nPRIORITY SUMMARY:")
    print(f"  HIGH:   {priority_counts['HIGH']} (Critical - implement immediately)")
    print(f"  MEDIUM: {priority_counts['MEDIUM']} (Important - implement soon)")
    print(f"  LOW:    {priority_counts['LOW']} (Nice to have - implement eventually)")
    
    # Detailed report by category
    for category, stubs in categorized.items():
        if not stubs:
            continue
            
        print("\n" + "="*80)
        print(f"{category.upper().replace('_', ' ')} STUBS ({len(stubs)} total)")
        print("-"*80)
        
        # Sort by priority
        sorted_stubs = sorted(stubs, key=lambda x: ['HIGH', 'MEDIUM', 'LOW'].index(x['priority']))
        
        for stub in sorted_stubs:
            location = f"{stub['file']}:{stub['line']}"
            if stub['class']:
                func_name = f"{stub['class']}.{stub['function']}"
            else:
                func_name = stub['function'] or 'N/A'
            
            print(f"\n[{stub['priority']}] {location}")
            print(f"  Function: {func_name}")
            print(f"  Type: {stub['type']}")
            print(f"  Description: {stub['description']}")
    
    # Generate implementation priority list
    print("\n" + "="*80)
    print("IMPLEMENTATION PRIORITY LIST")
    print("="*80)
    
    high_priority = [s for s in all_stubs if s['priority'] == 'HIGH']
    
    print("\n🔴 CRITICAL (Implement immediately):")
    for i, stub in enumerate(high_priority[:10], 1):
        location = f"{stub['file']}:{stub['line']}"
        func = f"{stub['class']}.{stub['function']}" if stub['class'] else stub['function']
        print(f"  {i}. {func} ({location})")
        print(f"     {stub['description']}")
    
    # Security-specific stubs
    security_stubs = categorized['security']
    if security_stubs:
        print("\n🔐 SECURITY-CRITICAL:")
        for stub in security_stubs[:5]:
            location = f"{stub['file']}:{stub['line']}"
            func = f"{stub['class']}.{stub['function']}" if stub['class'] else stub['function']
            print(f"  - {func} ({location})")
    
    # Save detailed report
    import json
    report = {
        'total_stubs': len(all_stubs),
        'priority_summary': priority_counts,
        'categorized': {
            cat: [{k: v for k, v in s.items()} for s in stubs]
            for cat, stubs in categorized.items()
        }
    }
    
    with open('stub_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n\nDetailed report saved to: stub_analysis_report.json")
    
    return all_stubs


if __name__ == "__main__":
    stubs = main()