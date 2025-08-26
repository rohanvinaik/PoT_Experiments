#!/usr/bin/env python3
"""
Verify that all citations in the paper match entries in references.bib
"""

import re
import sys
from pathlib import Path

def extract_citations_from_paper(paper_path):
    """Extract all citation keys from the paper."""
    with open(paper_path, 'r') as f:
        content = f.read()
    
    # Find all citations in the format [@key] or [@key1; @key2]
    citation_pattern = r'@([a-zA-Z0-9\-_]+)'
    citations = re.findall(citation_pattern, content)
    
    # Remove duplicates and sort
    return sorted(set(citations))

def extract_keys_from_bib(bib_path):
    """Extract all entry keys from the bibliography."""
    with open(bib_path, 'r') as f:
        content = f.read()
    
    # Find all BibTeX entry keys
    entry_pattern = r'@\w+\{([a-zA-Z0-9\-_]+),'
    keys = re.findall(entry_pattern, content)
    
    return sorted(set(keys))

def main():
    """Verify citations match between paper and bibliography."""
    paper_path = Path('POT_PAPER_COMPLETE_UPDATED.md')
    bib_path = Path('references.bib')
    
    if not paper_path.exists():
        print(f"Error: Paper file '{paper_path}' not found")
        sys.exit(1)
    
    if not bib_path.exists():
        print(f"Error: Bibliography file '{bib_path}' not found")
        sys.exit(1)
    
    # Extract citations and keys
    paper_citations = extract_citations_from_paper(paper_path)
    bib_keys = extract_keys_from_bib(bib_path)
    
    print("=" * 60)
    print("CITATION VERIFICATION REPORT")
    print("=" * 60)
    
    print(f"\nCitations in paper: {len(paper_citations)}")
    print(f"Entries in bibliography: {len(bib_keys)}")
    
    # Check for missing entries
    missing_in_bib = set(paper_citations) - set(bib_keys)
    if missing_in_bib:
        print("\n❌ MISSING IN BIBLIOGRAPHY:")
        for citation in sorted(missing_in_bib):
            print(f"  - @{citation}")
    else:
        print("\n✅ All paper citations have bibliography entries")
    
    # Check for unused entries
    unused_in_bib = set(bib_keys) - set(paper_citations)
    if unused_in_bib:
        print("\n📝 UNUSED BIBLIOGRAPHY ENTRIES (can be removed):")
        for key in sorted(unused_in_bib):
            print(f"  - @{key}")
    
    # List all citations used
    print("\n📚 CITATIONS USED IN PAPER:")
    for i, citation in enumerate(paper_citations, 1):
        print(f"  {i:2d}. @{citation}")
    
    # Summary
    print("\n" + "=" * 60)
    if not missing_in_bib:
        print("✅ VERIFICATION PASSED: All citations are properly defined")
    else:
        print("❌ VERIFICATION FAILED: Missing bibliography entries found")
    print("=" * 60)
    
    return 0 if not missing_in_bib else 1

if __name__ == "__main__":
    sys.exit(main())