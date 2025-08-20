#!/usr/bin/env python3
"""
Update Comprehensive Evidence Dashboard in README
Auto-generates comprehensive performance metrics with Statistical, Performance, Cryptographic, and Experimental Setup data
Replaces placeholder with full dashboard content from evidence logging system
"""

import json
import datetime
import pathlib
import sys
import os
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pot.core.evidence_logger import EvidenceLogger


def update_readme_evidence_section():
    """Update the comprehensive metrics placeholder in README"""
    
    readme_path = pathlib.Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        print("❌ README.md not found")
        return False
    
    # Initialize evidence logger to get latest comprehensive metrics
    logger = EvidenceLogger()
    logger.update_evidence_dashboard()
    
    # Load rolling metrics
    metrics_file = pathlib.Path("experimental_results") / "rolling_metrics.json"
    if not metrics_file.exists():
        print("❌ No rolling metrics data available")
        return False
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Get recent validation runs for comprehensive analysis
    recent_runs = logger._get_recent_runs(limit=20)
    
    # Generate comprehensive dashboard content using the evidence logger
    comprehensive_dashboard = logger._generate_dashboard_content(metrics, recent_runs)
    
    # Read current README
    with open(readme_path) as f:
        readme_content = f.read()
    
    # Find and replace the comprehensive metrics placeholder
    placeholder = "**COMPREHENSIVE_METRICS_PLACEHOLDER**"
    
    if placeholder not in readme_content:
        print("❌ Could not find comprehensive metrics placeholder in README")
        return False
    
    # Replace the placeholder with comprehensive dashboard
    updated_readme = readme_content.replace(placeholder, comprehensive_dashboard)
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(updated_readme)
    
    print(f"✅ Updated README.md with comprehensive dashboard ({metrics['total_runs']} total runs)")
    print(f"📊 Dashboard includes: Statistical, Performance, Cryptographic, and Setup metrics")
    return True




def main():
    """Update comprehensive evidence dashboard and README"""
    print("🔄 Updating Comprehensive Evidence Dashboard and README...")
    
    # Update evidence dashboard files
    logger = EvidenceLogger()
    logger.update_evidence_dashboard()
    
    # Update README with comprehensive metrics
    success = update_readme_evidence_section()
    
    if success:
        print("✅ Comprehensive evidence dashboard and README updated successfully")
        print(f"📊 View standalone dashboard: cat EVIDENCE_DASHBOARD.md")
        print(f"📈 View integrated README dashboard: README.md (section: Live Performance Dashboard)")
    else:
        print("❌ Failed to update README")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())