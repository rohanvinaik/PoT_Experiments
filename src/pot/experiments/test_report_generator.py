#!/usr/bin/env python3
"""
Test suite for the ReportGenerator class.
"""

import unittest
import json
import tempfile
from pathlib import Path
from pot.experiments.report_generator import ReportGenerator, create_sample_results

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_file = Path(self.temp_dir) / "test_results.json"
        create_sample_results(str(self.sample_file))
        
    def test_data_loading(self):
        """Test data loading from JSON files."""
        generator = ReportGenerator(str(self.sample_file))
        self.assertEqual(len(generator.data), 3)
        self.assertIn('far', generator.data[0])
        self.assertIn('frr', generator.data[0])
        
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        generator = ReportGenerator(str(self.sample_file))
        metrics = generator.metrics
        
        # Check that metrics are calculated
        self.assertGreater(metrics.far, 0)
        self.assertGreater(metrics.frr, 0)
        self.assertGreater(metrics.accuracy, 0)
        self.assertGreater(metrics.avg_queries, 0)
        
    def test_discrepancy_detection(self):
        """Test discrepancy detection."""
        generator = ReportGenerator(str(self.sample_file))
        
        # Should have some discrepancies with default paper claims
        self.assertGreater(len(generator.discrepancies), 0)
        
        # Check discrepancy structure
        for discrepancy in generator.discrepancies:
            self.assertIn(discrepancy.severity, ['minor', 'moderate', 'major'])
            self.assertIsInstance(discrepancy.metric, str)
            self.assertIsInstance(discrepancy.suggestion, str)
            
    def test_markdown_generation(self):
        """Test markdown report generation."""
        generator = ReportGenerator(str(self.sample_file))
        markdown = generator.generate_markdown_report()
        
        self.assertIn("# PoT Experimental Results Report", markdown)
        self.assertIn("Executive Summary", markdown)
        self.assertIn("FAR", markdown)
        self.assertIn("FRR", markdown)
        
    def test_latex_generation(self):
        """Test LaTeX table generation."""
        generator = ReportGenerator(str(self.sample_file))
        latex = generator.generate_latex_tables()
        
        self.assertIn("\\begin{table}", latex)
        self.assertIn("\\toprule", latex)
        self.assertIn("FAR", latex)
        
    def test_json_export(self):
        """Test JSON export functionality."""
        generator = ReportGenerator(str(self.sample_file))
        json_str = generator.generate_json_export()
        
        # Parse JSON to verify structure
        data = json.loads(json_str)
        self.assertIn("report_metadata", data)
        self.assertIn("metrics", data)
        self.assertIn("paper_claims", data)
        self.assertIn("discrepancies", data)
        
    def test_html_generation(self):
        """Test HTML report generation."""
        generator = ReportGenerator(str(self.sample_file))
        html = generator.generate_html_report()
        
        self.assertIn("<html", html)
        self.assertIn("PoT Experimental Results Report", html)
        self.assertIn("False Accept Rate", html)
        
    def test_complete_report_suite(self):
        """Test generation of complete report suite."""
        generator = ReportGenerator(str(self.sample_file))
        reports = generator.generate_all_reports()
        
        # Check that all expected report types are generated
        expected_types = ['markdown', 'latex', 'html', 'json', 'index']
        for report_type in expected_types:
            self.assertIn(report_type, reports)
            
        # Check that plot files are generated
        plot_types = ['roc_curve', 'query_distribution', 'confidence_intervals', 
                     'performance_comparison', 'challenge_analysis', 'timeline_analysis']
        for plot_type in plot_types:
            self.assertIn(plot_type, reports)


if __name__ == '__main__':
    unittest.main()