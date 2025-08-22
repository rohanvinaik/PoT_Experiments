# PoT Automated Report Generator

The `ReportGenerator` class provides comprehensive automated report generation for Proof-of-Training (PoT) experimental results. It supports multiple output formats, automatic discrepancy detection, and rich visualizations.

## Features

### 🔍 **Automated Analysis**
- **Executive Summary**: Key metrics overview with performance assessment
- **Statistical Analysis**: FAR/FRR calculations with confidence intervals  
- **Discrepancy Detection**: Automatic comparison with paper claims
- **Trend Analysis**: Performance over time and challenge difficulty

### 📊 **Rich Visualizations**
- **ROC Curves**: Receiver Operating Characteristic and DET curves
- **Distribution Plots**: Query count histograms and box plots
- **Confidence Intervals**: Bootstrap confidence interval visualization
- **Comparison Charts**: Actual vs claimed performance
- **Challenge Analysis**: Per-challenge family performance breakdown
- **Timeline Analysis**: Performance trends over time

### 📄 **Multiple Output Formats**
- **Markdown**: Complete textual analysis for documentation
- **HTML**: Interactive web-based reports with embedded plots
- **LaTeX**: Paper-ready tables for academic publications
- **JSON**: Machine-readable data for programmatic access

### 🎯 **Smart Features**
- **Automatic discrepancy detection** with severity classification
- **Reconciliation suggestions** for identified issues
- **Flexible data loading** from CSV/JSON files
- **Confidence interval calculations** using bootstrap methods
- **Custom paper claims** support

## Quick Start

### Basic Usage

```python
from pot.experiments.report_generator import ReportGenerator

# Load results and generate comprehensive reports
generator = ReportGenerator("path/to/results.json")
reports = generator.generate_all_reports()

# Open the HTML report for best viewing experience
print(f"View report: {generator.output_dir}/report.html")
```

### With Custom Paper Claims

```python
# Create custom claims file
claims = {
    "far": 0.01,
    "frr": 0.01, 
    "accuracy": 0.99,
    "average_queries": 10.0
}

generator = ReportGenerator("results.json", "claims.json")
reports = generator.generate_all_reports()
```

### Individual Report Types

```python
# Generate specific report formats
markdown = generator.generate_markdown_report()
latex_tables = generator.generate_latex_tables() 
html_report = generator.generate_html_report()
json_data = generator.generate_json_export()
plots = generator.generate_plots()
```

## Input Data Format

The report generator accepts experimental results in JSON or CSV format with the following structure:

### Required Fields
```json
{
  "far": 0.008,           // False Accept Rate
  "frr": 0.012,           // False Reject Rate  
  "accuracy": 0.994       // Overall accuracy
}
```

### Optional Fields
```json
{
  "queries": 8,                    // Number of queries used
  "processing_time": 0.234,        // Processing time in seconds
  "challenge_family": "vision:freq", // Challenge type
  "threshold": 0.05,               // Decision threshold
  "timestamp": "2024-01-15T10:30:00", // Experiment timestamp
  "experiment_id": "exp_001"       // Unique experiment identifier
}
```

### Multiple Records
The generator supports both single records and arrays of records:

```json
[
  {
    "experiment_id": "exp_001",
    "far": 0.008,
    "frr": 0.012,
    "accuracy": 0.994,
    "queries": 8,
    "challenge_family": "vision:freq"
  },
  {
    "experiment_id": "exp_002", 
    "far": 0.006,
    "frr": 0.015,
    "accuracy": 0.991,
    "queries": 12,
    "challenge_family": "vision:texture"
  }
]
```

## Output Structure

Generated reports are organized in timestamped directories:

```
reports/report_YYYYMMDD_HHMMSS/
├── README.md                    # Index of all reports
├── report.html                  # Interactive HTML report ⭐
├── report.md                    # Markdown analysis
├── report_data.json            # Machine-readable data
├── tables.tex                  # LaTeX tables for papers
├── roc_curve.png               # ROC/DET curves
├── query_distribution.png      # Query efficiency analysis
├── confidence_intervals.png    # Statistical reliability
├── performance_comparison.png  # Claims vs results
├── challenge_analysis.png      # Per-challenge breakdown
└── timeline_analysis.png       # Performance over time
```

## Advanced Features

### Custom Analysis

```python
# Access raw data for custom analysis
generator = ReportGenerator("results.json")

for result in generator.data:
    if 'custom_metric' in result:
        print(f"Custom analysis: {result['custom_metric']}")

# Access calculated metrics
print(f"Overall FAR: {generator.metrics.far:.4f}")
print(f"Confidence intervals: {generator.metrics.confidence_intervals}")
```

### Discrepancy Analysis

```python
# Review identified discrepancies
for discrepancy in generator.discrepancies:
    print(f"{discrepancy.metric}: {discrepancy.relative_difference:.1%} "
          f"({discrepancy.severity})")
    print(f"  Suggestion: {discrepancy.suggestion}")
```

### Integration with PoT Experiments

```python
# Integration with reproducible_runner results
from pot.experiments import ReproducibleExperimentRunner

# Run experiments
runner = ReproducibleExperimentRunner(config)
results = runner.run_experiment()

# Convert to report format
report_data = []
for result in results:
    record = {
        "far": result.far,
        "frr": result.frr,
        "accuracy": result.accuracy,
        "queries": result.queries,
        "challenge_family": result.challenge_family
    }
    report_data.append(record)

# Generate reports
with open("experiment_results.json", 'w') as f:
    json.dump(report_data, f)

generator = ReportGenerator("experiment_results.json")
reports = generator.generate_all_reports()
```

## API Reference

### ReportGenerator Class

```python
class ReportGenerator:
    def __init__(self, results_path: str, paper_claims_path: Optional[str] = None)
    
    # Report generation methods
    def generate_markdown_report() -> str
    def generate_html_report() -> str  
    def generate_latex_tables() -> str
    def generate_json_export() -> str
    def generate_plots() -> Dict[str, str]
    def generate_all_reports() -> Dict[str, str]
    
    # Data access
    def load_results(self, results_path: str) -> List[Dict[str, Any]]
    
    # Properties
    self.data: List[Dict[str, Any]]           # Raw experimental data
    self.metrics: ResultMetrics               # Calculated metrics
    self.discrepancies: List[Discrepancy]     # Identified issues
    self.paper_claims: PaperClaims           # Expected values
    self.output_dir: Path                    # Report output directory
```

### Data Classes

```python
@dataclass
class ResultMetrics:
    far: float
    frr: float  
    accuracy: float
    total_queries: int
    avg_queries: float
    processing_time: float
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class Discrepancy:
    metric: str
    claimed: float
    actual: float
    difference: float
    relative_difference: float
    severity: str  # 'minor', 'moderate', 'major'
    suggestion: str
```

## Usage Examples

See `example_report_generation.py` for comprehensive usage examples including:
- Basic report generation
- Custom paper claims
- Integration with PoT experiments  
- Custom analysis extensions

## Best Practices

### 1. Data Quality
- Ensure consistent field names across result records
- Include timestamps for timeline analysis
- Use meaningful experiment IDs for traceability

### 2. Report Interpretation
- Review the HTML report first for interactive exploration
- Use LaTeX tables for academic publications
- Export JSON data for programmatic analysis
- Check discrepancy analysis for potential issues

### 3. Performance
- For large datasets, consider data sampling
- Use directory-based loading for multiple files
- Cache frequently accessed reports

### 4. Customization
- Provide custom paper claims for accurate comparisons
- Include challenge family information for detailed analysis
- Add custom metrics as needed for specific analyses

## Troubleshooting

### Common Issues

1. **Missing Data Fields**
   ```
   Warning: No 'far' field found in results
   ```
   Solution: Ensure your data includes required FAR/FRR/accuracy fields

2. **Visualization Errors**
   ```
   ERROR: 'yerr' must not contain negative values
   ```
   Solution: Check confidence interval calculations for negative values

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'pot'
   ```
   Solution: Set PYTHONPATH or install the pot package properly

### Dependencies

Required packages:
- `numpy`, `pandas`, `matplotlib`, `seaborn` - Core functionality
- `tabulate` - Table formatting (optional)
- `plotly` - Interactive plots (optional)

## Testing

Run the test suite to validate functionality:

```bash
python -m pytest pot/experiments/test_report_generator.py -v
```

## Future Enhancements

- Interactive Plotly visualizations
- PDF report generation
- Email report distribution
- Real-time dashboard integration
- Advanced statistical tests
- Custom template support

---

For questions or contributions, see the main PoT repository documentation.