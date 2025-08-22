# Sequential Verification Visualization Tools

This module provides comprehensive visualization tools for sequential verification processes in the Proof-of-Training framework.

## Overview

The `visualize_sequential.py` module implements four main visualization functions plus an interactive demo:

1. **`plot_verification_trajectory()`** - Visualize single verification runs
2. **`plot_operating_characteristics()`** - Compare sequential vs fixed-sample performance  
3. **`plot_anytime_validity()`** - Demonstrate validity across multiple runs
4. **`create_interactive_demo()`** - Streamlit-based interactive exploration
5. **Helper functions** - Configuration, styling, and utilities

## Core Functions

### 1. Trajectory Visualization

```python
from pot.core.visualize_sequential import plot_verification_trajectory
from pot.core.sequential import sequential_verify

# Run sequential test
result = sequential_verify(
    stream=your_data_stream(),
    tau=0.05,
    alpha=0.05,
    beta=0.05
)

# Visualize trajectory
fig = plot_verification_trajectory(
    result=result,
    save_path='trajectory.png',
    show_details=True
)
```

**Features:**
- Running mean with confidence bounds
- Decision threshold (tau) visualization  
- H0/H1 decision regions
- Stopping point annotation
- Confidence bound evolution
- Efficiency metrics
- P-value display (if computed)

### 2. Operating Characteristics

```python
from pot.core.visualize_sequential import plot_operating_characteristics

fig = plot_operating_characteristics(
    tau=0.05,
    alpha=0.05, 
    beta=0.05,
    effect_sizes=[0.0, 0.02, 0.05, 0.08, 0.1],
    max_samples_fixed=1000
)
```

**Features:**
- Power curves (sequential vs fixed-sample)
- Expected stopping times
- Efficiency ratios
- Operating characteristic curves
- Multiple threshold comparisons

### 3. Anytime Validity

```python
from pot.core.visualize_sequential import plot_anytime_validity

# Generate multiple trajectories
trajectories = []
for i in range(50):
    result = sequential_verify(...)
    trajectories.append(result)

fig = plot_anytime_validity(
    trajectories=trajectories,
    alpha=0.05
)
```

**Features:**
- Multiple trajectory overlay
- Stopping time distributions
- Empirical error rate analysis
- Coverage maintenance verification
- Anytime validity demonstration

### 4. Interactive Demo

```python
# Install Streamlit first
# pip install streamlit

# Run interactive demo
streamlit run pot/core/visualize_sequential.py
```

**Features:**
- Real-time parameter adjustment
- Live sequential test visualization
- Educational annotations
- Multiple scenario comparison
- Power analysis tools
- Error rate exploration

## Configuration

```python
from pot.core.visualize_sequential import VisualizationConfig

# Publication-ready plots
config = VisualizationConfig(
    figsize=(8, 6),
    dpi=300,
    style='seaborn-whitegrid',
    palette='Set2',
    save_format='pdf'
)

# Presentation plots
config = VisualizationConfig(
    figsize=(12, 8),
    dpi=150,
    style='seaborn-darkgrid',
    palette='bright',
    show_grid=True
)

# Interactive plots
config = VisualizationConfig(
    interactive=True,
    theme='dark'
)
```

## Examples

### Basic Usage

```python
# Simple trajectory plot
from pot.core.visualize_sequential import demo_trajectory_plot
demo_trajectory_plot()

# Operating characteristics  
from pot.core.visualize_sequential import demo_operating_characteristics
demo_operating_characteristics()

# Anytime validity
from pot.core.visualize_sequential import demo_anytime_validity
demo_anytime_validity()
```

### Advanced Usage

```python
from pot.core.visualize_sequential import *
from pot.core.sequential import sequential_verify
import numpy as np

# Generate realistic data
def data_stream():
    for _ in range(500):
        yield np.random.normal(0.03, 0.02)

# Run sequential test
result = sequential_verify(
    stream=data_stream(),
    tau=0.05,
    alpha=0.01,
    beta=0.01,
    compute_p_value=True
)

# Custom visualization
config = VisualizationConfig(
    figsize=(10, 8),
    dpi=150,
    show_legend=True,
    save_format='png'
)

fig = plot_verification_trajectory(
    result=result,
    config=config,
    save_path='my_trajectory.png',
    show_details=True
)

print(f"Decision: {result.decision}")
print(f"Stopped at: {result.stopped_at} samples")
print(f"P-value: {result.p_value:.4f}")
```

### Batch Analysis

```python
# Compare different scenarios
scenarios = [
    {"name": "Low Effect", "true_mean": 0.02},
    {"name": "Medium Effect", "true_mean": 0.05}, 
    {"name": "High Effect", "true_mean": 0.08}
]

results = []
for scenario in scenarios:
    def scenario_stream():
        for _ in range(500):
            yield np.random.normal(scenario["true_mean"], 0.02)
    
    result = sequential_verify(
        stream=scenario_stream(),
        tau=0.05,
        alpha=0.05,
        beta=0.05
    )
    results.append(result)

# Plot anytime validity across scenarios
fig = plot_anytime_validity(results)
```

## Installation Requirements

### Core Requirements
```bash
pip install matplotlib seaborn numpy
```

### Optional (for enhanced features)
```bash
pip install plotly streamlit  # Interactive demos
pip install scipy            # Advanced statistics
```

### Font Issues
If you see Unicode glyph warnings, install additional fonts:
```bash
# macOS
brew install font-dejavu

# Ubuntu/Debian  
sudo apt-get install fonts-dejavu

# Or use matplotlib's font manager
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
```

## Testing

```python
# Run visualization tests
python -m pot.test_visualization

# Run example demonstrations
python -m pot.examples.visualization_examples

# Test individual functions
from pot.core.visualize_sequential import *
demo_trajectory_plot()
demo_operating_characteristics()
demo_anytime_validity()
```

## Performance Tips

1. **Large datasets**: Use `config.dpi=100` for faster rendering
2. **Multiple plots**: Close figures with `plt.close(fig)` to save memory
3. **Interactive mode**: Set `config.interactive=False` for batch processing
4. **High-quality**: Use `dpi=300` and `save_format='pdf'` for publications

## Troubleshooting

### Common Issues

**1. Import errors**
```python
# Run from project root
python -m pot.core.visualize_sequential

# Or add to path
import sys
sys.path.append('/path/to/PoT_Experiments')
```

**2. Empty trajectory error**
```python
# Ensure sequential_verify returns trajectory
result = sequential_verify(..., compute_trajectory=True)
```

**3. Font warnings**
```python
# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

**4. Streamlit issues**
```bash
# Install streamlit
pip install streamlit

# Run from correct directory
cd /path/to/PoT_Experiments
streamlit run pot/core/visualize_sequential.py
```

## Contributing

When adding new visualization features:

1. Follow the existing code structure
2. Handle both dict and SequentialState trajectory formats
3. Include comprehensive error handling
4. Add configuration options
5. Write tests in `test_visualization.py`
6. Update examples in `visualization_examples.py`
7. Document new features in this README

## Related Documentation

- [Sequential Testing Core](./sequential.py) - Main sequential verification
- [Confidence Bounds](./boundaries.py) - EB bounds implementation  
- [Advanced Features](../test_advanced_sequential.py) - Mixture tests, power analysis
- [Main Documentation](../../CLAUDE.md) - Complete framework overview