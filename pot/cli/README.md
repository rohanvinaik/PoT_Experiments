# PoT Attack CLI Documentation

Command-line interface for Proof-of-Training attack resistance evaluation.

## Installation

### Quick Install
```bash
# Install CLI and dependencies
pip install -e .

# Or install with extras
pip install -e ".[dashboard,advanced]"
```

### Docker Installation
```bash
# Build Docker image
docker build -f Dockerfile.attacks -t pot-attacks .

# Run with Docker
docker run --gpus all pot-attacks run-attacks -m model.pth
```

## Commands

### run-attacks
Run attack suite against a model.

```bash
pot-attack run-attacks -m model.pth -s standard -o results/

# Options:
#   -m, --model-path    Path to model checkpoint (required)
#   -s, --attack-suite  Attack suite: standard/adaptive/comprehensive/quick/vision
#   -o, --output-dir    Output directory for results
#   -d, --data-path     Path to test data
#   --device            Device: cpu/cuda/mps
#   --batch-size        Batch size for evaluation
#   --num-samples       Number of samples to use
#   --include-defenses  Include defense evaluation
#   -c, --config        Custom configuration file
```

#### Examples
```bash
# Quick evaluation
pot-attack run-attacks -m model.pth -s quick

# Comprehensive evaluation with defenses
pot-attack run-attacks -m model.pth -s comprehensive --include-defenses

# Custom configuration
pot-attack run-attacks -m model.pth -c custom_attacks.yaml

# GPU evaluation with specific data
pot-attack run-attacks -m model.pth -d test_data.pt --device cuda
```

### generate-report
Generate reports from attack results.

```bash
pot-attack generate-report -r results/ -o report.html

# Options:
#   -r, --results-dir   Directory containing results (required)
#   -o, --output-file   Output report filename
#   --include-plots     Include interactive plots
#   --format           Report format: html/pdf/markdown
```

#### Examples
```bash
# HTML report with plots
pot-attack generate-report -r results/ --include-plots

# Markdown report
pot-attack generate-report -r results/ --format markdown

# PDF report (requires additional dependencies)
pot-attack generate-report -r results/ --format pdf -o report.pdf
```

### detect-wrapper
Detect if a model has been wrapped.

```bash
pot-attack detect-wrapper -m model.pth -b baseline.pth

# Options:
#   -m, --model-path     Path to model to test (required)
#   -b, --baseline-path  Path to baseline model (required)
#   -n, --samples        Number of samples for detection
#   -o, --output         Output file for results
#   -t, --threshold      Detection threshold
#   -M, --methods        Detection methods to use
```

#### Examples
```bash
# Basic wrapper detection
pot-attack detect-wrapper -m suspicious.pth -b original.pth

# Detailed detection with more samples
pot-attack detect-wrapper -m model.pth -b baseline.pth -n 5000 -o detection.json

# Custom threshold and methods
pot-attack detect-wrapper -m model.pth -b baseline.pth -t 0.2 -M timing -M behavioral
```

### benchmark
Run standardized benchmarks.

```bash
pot-attack benchmark -c config.yaml -m model.pth

# Options:
#   -c, --config-file        Benchmark configuration (required)
#   -m, --model-path         Model to benchmark
#   -o, --output-dir         Output directory
#   -C, --compare            Additional models to compare
#   -a, --attacks            Comma-separated list of attacks
#   --device                 Device: cpu/cuda/mps
#   --generate-leaderboard   Generate comparison leaderboard
```

#### Examples
```bash
# Single model benchmark
pot-attack benchmark -c benchmark_config.yaml -m model.pth

# Compare multiple models
pot-attack benchmark -c config.yaml -m model1.pth -C model2.pth -C model3.pth --generate-leaderboard

# Specific attacks only
pot-attack benchmark -c config.yaml -m model.pth -a "distillation_weak,compression_pruning_50"

# GPU benchmark with custom output
pot-attack benchmark -c config.yaml -m model.pth --device cuda -o benchmark_results/
```

### verify
Verify model with PoT and evaluate attack resistance.

```bash
pot-attack verify -m model.pth -r reference.pth

# Options:
#   -m, --model-path       Model to verify (required)
#   -r, --reference-path   Reference model (required)
#   -p, --profile          Verification profile: quick/standard/comprehensive
#   -s, --security-level   Security level: low/medium/high
#   -o, --output           Output file for results
```

#### Examples
```bash
# Standard verification
pot-attack verify -m model.pth -r reference.pth

# Comprehensive verification with high security
pot-attack verify -m model.pth -r reference.pth -p comprehensive -s high

# Quick verification with output
pot-attack verify -m model.pth -r reference.pth -p quick -o verification.json
```

### dashboard
Launch interactive dashboard for metrics visualization.

```bash
pot-attack dashboard -r benchmark_results/

# Options:
#   -p, --port         Port for dashboard server
#   -r, --results-dir  Directory with benchmark results
#   -h, --host         Host to bind to
#   --debug            Run in debug mode
```

#### Examples
```bash
# Launch dashboard on default port
pot-attack dashboard -r results/

# Custom port and host
pot-attack dashboard -r results/ -p 8080 -h 0.0.0.0

# Debug mode
pot-attack dashboard -r results/ --debug
```

## Configuration Files

### Attack Configuration (YAML)
```yaml
attacks:
  suite: standard
  configs:
    - name: distillation_custom
      temperature: 5.0
      epochs: 20
    - name: compression_custom
      pruning_rate: 0.5
      quantization_bits: 8

defense:
  enabled: true
  adaptive_threshold: 0.05
  
evaluation:
  batch_size: 32
  num_samples: 1000
  device: cuda
```

### Benchmark Configuration (YAML)
```yaml
models:
  - path: model1.pth
    name: Model_A
  - path: model2.pth
    name: Model_B

attacks:
  - distillation_weak
  - distillation_strong
  - compression_pruning_50
  - wrapper_naive

metrics:
  - robustness_score
  - attack_success_rate
  - execution_time

output:
  format: html
  include_plots: true
  save_raw_data: true
```

## Docker Usage

### Build Image
```bash
docker build -f Dockerfile.attacks -t pot-attacks .
```

### Run Commands
```bash
# Run attacks
docker run --gpus all -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  pot-attacks run-attacks -m /app/models/model.pth

# Launch dashboard
docker run -p 8050:8050 -v $(pwd)/results:/app/results \
  pot-attacks dashboard -r /app/results
```

### Docker Compose
```bash
# Start all services
docker-compose -f docker-compose.attacks.yml up

# Run specific service
docker-compose -f docker-compose.attacks.yml run attack-evaluator \
  run-attacks -m /app/models/model.pth

# View dashboard
# Navigate to http://localhost:8050
```

## Integration with Training

### PyTorch Integration
```python
from pot.train.attack_integration import AttackResistanceMonitor

# Create monitor
monitor = AttackResistanceMonitor(
    evaluation_frequency=10,
    attack_suite='quick'
)

# In training loop
for epoch in range(epochs):
    train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    # Evaluate attack resistance
    evaluation = monitor.evaluate(model, epoch, val_loader)
    
    if evaluation:
        print(f"Robustness: {evaluation['robustness_score']:.1f}")
```

### PyTorch Lightning
```python
from pot.train.attack_integration import TrainingIntegration

# Create callback
monitor = AttackResistanceMonitor()
callback = TrainingIntegration.integrate_with_pytorch_lightning(monitor)

# Add to trainer
trainer = pl.Trainer(callbacks=[callback])
```

### Transformers
```python
from pot.train.attack_integration import TrainingIntegration

# Create callback
monitor = AttackResistanceMonitor()
callback = TrainingIntegration.integrate_with_transformers(monitor)

# Add to trainer
trainer = Trainer(
    model=model,
    callbacks=[callback]
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | 0 |
| `POT_CONFIG_DIR` | Configuration directory | ./configs |
| `POT_RESULTS_DIR` | Results directory | ./results |
| `POT_LOG_LEVEL` | Logging level | INFO |
| `WANDB_API_KEY` | Weights & Biases API key | - |

## Output Files

### Results Directory Structure
```
results/
├── results.json          # Raw attack results
├── metrics.json          # Computed metrics
├── results.csv           # Tabular results
├── summary.json          # Evaluation summary
├── dashboard.html        # Interactive dashboard
└── plots/               # Generated plots
    ├── success_rates.png
    ├── robustness.png
    └── far_frr.png
```

### Report Formats

#### HTML Report
- Interactive plots with Plotly
- Sortable tables
- Downloadable data

#### PDF Report
- Static document
- Print-friendly format
- Embedded charts

#### Markdown Report
- GitHub-compatible
- Easy to version control
- Plain text with tables

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 16`
   - Use CPU: `--device cpu`
   - Clear cache in code

2. **Module Not Found**
   - Install package: `pip install -e .`
   - Set PYTHONPATH: `export PYTHONPATH=$PWD:$PYTHONPATH`

3. **Permission Denied (Docker)**
   - Add user to docker group: `sudo usermod -aG docker $USER`
   - Use sudo: `sudo docker run ...`

4. **Dashboard Not Loading**
   - Check port availability: `lsof -i :8050`
   - Try different port: `-p 8080`
   - Check firewall settings

## Advanced Usage

### Custom Attack Implementation
```python
# In custom_attacks.py
from pot.core.attack_suites import BaseAttack

class CustomAttack(BaseAttack):
    def execute(self, model, data_loader):
        # Implementation
        return {'success': True, 'confidence': 0.8}

# Register and use
CUSTOM_ATTACKS = [CustomAttack()]
```

### Parallel Execution
```bash
# Run multiple benchmarks in parallel
parallel -j 4 pot-attack benchmark -c config.yaml -m ::: model1.pth model2.pth model3.pth model4.pth
```

### Batch Processing
```bash
# Process all models in directory
for model in models/*.pth; do
    pot-attack run-attacks -m "$model" -s quick -o "results/$(basename $model .pth)/"
done
```

## Support

- GitHub Issues: [Report bugs](https://github.com/pot/attacks/issues)
- Documentation: [Full docs](https://pot-attacks.readthedocs.io)
- Examples: See `examples/` directory