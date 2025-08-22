# 🛡️ Safe Large Model Testing Guide

## The Problem
Testing 34B+ parameter models (Yi-34B, Mixtral-8x22B) was causing catastrophic system failures:
- **118GB RAM usage** on 64GB systems
- Complete system freeze requiring hard restart
- Lost work and potential hardware damage

## The Solution
Complete memory-managed testing framework that runs the **FULL PoT pipeline** with throttling:

### ✅ **YES - Full PoT Pipeline Included:**
1. **Enhanced Diff Decision Framework** 
   - Empirical-Bernstein bounds
   - Confidence intervals (97.5-99%)
   - Effect size detection
   - Early stopping optimization

2. **Security Verification Suite**
   - Config hash verification (SHA-256)
   - TLSH fuzzy hashing (if available)
   - Tokenizer compatibility checks

3. **Provenance Auditing** (optional)
   - Merkle tree construction
   - Cryptographic root comparison

4. **KDF Challenge Generation**
   - Deterministic challenge creation
   - HMAC-SHA256 based PRF

5. **Teacher-Forced Scoring**
   - Actual PoT scoring methods
   - Proper log-likelihood computation

## Key Safety Features

### Memory Management
- **Sequential Loading**: Only ONE model in memory at a time
- **Automatic Cleanup**: Forces garbage collection between models
- **Memory Monitoring**: Real-time tracking with alerts
- **Emergency Kill**: Auto-terminates before system crash
- **Hard Limits**: Enforces max memory usage (default 40GB)

### Performance Optimization
- **8-bit Quantization**: Reduces memory by 75% (optional)
- **CPU Offloading**: Handles models larger than RAM
- **Thread Limiting**: Prevents CPU overload
- **Process Priority**: Runs at lower priority (nice)
- **Batch Size 1**: Minimal memory footprint

## Usage

### Quick Start (Recommended)
```bash
# Safe wrapper with automatic monitoring
./scripts/safe_test_large_models.sh /path/to/model1 /path/to/model2
```

### Direct Throttled Testing
```bash
python scripts/run_large_models_throttled.py \
  --model1 /path/to/yi-34b \
  --model2 /path/to/yi-34b-chat \
  --max-memory 40 \
  --enable-8bit \
  --enable-offload \
  --mode audit
```

### Memory Monitoring (Separate Terminal)
```bash
# Run this in another terminal while testing
python scripts/monitor_memory.py --warning 75 --critical 85 --kill 92
```

## Testing Modes

### QUICK_GATE (Default)
- 10-32 queries maximum
- 97.5% confidence
- 5-10 minutes runtime
- Suitable for initial screening

### AUDIT_GRADE
- 30-32 queries maximum  
- 99% confidence
- 10-20 minutes runtime
- Higher precision results

### EXTENDED
- 50-32 queries maximum
- 99.9% confidence
- 20-30 minutes runtime
- Maximum accuracy

## Memory Guidelines

### System Requirements
| Model Size | RAM Needed | Recommended | Max Memory Setting |
|------------|------------|-------------|-------------------|
| 7B params  | 16GB       | 32GB        | 20GB              |
| 13B params | 32GB       | 48GB        | 30GB              |
| 34B params | 48GB       | 64GB        | 40GB              |
| 70B params | 96GB       | 128GB       | 80GB              |

### Configuration Examples

**For 64GB System with 34B Models:**
```bash
python scripts/run_large_models_throttled.py \
  --model1 yi-34b \
  --model2 yi-34b-chat \
  --max-memory 40 \    # Leave 24GB for system
  --min-free 10 \      # Maintain 10GB free
  --enable-8bit \      # Essential for large models
  --enable-offload     # Use CPU+RAM together
```

**For 32GB System with 7B Models:**
```bash
python scripts/run_large_models_throttled.py \
  --model1 llama-2-7b \
  --model2 mistral-7b \
  --max-memory 20 \    # Conservative limit
  --min-free 8         # Keep system responsive
```

## Output Format

The throttled pipeline generates comprehensive results:

```json
{
  "model1": "/path/to/model1",
  "model2": "/path/to/model2",
  "final_decision": "DIFFERENT",
  "enhanced_diff": {
    "decision": "DIFFERENT",
    "confidence": 0.99,
    "effect_size": 0.884,
    "ci_lower": 0.234,
    "ci_upper": 0.567,
    "n_effective": 32
  },
  "security": {
    "config_hash": {
      "match": false
    },
    "fuzzy_hash": {
      "similarity": 0.234,
      "match": false
    }
  },
  "peak_memory_gb": 38.4,
  "success": true
}
```

## Troubleshooting

### "Memory limit exceeded" Error
- Reduce `--max-memory` parameter
- Enable 8-bit quantization with `--enable-8bit`
- Use smaller batch of queries

### "Model loading failed"
- Check model path exists
- Verify config.json present
- Ensure sufficient disk space for swap

### System Still Freezing
- Run memory monitor in separate terminal
- Reduce max-memory to 50% of RAM
- Use QUICK_GATE mode instead of AUDIT

### Results Say "UNDECIDED"
- Increase number of queries (within memory limits)
- Try AUDIT_GRADE mode for better precision
- Models may be genuinely borderline similar

## Safety Checklist

Before testing large models:
- [ ] Close unnecessary applications
- [ ] Save all work
- [ ] Start memory monitor
- [ ] Use safe wrapper script
- [ ] Set conservative memory limits
- [ ] Enable 8-bit quantization
- [ ] Have system monitor open

## Emergency Recovery

If system becomes unresponsive:
1. Wait 30 seconds - emergency kill may trigger
2. Try switching to another TTY (Ctrl+Alt+F2)
3. SSH from another machine and kill Python
4. Last resort: Hold power button for hard reset

## Technical Details

### What Happens During Testing

1. **Initialization**
   - Sets memory limits with ulimit
   - Configures thread restrictions
   - Starts memory monitoring

2. **Phase 1: Statistical Testing**
   - Generates challenges with KDF
   - Loads Model 1 → Evaluates → Unloads
   - Loads Model 2 → Evaluates → Unloads
   - Computes enhanced statistics

3. **Phase 2: Security Tests**
   - Config hash comparison (no model loading)
   - Fuzzy hash (sequential model loading)
   - Tokenizer compatibility check

4. **Phase 3: Final Decision**
   - Aggregates all evidence
   - Makes SAME/DIFFERENT determination
   - Saves comprehensive results

### Memory Flow
```
Start: 10GB used
Load Model 1: 45GB used → Evaluate → Unload
Cleanup: 12GB used
Load Model 2: 45GB used → Evaluate → Unload  
Cleanup: 10GB used
End: 10GB used
```

Never both models in memory simultaneously!

## Summary

This throttled system provides:
- ✅ **Full PoT pipeline** - not simplified
- ✅ **Memory safety** - prevents crashes
- ✅ **Production ready** - handles 34B+ models
- ✅ **Comprehensive results** - all PoT components
- ✅ **Emergency protection** - auto-recovery

The system appropriately throttles resource usage while maintaining the complete Proof-of-Training verification pipeline, ensuring you can safely test even the largest models without system crashes.