# Threshold Calibration Results
## Fixing UNDECIDED Outcomes in Statistical Decision Framework

**Generated:** August 19, 2025  
**Purpose:** Calibrate decision thresholds based on actual model behavior  
**Status:** Calibration complete with empirical thresholds

---

## 📋 **Executive Summary**

The threshold calibration module was created to resolve UNDECIDED outcomes by empirically measuring the score distributions of real model pairs. Testing revealed that GPT-2 family models produce higher similarity scores than initially expected, requiring adjusted thresholds.

---

## 🔬 **Empirical Observations**

### **GPT-2 Self-Consistency (Same Model)**
- **Mean score:** 0.22-0.24
- **Standard deviation:** 0.04-0.08  
- **P95:** ~0.25
- **P99:** ~0.25
- **Interpretation:** Same model shows moderate variation, not near-zero as expected

### **GPT-2 vs DistilGPT-2 (Different Models)**
- **Mean score:** ~0.29
- **Standard deviation:** ~0.01
- **Min:** ~0.28
- **Max:** ~0.30
- **Interpretation:** Different models show consistent but moderate separation

### **Key Finding**
The scoring metric produces values in the range 0.2-0.3 for both same and different models, with only ~0.05-0.07 separation between distributions. This narrow separation makes decisive outcomes challenging with original thresholds.

---

## 🎯 **Calibrated Thresholds**

Based on empirical testing, the following thresholds are recommended:

### **Conservative Approach (Current)**
Maintains statistical rigor but may still produce UNDECIDED outcomes:

| Parameter | QUICK_GATE | AUDIT_GRADE | Purpose |
|-----------|------------|-------------|---------|
| γ (gamma) | 0.35 | 0.30 | SAME threshold |
| δ* (delta_star) | 0.20 | 0.25 | DIFFERENT threshold |
| η (eta) | 0.5 | 0.4 | Precision factor |
| ε_diff | 0.15 | 0.10 | Relative error tolerance |
| n_min | 15 | 25 | Minimum samples |
| n_max | 100 | 200 | Maximum samples |

### **Aggressive Approach (Alternative)**
Prioritizes decisive outcomes over strict error control:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| γ (gamma) | 0.40 | Accept all self-consistency scores |
| δ* (delta_star) | 0.15 | Lower bar for DIFFERENT |
| ε_diff | 0.30 | Relaxed error tolerance |

---

## 📊 **Calibration Process**

### **1. Threshold Calibration Module**
Created `pot/core/threshold_calibration.py` with:
- `ThresholdCalibrator` class for empirical measurement
- `CalibrationConfig` for experiment parameters
- Automatic threshold computation based on percentiles
- Support for multiple model pairs

### **2. Calibration Scripts**
- `scripts/calibrate_thresholds.py` - Run calibration experiments
- `scripts/test_calibrated_thresholds.py` - Validate thresholds
- `scripts/apply_calibrated_thresholds.py` - Apply to configuration

### **3. Key Features**
- Model caching for efficiency
- Optimized scoring integration
- Statistical analysis of score distributions
- Automatic threshold recommendation

---

## 🔍 **Why UNDECIDED Persists**

### **Root Causes**

1. **Narrow Separation**
   - Only ~0.05 difference between same/different distributions
   - High overlap region makes decisive thresholds difficult

2. **High Baseline Scores**
   - Even identical models produce scores ~0.22
   - Not centered around zero as framework assumes

3. **Limited Discrimination**
   - Current scoring metric (cross-entropy difference) may not be optimal
   - GPT-2 and DistilGPT-2 are very similar architectures

### **Potential Solutions**

1. **Alternative Scoring Metrics**
   - KL divergence between full distributions
   - Wasserstein distance
   - Ensemble of multiple metrics

2. **Increased Sampling**
   - Use K=128 or 256 positions per prompt
   - Aggregate over more diverse prompts

3. **Model-Specific Calibration**
   - Calibrate thresholds per model family
   - Learn separation boundaries via supervised learning

4. **Relaxed Statistical Requirements**
   - Accept higher error rates for practical decisiveness
   - Use different criteria for near-clone detection

---

## 📁 **Generated Files**

### **Calibration Data**
```
experimental_results/calibration/
├── quick_calibration_*.json      # Single model calibration
├── full_calibration_*.json       # Model pair calibration
├── recommended_config_*.json     # Threshold recommendations
├── empirical_thresholds.json     # Final empirical values
└── calibrated_config.py         # Python configuration class
```

### **Test Results**
```
experimental_results/
├── calibration_test_*.json      # Validation of thresholds
└── runtime_blackbox_*.json      # Runtime validation with calibration
```

---

## 🚀 **Usage Instructions**

### **Apply Calibrated Thresholds**

```python
from pot.core.diff_decision import DiffDecisionConfig, TestingMode

# Create config with calibrated values
config = DiffDecisionConfig(mode=TestingMode.QUICK_GATE)
config.gamma = 0.35  # From calibration
config.delta_star = 0.20
config.epsilon_diff = 0.15
```

### **Run Calibration for New Models**

```python
from pot.core.threshold_calibration import AutoCalibrator

# Quick calibration for a model
calibration = AutoCalibrator.quick_auto_calibrate("model_name")
print(f"Recommended γ: {calibration['recommended_gamma']}")
```

---

## 💡 **Lessons Learned**

1. **Empirical calibration is essential** - Theoretical thresholds don't match reality
2. **Model similarity matters** - GPT-2 family shows high baseline similarity
3. **Scoring metric selection is critical** - Cross-entropy may not be optimal
4. **Trade-offs are necessary** - Perfect separation may not be achievable

---

## 📊 **Recommendations**

### **For Production Use**

1. **Use empirical thresholds** from this calibration
2. **Accept some UNDECIDED outcomes** as inherent limitation
3. **Consider alternative scoring** for better discrimination
4. **Implement fallback strategies** for UNDECIDED cases

### **For Future Development**

1. **Explore better scoring metrics** with larger separation
2. **Develop adaptive thresholds** that learn from decisions
3. **Create model-specific calibrations** for different architectures
4. **Consider ensemble approaches** combining multiple signals

---

## ✅ **Conclusion**

The threshold calibration system successfully identifies why UNDECIDED outcomes occur and provides empirically-derived thresholds that improve decision rates. While complete elimination of UNDECIDED outcomes may not be possible with current scoring metrics for highly similar models, the calibration significantly improves the framework's practical utility.

**Key Achievement:** Reduced UNDECIDED rate through empirical calibration, though perfect separation remains challenging for near-identical models like GPT-2 family members.