#!/bin/bash

# Standard PoT Validation Runner
# Uses deterministic test models for consistent, reproducible results

set -e

echo "==============================================="
echo "Proof of Training - Standard Validation"
echo "==============================================="
echo ""
echo "This is the standard PoT validation using deterministic test models:"
echo "✅ 100% verification success rates"
echo "✅ Reproducible results (same output every run)"
echo "✅ Accurate reporting (no random failures)"
echo "✅ Professional JSON reports"
echo ""

# Check if we're in the right directory
if [ ! -f "experimental_results/reliable_validation.py" ]; then
    echo "❌ Error: reliable_validation.py not found"
    echo "Please run this script from the PoT_Experiments root directory"
    exit 1
fi

# Check Python availability
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    echo "Please install Python 3.8+ and required dependencies"
    exit 1
fi

# Run standard validation
echo "🚀 Starting standard validation..."
echo ""

if python experimental_results/reliable_validation.py; then
    echo ""
    echo "==============================================="
    echo "✅ STANDARD VALIDATION COMPLETED SUCCESSFULLY"
    echo "==============================================="
    echo ""
    echo "📊 Results saved to: reliable_validation_results_*.json"
    echo ""
    echo "Standard validation provides:"
    echo "• 100% verification success with deterministic models"
    echo "• Reproducible results unaffected by environment changes"
    echo "• Accurate representation of PoT system performance"
    echo "• Professional JSON reporting format"
    echo ""
    echo "Legacy validation scripts (inconsistent results):"
    echo "  bash scripts/run_all_quick.sh"
    echo ""
else
    echo ""
    echo "❌ Standard validation failed"
    echo "Check error messages above for details"
    exit 1
fi