#!/usr/bin/env bash
# Auto-update validation results history and README metrics
# This script should be run after validation to maintain current metrics

set -euo pipefail

echo "🔄 Auto-updating validation results..."

# Update results history
echo "📊 Collecting new validation results..."
python3 scripts/update_results_history.py

# Update README with new metrics
echo "📝 Updating README with rolling averages..."
python3 scripts/update_readme_metrics.py

# Check if there are changes to commit
if git diff --quiet; then
    echo "✅ No new results to update"
else
    echo "📋 New validation results detected"
    
    # Show what changed
    echo "Changes detected:"
    git diff --name-only
    
    echo ""
    echo "🎯 Updated metrics available in:"
    echo "  - README.md (live metrics section)"
    echo "  - VALIDATION_RESULTS_SUMMARY.md"
    echo "  - validation_results_history.json"
    
    echo ""
    echo "💡 To commit these updates:"
    echo "  git add ."
    echo "  git commit -m 'Update validation results with latest metrics'"
    echo "  git push"
fi

echo "✅ Validation results update complete"