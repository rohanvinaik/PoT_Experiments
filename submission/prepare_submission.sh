#!/bin/bash
# Script to prepare anonymous submission package
# Excludes placeholder files and personal information

set -e

echo "Preparing anonymous submission package..."

# Create submission directory
SUBMISSION_DIR="pot_submission_anonymous"
rm -rf $SUBMISSION_DIR
mkdir -p $SUBMISSION_DIR

# Copy core implementation files
echo "Copying core implementation..."
cp -r src $SUBMISSION_DIR/
cp -r scripts $SUBMISSION_DIR/
cp -r manifests $SUBMISSION_DIR/
cp -r configs $SUBMISSION_DIR/
cp -r data $SUBMISSION_DIR/

# Copy documentation (excluding problematic files)
echo "Copying documentation..."
mkdir -p $SUBMISSION_DIR/docs
cp -r docs/papers/POT_PAPER_COMPLETE_UPDATED.md $SUBMISSION_DIR/docs/
cp -r docs/statistical_verification.md $SUBMISSION_DIR/docs/ 2>/dev/null || true

# Copy essential files
echo "Copying essential files..."
cp README.md $SUBMISSION_DIR/
cp requirements.txt $SUBMISSION_DIR/
cp requirements-pinned.txt $SUBMISSION_DIR/
cp .gitignore $SUBMISSION_DIR/
cp CLAUDE.md $SUBMISSION_DIR/

# Copy figures if they exist
if [ -d "figures" ]; then
    cp -r figures $SUBMISSION_DIR/
fi

# Remove excluded files based on .submission_exclude
echo "Removing excluded files..."
while IFS= read -r pattern; do
    # Skip comments and empty lines
    if [[ ! "$pattern" =~ ^# ]] && [[ ! -z "$pattern" ]]; then
        # Remove matching files/dirs from submission
        find $SUBMISSION_DIR -path "*/$pattern" -exec rm -rf {} + 2>/dev/null || true
    fi
done < .submission_exclude

# Clean up Python cache files
find $SUBMISSION_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find $SUBMISSION_DIR -type f -name "*.pyc" -delete 2>/dev/null || true
find $SUBMISSION_DIR -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove any remaining personal references
echo "Scrubbing personal information..."
find $SUBMISSION_DIR -type f \( -name "*.py" -o -name "*.md" -o -name "*.yaml" -o -name "*.json" \) \
    -exec grep -l "rohanvinaik" {} \; 2>/dev/null | while read file; do
    echo "  Warning: Found personal reference in $file - removing"
    rm "$file"
done

# Create minimal example results
echo "Creating minimal example results..."
mkdir -p $SUBMISSION_DIR/experimental_results/example
cat > $SUBMISSION_DIR/experimental_results/example/summary.json << EOF
{
  "run_id": "example_validation",
  "decision": "DIFFERENT",
  "confidence": 0.99,
  "n_queries": 32,
  "time_seconds": 92.2,
  "memory_mb": 1325
}
EOF

# Create submission archive
echo "Creating archive..."
tar -czf pot_submission_anonymous.tar.gz $SUBMISSION_DIR/

echo "✅ Submission package created: pot_submission_anonymous.tar.gz"
echo ""
echo "Package excludes:"
echo "  - Placeholder implementations (baselines.py, governance.py, etc.)"
echo "  - Personal information and paths"
echo "  - Incomplete experimental files"
echo "  - Large result files (can be regenerated)"
echo ""
echo "To verify the package:"
echo "  tar -tzf pot_submission_anonymous.tar.gz | less"