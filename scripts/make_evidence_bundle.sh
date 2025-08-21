#!/bin/bash
# Generate evidence bundle for verification runs

set -e

# Parse arguments
RUN_ID=""
INCLUDE_FILES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --include)
            INCLUDE_FILES+=("$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default run ID if not provided
if [ -z "$RUN_ID" ]; then
    RUN_ID=$(date +"%Y-%m-%dT%H-%M-%SZ")
fi

BUNDLE_DIR="evidence_bundle_${RUN_ID}"

echo "Creating evidence bundle: ${BUNDLE_DIR}"

# Create bundle directory
mkdir -p "${BUNDLE_DIR}"

# Copy verification results
echo "Copying verification results..."
for pattern in "${INCLUDE_FILES[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            cp "$file" "${BUNDLE_DIR}/"
            echo "  Added: $file"
        fi
    done
done

# Add environment information
echo "Recording environment..."
cat > "${BUNDLE_DIR}/environment.txt" <<EOF
Evidence Bundle Generated: $(date)
Run ID: ${RUN_ID}

System Information:
$(uname -a)

Python Version:
$(python --version)

Python Packages:
$(pip freeze)

Git Commit:
$(git rev-parse HEAD 2>/dev/null || echo "Not in git repository")

Hardware:
$(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo | grep "model name" | head -1)
Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 " GB"}' || free -h | grep Mem)
EOF

# Add verification script checksums
echo "Computing checksums..."
find scripts -name "*.py" -o -name "*.sh" | while read script; do
    shasum -a 256 "$script" >> "${BUNDLE_DIR}/checksums.txt"
done

# Add README with instructions
cat > "${BUNDLE_DIR}/README.md" <<EOF
# Verification Evidence Bundle

Run ID: ${RUN_ID}
Generated: $(date)

## Contents

- Verification results (*.json)
- Environment information (environment.txt)
- Script checksums (checksums.txt)

## To Verify

1. Check environment matches requirements
2. Verify script checksums haven't changed
3. Re-run verification with same parameters
4. Compare results with included JSON files

## Reproducibility

\`\`\`bash
# Re-run the exact verification
python scripts/test_yi34b_sharded.py --max-memory 30
\`\`\`

EOF

# Create tarball
tar -czf "${BUNDLE_DIR}.tar.gz" "${BUNDLE_DIR}"

echo "✅ Evidence bundle created: ${BUNDLE_DIR}.tar.gz"
echo "   Size: $(du -h ${BUNDLE_DIR}.tar.gz | cut -f1)"
echo "   Files: $(tar -tzf ${BUNDLE_DIR}.tar.gz | wc -l)"