#!/bin/bash

# Build script for compiling formal proofs to PDFs
# Supports LaTeX proofs with automatic compilation

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROOFS_DIR="$PROJECT_ROOT/proofs"
BUILD_DIR="$PROJECT_ROOT/build/proofs"
OUTPUT_DIR="$PROJECT_ROOT/docs/proofs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}Building Proof-of-Training formal proofs...${NC}"

# Function to check for required commands
check_requirements() {
    local missing_deps=()
    
    if ! command -v pdflatex &> /dev/null; then
        missing_deps+=("pdflatex")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${YELLOW}Warning: Missing dependencies: ${missing_deps[*]}${NC}"
        echo -e "${YELLOW}Install with:${NC}"
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  brew install --cask mactex  # for pdflatex"
        else
            echo "  sudo apt-get install texlive-full  # for pdflatex"
        fi
        
        echo -e "${YELLOW}Continuing with available tools...${NC}"
    fi
}

# Function to build LaTeX documents
build_latex() {
    local tex_file=$1
    local base_name=$(basename "$tex_file" .tex)
    
    echo -e "${BLUE}Building LaTeX: $base_name${NC}"
    
    if ! command -v pdflatex &> /dev/null; then
        echo -e "${YELLOW}Skipping $base_name.tex (pdflatex not found)${NC}"
        return 1
    fi
    
    cd "$BUILD_DIR"
    cp "$tex_file" .
    
    # Copy any additional LaTeX packages if needed
    if [ -f "$PROOFS_DIR/complexity.sty" ]; then
        cp "$PROOFS_DIR/complexity.sty" .
    fi
    
    # Run pdflatex twice for references
    echo -e "  Running pdflatex (first pass)..."
    pdflatex -interaction=nonstopmode "$base_name.tex" > "$base_name.log" 2>&1 || {
        echo -e "${RED}Error building $base_name.tex${NC}"
        echo -e "${RED}Check log file: $BUILD_DIR/$base_name.log${NC}"
        tail -20 "$base_name.log"
        return 1
    }
    
    echo -e "  Running pdflatex (second pass for references)..."
    pdflatex -interaction=nonstopmode "$base_name.tex" > /dev/null 2>&1
    
    # Run bibtex if needed
    if grep -q "\\bibliography" "$base_name.tex"; then
        echo -e "  Running bibtex..."
        bibtex "$base_name" > /dev/null 2>&1 || true
        pdflatex -interaction=nonstopmode "$base_name.tex" > /dev/null 2>&1
        pdflatex -interaction=nonstopmode "$base_name.tex" > /dev/null 2>&1
    fi
    
    # Move PDF to output directory
    mv "$base_name.pdf" "$OUTPUT_DIR/"
    echo -e "${GREEN}✓ Built $base_name.pdf${NC}"
    
    cd "$PROJECT_ROOT"
}

# Function to create a simple PDF alternative if LaTeX is not available
create_text_version() {
    local tex_file=$1
    local base_name=$(basename "$tex_file" .tex)
    
    echo -e "${BLUE}Creating text version: $base_name.txt${NC}"
    
    # Extract text content from LaTeX
    grep -v '^\s*\\' "$tex_file" | \
    grep -v '^\s*%' | \
    sed 's/\\[a-zA-Z]*{//g' | \
    sed 's/}//g' | \
    sed 's/\$//g' | \
    sed '/^$/N;/^\n$/d' > "$OUTPUT_DIR/$base_name.txt"
    
    echo -e "${GREEN}✓ Created $base_name.txt${NC}"
}

# Check requirements
check_requirements

# Build all LaTeX proofs
echo -e "\n${BLUE}=== Building LaTeX Proofs ===${NC}"
for tex_file in "$PROOFS_DIR"/*.tex; do
    if [ -f "$tex_file" ]; then
        build_latex "$tex_file" || create_text_version "$tex_file"
    fi
done

# Generate combined PDF with all proofs (if possible)
echo -e "\n${BLUE}=== Generating Combined PDF ===${NC}"
cd "$OUTPUT_DIR"

if command -v pdfunite &> /dev/null && [ -f "coverage_separation.pdf" ] && [ -f "wrapper_detection.pdf" ]; then
    pdfunite coverage_separation.pdf wrapper_detection.pdf pot_proofs_combined.pdf 2>/dev/null && {
        echo -e "${GREEN}✓ Created combined PDF: pot_proofs_combined.pdf${NC}"
    } || {
        echo -e "${YELLOW}Could not combine PDFs${NC}"
    }
elif command -v gs &> /dev/null && [ -f "coverage_separation.pdf" ] && [ -f "wrapper_detection.pdf" ]; then
    # Alternative using ghostscript
    gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=pot_proofs_combined.pdf \
       coverage_separation.pdf wrapper_detection.pdf 2>/dev/null && {
        echo -e "${GREEN}✓ Created combined PDF using ghostscript${NC}"
    } || {
        echo -e "${YELLOW}Could not combine PDFs${NC}"
    }
else
    echo -e "${YELLOW}PDF combination tools not found or PDFs not generated${NC}"
fi

# Generate index HTML
cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Proof-of-Training Formal Proofs</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 40px auto;
            max-width: 900px;
            line-height: 1.6;
            color: #333;
        }
        h1 { 
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 { 
            color: #34495e;
            margin-top: 30px;
        }
        .proof-list { 
            list-style-type: none; 
            padding: 0; 
        }
        .proof-list li { 
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            transition: background 0.3s;
        }
        .proof-list li:hover {
            background: #e9ecef;
        }
        .proof-list a { 
            text-decoration: none; 
            color: #3498db;
            font-weight: 500;
        }
        .proof-list a:hover { 
            text-decoration: underline; 
        }
        .description {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .status {
            float: right;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .status.available {
            background: #d4edda;
            color: #155724;
        }
        .status.text-only {
            background: #fff3cd;
            color: #856404;
        }
        .abstract {
            background: #f0f7ff;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>Proof-of-Training Formal Proofs</h1>
    
    <div class="abstract">
        <strong>Abstract:</strong> This collection provides rigorous mathematical foundations 
        for the Proof-of-Training (PoT) neural network verification framework. We establish 
        theoretical guarantees for model identity verification through challenge-response protocols,
        including coverage-separation theorems and wrapper detection analysis.
    </div>
    
    <h2>Core Theoretical Results</h2>
    <ul class="proof-list">
        <li>
            <span class="status available">PDF</span>
            <a href="coverage_separation.pdf">Coverage-Separation Theorem</a>
            <div class="description">
                Proves that challenge vectors can both cover network behavior space 
                and separate distinct neural networks with high probability.
                Key result: O((diam(Θ)/ε)^p) challenges suffice for ε-verification.
            </div>
        </li>
        <li>
            <span class="status available">PDF</span>
            <a href="wrapper_detection.pdf">Wrapper Detection Analysis</a>
            <div class="description">
                Analyzes security against wrapper attacks where adversaries attempt 
                to substitute models using input/output transformations.
                Shows detection probability ≥ 1 - exp(-n·d²/2C) for n-dimensional challenges.
            </div>
        </li>
    </ul>
    
    <h2>Text Versions</h2>
    <ul class="proof-list">
        <li>
            <span class="status text-only">TXT</span>
            <a href="coverage_separation.txt">Coverage-Separation (Plain Text)</a>
            <div class="description">Simplified text version for quick reference</div>
        </li>
        <li>
            <span class="status text-only">TXT</span>
            <a href="wrapper_detection.txt">Wrapper Detection (Plain Text)</a>
            <div class="description">Simplified text version for quick reference</div>
        </li>
    </ul>
    
    <h2>Combined Document</h2>
    <ul class="proof-list">
        <li>
            <span class="status available">PDF</span>
            <a href="pot_proofs_combined.pdf">All Proofs (Combined PDF)</a>
            <div class="description">
                Complete collection of all formal proofs in a single document
            </div>
        </li>
    </ul>
    
    <h2>Key Theoretical Contributions</h2>
    <ol>
        <li><strong>Coverage Guarantee:</strong> Finite challenge sets can ε-approximate infinite network behaviors</li>
        <li><strong>Separation Property:</strong> Distinct networks are distinguishable with probability ≥ 1 - 2^(-k)</li>
        <li><strong>Wrapper Hardness:</strong> Perfect model wrapping is NP-hard</li>
        <li><strong>Adaptive Detection:</strong> O(k log n) queries suffice to detect k-complexity wrappers</li>
    </ol>
    
    <h2>Implementation Notes</h2>
    <p>
        These theoretical results are implemented in the PoT framework at 
        <a href="https://github.com/rohanvinaik/PoT_Experiments">github.com/rohanvinaik/PoT_Experiments</a>.
        The proofs establish security guarantees with concrete parameters suitable for 
        practical deployment.
    </p>
    
    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em;">
        Generated: <script>document.write(new Date().toLocaleString());</script>
    </footer>
</body>
</html>
EOF

echo -e "${GREEN}✓ Created index.html${NC}"

# Create README for the proofs directory
cat > README.md << 'EOF'
# Formal Proofs for Proof-of-Training

This directory contains rigorous mathematical proofs establishing the theoretical foundations of the Proof-of-Training (PoT) framework.

## Contents

### LaTeX Documents
- `coverage_separation.tex` - Coverage-Separation theorem proving challenge effectiveness
- `wrapper_detection.tex` - Security analysis against wrapper attacks

### Generated Output (in docs/proofs/)
- PDF versions of all proofs
- Combined PDF document
- HTML index for web viewing

## Building the Proofs

To compile the LaTeX documents to PDFs:

```bash
./scripts/build_proofs.sh
```

### Requirements
- LaTeX distribution (e.g., TeX Live, MiKTeX, MacTeX)
- pdflatex command
- Optional: pdfunite or ghostscript for combining PDFs

### Installation

#### macOS
```bash
brew install --cask mactex
```

#### Ubuntu/Debian
```bash
sudo apt-get install texlive-full
```

#### Windows
Download and install MiKTeX from https://miktex.org/

## Key Results

### Coverage-Separation Theorem
- Challenge sets of size O((diam(Θ)/ε)^p) provide ε-coverage
- Distinct networks separated with probability ≥ 1 - 2^(-256)
- Enables both completeness and soundness of verification

### Wrapper Detection
- Detection probability ≥ 1 - exp(-n·d²/2C) for n-dimensional challenges
- Perfect wrapping is NP-hard
- Multiple detection methods: statistical, timing, adaptive

## Citation

If you use these theoretical results, please cite:
```
@article{pot2024,
  title={Proof-of-Training: Formal Verification for Neural Networks},
  author={...},
  year={2024}
}
```
EOF

echo -e "${GREEN}✓ Created README.md${NC}"

# Summary
echo -e "\n${BLUE}=== Build Summary ===${NC}"
echo "Output directory: $OUTPUT_DIR"

if [ -d "$OUTPUT_DIR" ]; then
    echo -e "\nGenerated files:"
    ls -la "$OUTPUT_DIR" 2>/dev/null | grep -E "\.(pdf|txt|html)" | while read line; do
        echo "  $line"
    done
fi

echo -e "\n${GREEN}Build complete!${NC}"
echo "View proofs at: $OUTPUT_DIR/index.html"

# Check if we can open the index in a browser
if command -v open &> /dev/null; then
    echo -e "\n${BLUE}Opening index.html in browser...${NC}"
    open "$OUTPUT_DIR/index.html" 2>/dev/null || true
elif command -v xdg-open &> /dev/null; then
    echo -e "\n${BLUE}Opening index.html in browser...${NC}"
    xdg-open "$OUTPUT_DIR/index.html" 2>/dev/null || true
fi