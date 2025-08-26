#!/bin/bash
# Build the paper PDF with proper bibliography using pandoc

set -e

echo "Building PoT paper with bibliography..."

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed"
    echo "Install with: brew install pandoc pandoc-citeproc"
    exit 1
fi

# Input and output files
PAPER_MD="POT_PAPER_COMPLETE_UPDATED.md"
BIBLIOGRAPHY="references.bib"
OUTPUT_PDF="POT_PAPER_NeurIPS2025.pdf"
OUTPUT_TEX="POT_PAPER_NeurIPS2025.tex"

# Check if input files exist
if [ ! -f "$PAPER_MD" ]; then
    echo "Error: Paper file '$PAPER_MD' not found"
    exit 1
fi

if [ ! -f "$BIBLIOGRAPHY" ]; then
    echo "Error: Bibliography file '$BIBLIOGRAPHY' not found"
    exit 1
fi

# Create a temporary YAML header for pandoc
cat > paper_header.yaml << EOF
---
title: "Proof-of-Training (PoT) Verifier: Cryptographically Pre-Committed, Anytime Behavioral Model Identity Checks"
author: "Anonymous Submission"
date: "NeurIPS 2025 Workshop on Reliable ML from Unreliable Data"
abstract: |
  We present a post-training behavioral verifier for model identity. Given two models (or a model and a reference), we decide SAME / DIFFERENT / UNDECIDED with controlled error using dozens of queries rather than thousands. The verifier (i) pre-commits to a challenge set via HMAC-derived seeds, (ii) maintains an anytime confidence sequence using Empirical-Bernstein (EB) bounds, and (iii) stops early when the interval is decisively within a SAME/DIFFERENT region. Each run exports a reproducible audit bundle (transcripts, seeds/commitments, configs, environment). On the systems side, we support sharded verification to validate 34B-class models (aggregate ≈206 GB weights) on a 64 GB host with peak ≈52% RAM by loading/releasing shards. The repository includes single-command runners for local and API (black-box) verification. For remote identity binding, we clarify when TEE attestation or vendor commitments are required and how ZK can attest correctness of the verifier computation from a published transcript.
bibliography: references.bib
csl: ieee.csl
link-citations: true
colorlinks: true
linkcolor: blue
citecolor: blue
urlcolor: blue
geometry: margin=1in
fontsize: 10pt
documentclass: article
header-includes:
  - \usepackage{times}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage[ruled,vlined]{algorithm2e}
---
EOF

# Remove the title and abstract from the markdown (they're in the YAML header now)
# Skip the first few lines that contain the title
tail -n +10 "$PAPER_MD" > paper_body.md

# Combine header and body
cat paper_header.yaml paper_body.md > paper_complete.md

# Build PDF with bibliography
echo "Generating PDF..."
pandoc paper_complete.md \
    --pdf-engine=xelatex \
    --citeproc \
    --bibliography="$BIBLIOGRAPHY" \
    --csl=ieee.csl \
    -o "$OUTPUT_PDF" \
    2>/dev/null || {
    echo "Note: PDF generation requires LaTeX. Trying HTML instead..."
    pandoc paper_complete.md \
        --standalone \
        --citeproc \
        --bibliography="$BIBLIOGRAPHY" \
        --mathjax \
        -o "POT_PAPER_NeurIPS2025.html"
    echo "✅ Generated HTML version: POT_PAPER_NeurIPS2025.html"
}

# Also generate LaTeX for submission if needed
echo "Generating LaTeX..."
pandoc paper_complete.md \
    --citeproc \
    --bibliography="$BIBLIOGRAPHY" \
    --natbib \
    -o "$OUTPUT_TEX" \
    2>/dev/null || echo "Note: LaTeX generation skipped"

# Clean up temporary files
rm -f paper_header.yaml paper_body.md paper_complete.md

# Check results
if [ -f "$OUTPUT_PDF" ]; then
    echo "✅ Successfully generated: $OUTPUT_PDF"
fi

if [ -f "$OUTPUT_TEX" ]; then
    echo "✅ Successfully generated: $OUTPUT_TEX"
fi

echo ""
echo "To view references in the paper:"
echo "  - PDF: All citations will be properly formatted with bibliography"
echo "  - LaTeX: Can be further edited for conference submission"
echo ""
echo "To manually compile with references:"
echo "  pandoc $PAPER_MD --citeproc --bibliography=$BIBLIOGRAPHY -o output.pdf"