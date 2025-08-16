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
