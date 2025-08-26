# NeurIPS 2025 Workshop Submission Packet

## Proof-of-Training (PoT) Verifier: Cryptographically Pre-Committed, Anytime Behavioral Model Identity Checks

**Submission to**: NeurIPS 2025 Workshop on Reliable ML from Unreliable Data (non-archival)

## Contents

### Main Paper
- `pot_neurips2025.pdf` - **Main submission PDF** (9 pages, 244KB)
- `pot_neurips2025.tex` - LaTeX source file
- `pot_neurips2025.bbl` - Bibliography (compiled)
- `references.bib` - Bibliography source

### Figures
- `figures/fig1_time_to_decision.pdf` - Time-to-decision trajectories 
- `figures/fig2_error_rates.pdf` - FAR/FRR curves with operating points
- `figures/confusion_matrix.pdf` - Decision accuracy visualization

### Supplementary
- `POT_PAPER_COMPLETE_UPDATED.md` - Extended markdown version with additional details

## Key Results

- **Query Efficiency**: 95.2-98.6% reduction vs fixed-N=1000 baseline
- **Wall-time**: 1-2 minutes for GPT-2 class and API models  
- **Hardware**: All experiments on Apple M1 Max (64GB)
- **Sharding**: Demonstrated 34B models on consumer hardware (extended experiments)

## Compilation

To recompile the PDF:
```bash
pdflatex pot_neurips2025.tex
bibtex pot_neurips2025
pdflatex pot_neurips2025.tex
pdflatex pot_neurips2025.tex
```

## Repository

Full implementation available at: https://github.com/[anonymized]

## Contact

[Anonymized for review]