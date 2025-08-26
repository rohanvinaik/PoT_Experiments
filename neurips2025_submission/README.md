# NeurIPS 2025 Workshop Submission

## Proof-of-Training (PoT) Verifier: Cryptographically Pre-Committed, Anytime Behavioral Model Identity Checks

**Target**: NeurIPS 2025 Workshop on Reliable ML from Unreliable Data (non-archival)

## Submission Files

### Main Paper
- **`pot_neurips2025.pdf`** - Main submission (9 pages, 228KB)
- `pot_neurips2025.tex` - LaTeX source with professional typography
- `pot_neurips2025.bbl` - Compiled bibliography
- `references.bib` - Bibliography source

### Figures
- `figures/fig1_time_to_decision.pdf` - Time-to-decision trajectories 
- `figures/fig2_error_rates.pdf` - FAR/FRR curves
- Additional figures in `figures/` directory

### Supplementary
- `POT_PAPER_COMPLETE_UPDATED.md` - Extended markdown version

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