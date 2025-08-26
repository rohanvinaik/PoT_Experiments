# NeurIPS 2025 Workshop Submission Guide

## Paper: Proof-of-Training (PoT) Verifier
**Target Workshop**: Reliable ML from Unreliable Data (non-archival)

## ✅ Format Compliance Summary

The paper **POT_PAPER_COMPLETE_UPDATED.md** is fully aligned with NeurIPS 2025 workshop requirements:

### Required Elements Present:
1. **Page Limit**: ~8-9 pages of content (within 9-page limit)
2. **Abstract**: Concise summary with key contributions
3. **Sections**: Complete structure (Intro, Related Work, Method, Experiments, Results, Limitations, Broader Impact, Conclusion)
4. **References**: Properly formatted citations (20+ references)
5. **Reproducibility**: Evidence bundles with cryptographic hashes
6. **Paper Checklist**: Complete NeurIPS checklist in Appendix B
7. **Broader Impact**: Section 9 addresses societal impacts and ethics
8. **Anonymous**: No author identifying information

### Key Strengths for Workshop:
- **Novel contribution**: First anytime behavioral verifier with pre-commitment
- **Strong evaluation**: 8 model pairs tested with reproducible evidence
- **Systems contribution**: Sharded verification for 206GB models on 64GB RAM
- **Clear limitations**: Section 8 explicitly discusses limitations
- **Practical impact**: 96.8% query reduction, 30×-300× speedup

## 📋 Conversion to LaTeX

To generate the final PDF submission:

```bash
# 1. Download NeurIPS style files
wget https://media.neurips.cc/Conferences/NeurIPS2025/Styles/neurips_2025.sty
wget https://media.neurips.cc/Conferences/NeurIPS2025/Styles/neurips_2025.tex

# 2. Convert Markdown to LaTeX (use pandoc or manual conversion)
pandoc POT_PAPER_COMPLETE_UPDATED.md -o pot_neurips.tex \
    --template=neurips_template.tex \
    --bibliography=references.bib

# 3. Add to preamble:
\documentclass{article}
\usepackage[preprint]{neurips_2025}  # Change to [final] for camera-ready
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}

# 4. Compile
pdflatex pot_neurips.tex
bibtex pot_neurips
pdflatex pot_neurips.tex
pdflatex pot_neurips.tex
```

## 📊 Figures to Generate

The paper references 3 figures that need to be generated:

1. **Figure 1**: Time-to-decision trajectories
   - File: `figures/fig1_time_to_decision.png`
   - Shows convergence for SAME vs DIFFERENT decisions

2. **Figure 2**: Error rates vs decision threshold  
   - File: `figures/fig2_error_rates.png`
   - Shows FAR/FRR tradeoff curves

3. **Figure 3**: Confusion matrix
   - File: `figures/confusion_matrix.png`
   - Shows 8/8 correct decisions

Generate with: `python scripts/generate_paper_figures.py`

## 📝 Submission Checklist

Before submitting:

- [ ] Convert to LaTeX using neurips_2025.sty
- [ ] Generate all 3 figures
- [ ] Create references.bib file
- [ ] Ensure anonymous (no author info)
- [ ] Verify 9-page limit (excluding refs/checklist)
- [ ] Complete paper checklist
- [ ] Include GitHub link for code
- [ ] Single PDF with paper + checklist
- [ ] Abstract submission by May 11, 2025
- [ ] Full paper by May 15, 2025

## 🎯 Workshop Alignment

The paper strongly aligns with "Reliable ML from Unreliable Data":

1. **Unreliable aspects addressed**:
   - Opaque API-served models
   - Model drift and substitution
   - Adversarial prompt selection
   - Temperature/wrapper attacks

2. **Reliability guarantees provided**:
   - Statistical error control (α, β)
   - Cryptographic pre-commitment
   - Anytime confidence sequences
   - Reproducible evidence bundles

3. **Workshop themes**:
   - Verification and trustworthiness
   - Black-box auditing
   - Production deployment challenges
   - Sample efficiency (96.8% reduction)

## 📧 Contact for Questions

For technical questions about the implementation:
- See GitHub issues: [anonymous repository]
- Evidence bundles available for all claims

## 🚀 Quick Demo

Reviewers can run a 5-minute demo:

```bash
git clone https://github.com/ANONYMOUS/PoT_Experiments.git
cd PoT_Experiments
pip install -r requirements-pinned.txt
python scripts/run_e2e_validation.py \
    --ref-model gpt2 \
    --cand-model distilgpt2 \
    --mode quick \
    --max-queries 20
```

Expected: DIFFERENT decision in ~16 queries, 2-3 minutes