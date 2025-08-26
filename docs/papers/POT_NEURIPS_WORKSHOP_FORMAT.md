# NeurIPS 2025 Workshop Paper Format Requirements Checklist

## Current Paper: "Proof-of-Training (PoT) Verifier"
**Target Workshop**: Reliable ML from Unreliable Data (non-archival)

## Format Compliance Checklist

### ✅ Page Limit
- **Requirement**: 9 content pages max (excluding references and checklist)
- **Current**: ~8-9 pages of content in LaTeX (521 lines in Markdown ≈ 8-9 pages)
- **Status**: COMPLIANT

### ✅ Structure
- **Title**: Clear and descriptive ✓
- **Anonymous Submission**: Marked as "Anonymous Submission" ✓
- **Abstract**: Present and concise (150-200 words) ✓
- **Sections**: Standard structure (Intro, Related Work, Method, Experiments, Results, Conclusion) ✓

### ✅ Content Requirements
1. **Problem Statement**: Clearly defined in Introduction ✓
2. **Contributions**: Explicitly listed (4 contributions) ✓
3. **Related Work**: Comprehensive with proper citations ✓
4. **Method**: Detailed with mathematical formulation ✓
5. **Experiments**: Extensive with reproducible evidence bundles ✓
6. **Results**: Multiple tables, figures referenced ✓
7. **Limitations**: Section 8 addresses limitations ✓

### ⚠️ Formatting for LaTeX Conversion

The paper needs these adjustments for NeurIPS LaTeX template:

1. **Citations**: Convert inline citations from [@author2009] to \cite{author2009}
2. **Math**: Ensure all math is in proper LaTeX format (currently using \(...\))
3. **Tables**: Convert Markdown tables to LaTeX format
4. **Figures**: Add proper figure captions and labels
5. **References**: Convert to BibTeX format

### 📝 Paper Checklist Items to Address

1. **Code Availability**: ✓ (GitHub repository mentioned)
2. **Reproducibility**: ✓ (Evidence bundles with hashes provided)
3. **Compute Requirements**: ✓ (Detailed in paper - M1 Max, 64GB RAM)
4. **Limitations**: ✓ (Section 8)
5. **Societal Impact**: Needs brief discussion
6. **LLM Usage**: Not applicable (no LLM in core methodology)

### 📋 Sections Summary

1. **Abstract** (1 paragraph)
2. **Introduction** (1 page)
3. **Related Work** (0.75 pages)
4. **Preliminaries and Threat Model** (0.5 pages)
5. **Method** (1.5 pages)
   - 4.1 Pre-committed challenges
   - 4.2 Scoring
   - 4.3 Anytime EB confidence sequence
   - 4.4 Decision rules
   - 4.5 API verification
6. **Implementation** (0.5 pages)
   - 5.1 Runner and artifacts
   - 5.2 Sharded verification
7. **Experimental Setup** (0.5 pages)
8. **Results** (3 pages)
   - 7.1 Query Efficiency
   - 7.2 Wall-Time Performance
   - 7.3 Operational Impact
   - 7.4 Large-model Feasibility
   - 7.5 Robustness
   - 7.6 Comparison to Prior
   - 7.7 Bootstrap power
9. **Limitations** (0.5 pages)
   - 8.1 Behavioral Fingerprinting
10. **Conclusion** (0.25 pages)
11. **References** (not counted)

### 🔧 Required LaTeX Template Elements

```latex
\documentclass{article}
\usepackage[preprint]{neurips_2025}
% For final submission, replace with:
% \usepackage[final]{neurips_2025}

\title{Proof-of-Training (PoT) Verifier: Cryptographically Pre-Committed, Anytime Behavioral Model Identity Checks}

% Anonymous submission - no author block

\begin{abstract}
% Abstract here
\end{abstract}

\begin{document}
\maketitle

\section{Introduction}
% Content...

\section{Related Work}
% Content...

% ... rest of sections

\bibliography{references}
\bibliographystyle{neurips_2025}

\appendix
% Optional appendices

% Paper checklist
\section*{NeurIPS Paper Checklist}
% Checklist items...

\end{document}
```

### 📊 Key Tables/Figures to Include

1. **Table 1**: SAME/DIFFERENT decisions with evidence bundles ✓
2. **Table 2**: Wall-time performance comparison ✓
3. **Table 3**: Comparison to prior methods ✓
4. **Figure 1**: Time-to-decision trajectories (referenced)
5. **Figure 2**: FAR/FRR tradeoff curves (referenced)
6. **Figure 3**: Confusion matrix (referenced)

### ⚡ Action Items for Final Submission

1. **Convert to LaTeX**: Use neurips_2025.sty template
2. **Add BibTeX**: Create .bib file with all references
3. **Generate Figures**: Include the 3 referenced figures
4. **Complete Checklist**: Fill out NeurIPS paper checklist
5. **Add Appendices**: Include additional proofs/derivations if needed
6. **Anonymize**: Ensure no author-identifying information
7. **PDF Generation**: Single PDF with paper + appendices + checklist

### 🎯 Workshop-Specific Considerations

Since this is for "Reliable ML from Unreliable Data" workshop:
- Emphasize the unreliability aspects (API opacity, model drift)
- Highlight the reliability guarantees (statistical bounds, evidence bundles)
- Connect to workshop themes of verification and trustworthiness

The paper is well-aligned with NeurIPS workshop requirements and ready for LaTeX conversion!