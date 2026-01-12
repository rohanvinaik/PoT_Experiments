# PoT NeurIPS 2025 Submission Package

## Contents

This package contains the validated experimental submission for the PoT (Proof-of-Training) Verifier paper.

### Files

- `pot_neurips2025.pdf` - Compiled paper (13 pages)
- `pot_neurips2025.tex` - LaTeX source with **corrected experimental data**
- `references.bib` - Bibliography with 19 validated citations
- `figures/` - All figures used in the paper
  - `fig1_time_to_decision.pdf` - Time-to-decision trajectories
  - `fig2_error_rates.pdf` - FAR/FRR vs decision threshold

### Key Corrections Made (Oct 2, 2025)

This version corrects significant discrepancies between the original paper and actual experimental validation:

#### Table 3 Corrections
- **Removed fabricated effect sizes** that didn't exist in experimental data
- **Fixed all timing values** to match experimental JSONs exactly:
  - gpt2→gpt2: 71.7s (was 65.2s)
  - gpt2→distilgpt2: 92.2s (was 61.4s)
  - dialogpt→gpt2: 17.3s (was 42.1s)
  - Llama-7B: 1346.7s = 22.4 min (was incorrectly labeled)
- **Fixed query counts**: gpt2→gpt2-medium used 40 queries (not 64)
- **Added memory usage column** with actual experimental data (1.27-8.01 GB)
- **Added both Llama-7B variants** (base and chat self-tests)

#### Abstract & Claims
- Changed "14-48 queries" → **"14-40 queries"** (actual experimental range)
- Changed "1-2 minutes" → **"17-92 seconds for small models, 22 minutes for 7B"**
- Changed "30×-300× speedup" → **"30×-60× speedup"** (realistic calculation)

#### Throughout Paper
- Fixed speedup claims from inflated 300× to realistic 60-200× maximum
- Removed unsupported "effect size" claims (no |X̄_n| values in experimental data)
- Updated query reduction from "96.8%" to "96%+" (14-40 vs 1000 baseline)

### Experimental Evidence

All claims are now backed by 8 actual experimental runs from Aug 20-23, 2025:
- `gpt2_self_audit/` - 30 queries, 71.7s, 1.56GB RAM
- `pythia70m_self_audit/` - 30 queries, 66.9s, 1.27GB RAM
- `gpt2_distilgpt2_audit/` - 32 queries, 92.2s, 1.33GB RAM
- `gpt2_to_gpt2medium_fresh/` - 40 queries, 84.6s, 1.71GB RAM
- `dialogpt_gpt2_quick/` - 16 queries, 17.3s, 1.85GB RAM
- `neo_pythia_cross_audit/` - 32 queries, 133.3s, 2.36GB RAM
- `llama_self_test_base/` - 14 queries, 1346.7s, 8.01GB RAM
- `llama_self_test_chat/` - 14 queries, 1381.4s, 7.95GB RAM

### Citation Validation

All 19 citations verified present in `references.bib`:
- Sequential testing: Wald (1945), Maurer & Pontil (2009), Howard et al. (2021)
- Cryptography: HMAC (RFC 2104), SHA-256 (FIPS 180-4), SGX (Costan 2016)
- Model verification: Jia et al. (2021), Uchida et al. (2017)
- Baselines: Hendrycks et al. (2021), Johari et al. (2017), Ramdas et al. (2023)

### Compilation

```bash
pdflatex pot_neurips2025.tex
bibtex pot_neurips2025
pdflatex pot_neurips2025.tex
pdflatex pot_neurips2025.tex
```

Produces 13-page PDF with proper citations and no warnings.

### Usage

For website deployment, use `pot_neurips2025.pdf` directly.

For arXiv submission, include:
- `pot_neurips2025.tex`
- `references.bib`
- `figures/` directory

---

**Validation Status**: ✅ All experimental claims verified against actual results
**Last Updated**: October 2, 2025
**Compiled with**: pdfLaTeX (TeX Live 2025)
