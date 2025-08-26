# NeurIPS 2025 Workshop Submission Checklist

## ✅ Completed Requirements

### 1. Anonymization
- [x] All author names removed
- [x] All institutional affiliations removed  
- [x] GitHub URLs changed to ANONYMOUS
- [x] Personal paths changed to ~/
- [x] Acknowledgments section placeholder added

### 2. LLM Usage Disclosure (NeurIPS Requirement)
- [x] **Section Added**: "LLM Usage Statement" in Acknowledgments
- [x] Disclosed code development assistance (Claude 3)
- [x] Disclosed paper writing assistance
- [x] Disclosed data analysis assistance
- [x] Stated all results are from actual execution, not LLM generation
- [x] Noted manual review and validation of LLM contributions

### 3. Code Availability
- [x] Anonymous GitHub repository link provided
- [x] Statement about post-review release under MIT license
- [x] Pinned dependency versions (requirements-pinned.txt)
- [x] Minimal reproducibility command provided (5-10 min)
- [x] Docker container mentioned for post-review

### 4. Reproducibility Statement
- [x] Environment specifications (Python 3.8+, hardware tested)
- [x] Key parameters explicitly listed (α, γ, δ*, n_min, n_max)
- [x] Cryptographic parameters specified (HMAC-SHA256, 256-bit keys)
- [x] Evidence bundle checksums in tables
- [x] Deterministic seed generation documented

### 5. Ethics Statement
- [x] Broader impacts section included
- [x] Benefits clearly stated
- [x] Risks and limitations acknowledged
- [x] Scope boundaries explicitly defined

### 6. Technical Content
- [x] Abstract within word limit
- [x] Real experimental results (not simulated)
- [x] Comparison to prior work (Section 7.5)
- [x] Statistical guarantees formally stated
- [x] Figures generated and referenced (Figures 1-2)
- [x] All citations have bibliography entries (16 citations verified)

### 7. Paper Structure
- [x] Title and abstract
- [x] Introduction with contributions
- [x] Related work with comparisons
- [x] Method section with formal definitions
- [x] Implementation details
- [x] Experimental setup
- [x] Results with real data
- [x] Limitations and negative results
- [x] Future directions
- [x] Broader impacts
- [x] References
- [x] Appendices

## ⚠️ Pre-Submission Checklist

Before submitting, verify:

1. **Format Requirements**:
   - [ ] Convert to NeurIPS LaTeX template if required
   - [ ] Check page limit (typically 4-9 pages for workshops)
   - [ ] Ensure figures are high quality (300 DPI)

2. **Submission System**:
   - [ ] Create CMT/OpenReview account
   - [ ] Prepare supplementary materials zip
   - [ ] Check submission deadline and timezone

3. **Final Review**:
   - [ ] Spell check and grammar check
   - [ ] Verify all links work (anonymous GitHub)
   - [ ] Test minimal reproducibility command
   - [ ] Ensure no author information leaked

## 📝 Submission Components

1. **Main Paper**: `POT_PAPER_COMPLETE_UPDATED.md` (or PDF via build_paper.sh)
2. **Bibliography**: `references.bib` (all 16 citations verified)
3. **Supplementary Code**: Anonymous GitHub repository
4. **Figures**: In `figures/` directory
5. **Evidence**: Example bundles in repository

## 🚀 Build Commands

```bash
# Generate PDF with bibliography
cd docs/papers
./build_paper.sh

# Verify citations
python verify_citations.py

# Create anonymous submission package
cd ../..
./prepare_submission.sh
```

## ⚠️ Important Notes

1. **LLM Disclosure is MANDATORY**: NeurIPS requires disclosure of any LLM use
2. **Code must be available**: At least anonymously during review
3. **Reproducibility is key**: Reviewers will check the 5-10 min test
4. **Workshop vs Main Track**: This is formatted for workshop (shorter)

## 📅 Typical Timeline

- Abstract submission: ~May 2025
- Full paper deadline: ~June 2025  
- Review period: ~June-July 2025
- Author response: ~August 2025
- Notification: ~September 2025
- Camera ready: ~October 2025
- Conference: December 2025

## ✅ Final Confirmation

**The paper is NeurIPS-ready with:**
- Complete LLM usage disclosure ✓
- Code availability statement ✓
- Reproducibility information ✓
- Proper anonymization ✓
- Real experimental results ✓
- Formal guarantees ✓
- Ethics statement ✓
- All citations verified ✓