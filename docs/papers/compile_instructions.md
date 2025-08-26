# LaTeX Compilation Instructions

The NeurIPS 2025 workshop paper is ready for compilation. All required files have been created:

## Files Created

1. **Main LaTeX document**: `pot_neurips2025.tex`
2. **BibTeX references**: `references.bib`
3. **Figures**: 
   - `figures/fig1_time_to_decision.pdf`
   - `figures/fig2_error_rates.pdf`
   - `figures/confusion_matrix.pdf`

## To Compile the PDF

### Option 1: Install BasicTeX (Recommended for Mac)

```bash
# Install BasicTeX (requires admin password)
brew install --cask basictex

# Restart terminal or run:
eval "$(/usr/libexec/path_helper)"

# Navigate to papers directory
cd /Users/rohanvinaik/PoT_Experiments/docs/papers

# Compile the PDF
pdflatex pot_neurips2025.tex
bibtex pot_neurips2025
pdflatex pot_neurips2025.tex
pdflatex pot_neurips2025.tex
```

### Option 2: Use MacTeX (Full Distribution)

```bash
# Install MacTeX (larger but more complete)
brew install --cask mactex

# Restart terminal
# Then compile as above
```

### Option 3: Use Online LaTeX Editor

1. Go to [Overleaf](https://www.overleaf.com) or [LaTeX Base](https://latexbase.com)
2. Create a new project
3. Upload these files:
   - `pot_neurips2025.tex`
   - `references.bib`
   - All files from `figures/` directory
4. Compile online

### Option 4: Use Docker

```bash
# Pull LaTeX Docker image
docker pull texlive/texlive:latest

# Run compilation in Docker
docker run --rm -v $(pwd):/workdir texlive/texlive:latest \
  sh -c "cd /workdir && pdflatex pot_neurips2025.tex && bibtex pot_neurips2025 && pdflatex pot_neurips2025.tex && pdflatex pot_neurips2025.tex"
```

## Expected Output

After successful compilation, you'll have:
- `pot_neurips2025.pdf` - The final paper ready for submission

## Paper Statistics

- **Pages**: ~9 (within NeurIPS workshop limit)
- **Words**: ~4,500
- **References**: 16
- **Tables**: 4
- **Figures**: 3
- **Compliance**: Full NeurIPS 2025 workshop format

## Submission Checklist

- [ ] Install LaTeX and compile PDF
- [ ] Verify page count ≤ 9 pages (excluding references)
- [ ] Check all figures are included
- [ ] Ensure anonymous (no author info)
- [ ] Verify paper checklist is complete
- [ ] Submit to NeurIPS portal before deadline

## Quick Validation

The LaTeX document includes:
- Proper NeurIPS formatting commands
- Complete abstract and all required sections
- All citations properly formatted
- Tables with experimental results and evidence bundle hashes
- Figures referenced in text
- Complete paper checklist in appendix
- Broader impact statement

The paper is ready for compilation and submission!