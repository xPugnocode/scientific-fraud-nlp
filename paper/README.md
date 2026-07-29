# Paper Source

This folder contains the LaTeX source used to build the paper.

## Files

- `icr_paper.tex`: main LaTeX source file
- `references.bib`: bibliography database
- `naaclhlt2016.sty`: local conference-style template used by the paper
- `appendix/appendix1.tex`: supplementary appendix included by the main file
- `figures/selected_feature_distributions.png`: figure used by the paper

## Build

From this folder, build the PDF with `latexmk`:

```bash
cd paper
latexmk -pdf icr_paper.tex
```

If `latexmk` is not installed, run the LaTeX and BibTeX steps manually:

```bash
cd paper
pdflatex icr_paper.tex
bibtex icr_paper
pdflatex icr_paper.tex
pdflatex icr_paper.tex
```

The output PDF will be `icr_paper.pdf`.

To remove generated LaTeX build files while keeping the PDF:

```bash
latexmk -c
```
