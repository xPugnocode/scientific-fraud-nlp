# Paper Source

This folder contains the LaTeX source used to build the paper.

## Files

- `icr_paper.tex`: main LaTeX source file
- `references.bib`: bibliography database
- `naaclhlt2016.sty`: local conference-style template used by the paper
- `appendix/appendix1.tex`: supplementary appendix included by the main file
- `figures/selected_feature_distributions.png`: figure used by the paper

## Build

Install the required build command:

```bash
sudo apt update
sudo apt install latexmk
```

From this folder, build the PDF:

```bash
cd paper
latexmk -pdf icr_paper.tex
```

The output PDF will be `icr_paper.pdf`.

To remove generated LaTeX build files while keeping the PDF:

```bash
latexmk -c
```
