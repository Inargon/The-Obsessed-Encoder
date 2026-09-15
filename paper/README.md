# Paper draft

This directory contains the working paper. The current document deliberately
separates completed evidence from in-progress and proposed experiments.

Compile with a standard TeX Live installation:

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Before submission:

1. Replace anonymous authors and select the venue template.
2. Replace all red `TODO` markers.
3. Report multi-seed means and confidence intervals under one evaluator.
4. Add the method schematic, learning curves, geometry plot, and gradient-span
   diagnostic.
5. Audit every number against archived metrics and record its protocol.
