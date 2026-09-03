# Report source

`report.tex` and `references.bib` are the LaTeX source of *Count Each Idea Once: Building
Robust Composites from Correlated Alpha Signals* (28 pages, July 2026). The compiled PDF is
`../Count_Each_Idea_Once_report.pdf`.

## Building

The report compiles with [tectonic](https://tectonic-typesetting.github.io/) (no TeX
installation needed) or with any `pdflatex`/`biber`-free XeLaTeX toolchain:

```bash
cd docs/report-source
tectonic report.tex
```

All fifteen figures are shipped prebuilt as vector PDFs in `figures/v2/`, so the build has
no data dependency.

## Regenerating the figures

`figures/make_v2_figures.py` (with the shared style in `figures/figstyle.py`) draws every
figure from the research notebook's checkpoint pickle, which is derived from the proprietary
Ultramarin data and is therefore not distributed. The script is included as the record of
exactly how each figure was produced; it cannot be run from a fresh clone.
