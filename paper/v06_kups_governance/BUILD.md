# Building the v0.6 PDF

Standalone v0.6 report, built with the standard LaTeX toolchain (pdflatex + bibtex).

Sources: `main.tex` (preamble, frontmatter, figures appendix), `sections_body.tex`,
`metadata.yaml`, and the committed figure PDFs under `figures/`.

```bash
latexmk -pdf main.tex   # -> main.pdf
```

Every reported number traces to committed JSON under `evidence/` in the
version-controlled repository state.
