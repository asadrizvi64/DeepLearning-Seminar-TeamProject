# Team project report

Written in the TU Dresden team-project template of the Chair of Machine Learning for
Spatial Understanding ([weigertlab/tud_templates](https://github.com/weigertlab/tud_templates),
`team_project/`): `tudscrartcl` class, corporate-design header, biblatex.

- `main.tex`: the report (title page, abstract, Sections 1-9)
- `appendix.tex`: glossary, evaluation bugs, pre-registered rules, reproducibility
- `figs/`: figures (copied from `presentation/figs`, `report/figures` and `paper/figures`)
- `generated/`: numbers and tables of Section 7, produced by
  `scripts/fidelity/paper_numbers.py`; refresh with `make sync`
- `references.bib`: `paper/refs.bib` plus the entries only the report cites

Still to fill in: Akim Al-Makhdar's matriculation number (title page, in brackets).

## Build

```sh
make          # latexmk + pdflatex + biber (TeX Live), as in the template
make sync     # copy the generated numbers/tables from ../paper
make preview  # without TeX Live: Tectonic build, see build_preview.py
```

On the TU Dresden Overleaf, upload this folder and compile with pdfLaTeX.

The committed `main.pdf` comes from `make preview`: it uses BibTeX instead of biber and a
fallback font for the headings, because Tectonic has neither biber nor the Open Sans
OpenType fonts. Text and numbering are the same as in the official build; page breaks
may shift slightly.
