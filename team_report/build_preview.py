"""Preview build of the report with Tectonic (XeTeX) when no TeX Live is installed.

The official build is `make` (latexmk + pdflatex + biber, as in the TUD template) or the
TU Dresden Overleaf. Tectonic differs in two ways, so this script compiles a patched copy,
_preview.tex, and leaves main.tex untouched:
  * no biber: biblatex uses its BibTeX backend instead;
  * XeTeX + fontspec: tudscr would look up Open Sans as a system font, which is not
    installed, so its fontspec path is switched off and headings fall back to the
    default sans font.
Text, layout and numbering are otherwise the same; heading fonts differ slightly.

    conda run -n tex python team_report/build_preview.py
"""
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
BEGIN = r"\begin{document}"
SHIM = r"\makeatletter\@tud@x@fontspec@enabledfalse\makeatother"

src = (HERE / "main.tex").read_text(encoding="utf-8")
assert "backend=biber" in src and BEGIN in src
src = src.replace("backend=biber", "backend=bibtex", 1)
src = src.replace(BEGIN, SHIM + "\n" + BEGIN, 1)
(HERE / "_preview.tex").write_text(src, encoding="utf-8")

out = HERE / "_build"
out.mkdir(exist_ok=True)
r = subprocess.run(["tectonic", "-X", "compile", "_preview.tex", "--keep-logs", "--outdir", str(out)],
                   cwd=HERE)
if r.returncode:
    sys.exit(f"tectonic failed; see {out / '_preview.log'}")
shutil.copy(out / "_preview.pdf", HERE / "main.pdf")
print("wrote", HERE / "main.pdf")
