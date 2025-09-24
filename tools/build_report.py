import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "docs" / "reports" / "main.tex"
OUTDIR = ROOT / "docs" / "reports"

if not TEX.exists():
    sys.exit(f"Missing {TEX}. Create it first.")

# Read the original LaTeX content
content = TEX.read_text()

# Add \usepackage{svg} before \begin{document}
if r"\usepackage{svg}" not in content:
    content = content.replace(r"\begin{document}", r"\usepackage{svg}" + "\n" + r"\begin{document}")

# Replace \includegraphics with \includesvg for pipeline_overview
content = content.replace(r"\includegraphics[width=\linewidth]{pipeline_overview}", r"\includesvg[width=\linewidth]{pipeline_overview}")

# Write the modified content back to the file
TEX.write_text(content)

cmd = ["tectonic", "--outdir", str(OUTDIR), str(TEX)]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
pdf = OUTDIR / "main.pdf"
print("OK ->", pdf if pdf.exists() else "No PDF found")