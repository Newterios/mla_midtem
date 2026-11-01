"""Replace the ⟦W…⟧ placeholders of the Endterm notebook with the written interpretations.

    .venv/bin/python tools/fill_text.py

Edits the markdown cells of the executed notebook in place (outputs are kept,
markdown has no outputs) and the percent-format source, so that a rebuild
gives the same notebook.  The texts live in tools/texts_endterm.md: each
block starts with a line `⟦KEY⟧` and runs to the next such line.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb"
SRC = ROOT / "notebooks_src/endterm.py"
TEXTS = ROOT / "tools/texts_endterm.md"

blocks = re.split(r"^(⟦W[^⟧]+⟧)\s*$", TEXTS.read_text(encoding="utf-8"), flags=re.M)
texts = {blocks[i]: blocks[i + 1].strip() for i in range(1, len(blocks) - 1, 2)}

nb = json.loads(NB.read_text(encoding="utf-8"))
n_nb = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    src = "".join(cell["source"])
    for key, txt in texts.items():
        if key in src:
            src = src.replace(key, txt); n_nb += 1
    cell["source"] = src
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

src = SRC.read_text(encoding="utf-8"); n_src = 0
for key, txt in texts.items():
    if key in src:
        first, *rest = txt.split("\n")   # the placeholder line already starts with "# "
        src = src.replace(key, "\n".join([first] + ["# " + l if l else "#" for l in rest])); n_src += 1
SRC.write_text(src, encoding="utf-8")
left = re.findall(r"⟦W[^⟧]+⟧", "".join("".join(c["source"]) for c in nb["cells"]))
print(f"filled {n_nb} placeholders in the notebook, {n_src} in the source; still empty: {left or 'none'}")
