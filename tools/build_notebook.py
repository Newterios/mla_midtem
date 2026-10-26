"""Turn a percent-format Python file into a Jupyter notebook.

    .venv/bin/python tools/build_notebook.py notebooks_src/endterm.py Endterm.ipynb

Cell markers:  `# %%` starts a code cell, `# %% [markdown]` starts a markdown
cell (its lines are prefixed with `# `, which is stripped).  Everything before
the first marker is ignored.
"""
import sys
from pathlib import Path

import nbformat as nbf


def parse(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l for l in body.split("\n")]
            cells.append(nbf.v4.new_markdown_cell("\n".join(lines).strip("\n")))
        else:
            cells.append(nbf.v4.new_code_cell(body))

    for line in text.split("\n"):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def main(src: str, dst: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb["cells"] = parse(Path(src).read_text(encoding="utf-8"))
    nbf.write(nb, dst)
    print(f"{dst}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
