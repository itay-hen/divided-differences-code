"""Copy the Chapter 13 listings from ch13_checks.py into the chapter."""
import re
from pathlib import Path
here = Path(__file__).resolve().parent
src = (here / "ch13_checks.py").read_text()
tex_p = here.parent / "chapters" / "ch13-qmc.tex"
t = tex_p.read_text()


def grab(start, end):
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j].rstrip() + "\n"


L = {"lst:qmc-state": grab("class PMRQMC:", "\n\n# ---- Listing 13.2"),
     "lst:qmc-moves": grab("def pmr_move(", "\n\n# ---- Listing 13.3"),
     "lst:qmc-est": grab("def estimators(", "\n\n# ---- Listing 13.4"),
     "lst:qmc-fidsus": grab("def fidsus_terms(", "\n\n# ===================================================================== helpers")}
for lab, code in L.items():
    pat = re.compile(r"(label=" + re.escape(lab) + r"\]\n)(.*?)(\\end\{lstlisting\})", re.S)
    m = pat.search(t)
    assert m, lab
    t = t[:m.start(2)] + code + t[m.end(2):]
tex_p.write_text(t)
long = [(lab, l) for lab, c in L.items() for l in c.split("\n") if len(l) > 70]
print("synced", {k: v.count("\n") for k, v in L.items()}, "long lines:", len(long))
for lab, l in long:
    print(" ", lab, len(l), l)
