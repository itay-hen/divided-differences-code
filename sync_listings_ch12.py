"""Copy the Chapter 12 listings from ch12_checks.py into the chapter."""
import re
from pathlib import Path
here = Path(__file__).resolve().parent
src = (here / "ch12_checks.py").read_text()
tex_p = here.parent / "chapters" / "ch12-partition-functions.tex"
t = tex_p.read_text()


def grab(start, end):
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j].rstrip() + "\n"


L = {"lst:pf-pmr": grab("def pmr_decompose(", "\n\n# ---- Listing 12.2"),
     "lst:pf-cheb": grab("def cheb_walk_moment(", "\n\n# ===================================================================== helpers")}
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
