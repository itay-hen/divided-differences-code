"""Copy the Chapter 10 listings from ch10_checks.py into the chapter."""
import re
from pathlib import Path
here = Path(__file__).resolve().parent
src = (here / "ch10_checks.py").read_text()
tex_p = here.parent / "chapters" / "ch10-matrix-functions.tex"
t = tex_p.read_text()


def grab(start, end):
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j].rstrip() + "\n"


L = {"lst:mf-dk": grab("def loewner_exp(", "\n\n# ---- Listing 10.2"),
     "lst:mf-second": grab("def frechet2_hermitian(", "\n\n# ---- Listing 10.3"),
     "lst:mf-leja": grab("def leja_interval(", "\n\n# ---- Listing 10.4"),
     "lst:mf-walks": grab("def dd_push(", "\n\n# ---- Listing 10.5"),
     "lst:mf-matrix-dd": grab("def matrix_dd_exp(", "\n\n# ===================================================================== helpers")}
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
