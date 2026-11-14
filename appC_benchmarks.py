"""Benchmark node sets B1-B8 of Section 9.5 and Appendix C, with reference
values of ln exp[x_0..x_q] to 30 significant digits.

Run:  python3 code/appC_benchmarks.py
Writes code/benchmarks/B1.txt ... B8.txt (one node per line, as the
shortest repr of the double) and code/benchmarks/reference.txt.
"""
import math
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
OUT = HERE / "benchmarks"
OUT.mkdir(exist_ok=True)
ns = {"__file__": str(HERE / "ch09_checks.py")}
src = (HERE / "ch09_checks.py").read_text()
exec(src.split("# ============================================================== 9.1 Taylor")[0], ns)
benchmark_sets = None
exec(src[src.index("def benchmark_sets():"):src.index("def table_dd(")], ns)
B = ns["benchmark_sets"]()
ref_exp = ns["ref_exp"]
lines = []
for k, (name, xs) in enumerate(B.items(), 1):
    (OUT / f"B{k}.txt").write_text("\n".join(repr(float(x)) for x in xs) + "\n")
    with mp.workdps(60):
        v = ref_exp(xs, dps=50)
        lv = mp.log(v)
    s = mp.nstr(lv, 30)
    lines.append(f"B{k}\t{len(xs) - 1}\t{min(xs)!r}\t{max(xs)!r}\t{s}")
    print(f"B{k}  q={len(xs) - 1:4d}  ln exp[x] = {s}   ({name})")
(OUT / "reference.txt").write_text(
    "# set\tq\tmin\tmax\tln exp[x_0..x_q] (30 digits)\n" + "\n".join(lines) + "\n")
