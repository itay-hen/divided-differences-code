"""Figure for Chapter 12.

Run:  python3 code/fig_ch12.py
  figures/ch12_cheb_sign.pdf     (Fig. 12.1)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
INK = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "mathtext.fontset": "cm",
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.linewidth": 0.6,
})

src = (HERE / "ch12_checks.py").read_text()
ns = {"__file__": str(HERE / "ch12_checks.py")}
exec(src.split("# ============================================================== 12.1")[0], ns)
pmr_decompose, cheb_walk_moment, op, SX = (ns["pmr_decompose"], ns["cheb_walk_moment"],
                                           ns["op"], ns["SX"])

Hx = (sum(op(3, i, SX) for i in range(3)) + 0.5 * op(3, 0, SX) @ op(3, 1, SX)) / 3.6
pm = pmr_decompose(Hx, 1e-14)
E = pm.pop(0).real
offs = {b: d.real for b, d in pm.items()}
B = sum(np.diag(np.abs(d))[:, np.arange(8) ^ b] for b, d in offs.items())
lb = np.linalg.eigvals(B)
lam = np.linalg.eigvalsh(Hx)

ns_all = np.arange(2, 31, 2)
mu = [abs(np.cos(n * np.arccos(lam)).sum()) for n in ns_all]
pred = [sum(np.cosh(n * np.arcsinh(complex(l))).real for l in lb) for n in ns_all]
ns_dfs = [2, 4, 6, 8]
dfs = [cheb_walk_moment(E, offs, n, absolute=True) for n in ns_dfs]

fig, ax = plt.subplots(figsize=(4.0, 2.6))
ax.semilogy(ns_all, pred, color=INK[1], lw=1.0, label=r"$\sum_i\cosh(n\,\mathrm{arsinh}\,\lambda_i(B))$")
ax.semilogy(ns_dfs, dfs, "s", ms=3.2, mew=0, color=INK[1], label=r"$\sum|W_n|$ (walks)")
ax.semilogy(ns_all, mu, "o", ms=3, mew=0, color=INK[0], label=r"$|\mu_n|$")
ax.axhline(8, color=MUTED, lw=0.7, ls=(0, (3, 3)))
ax.annotate("$N=8$", (26, 8), xytext=(0, 3), textcoords="offset points", color=MUTED, fontsize=7)
ax.set_xlabel("moment index $n$")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "ch12_cheb_sign.pdf")
fig.savefig(OUT / "ch12_cheb_sign.png", dpi=200)
print("Fig 12.1 done; ratio at n=10:", mu[4] / pred[4], " n=30:", mu[-1] / pred[-1])
