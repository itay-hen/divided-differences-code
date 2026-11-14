"""Figure for Chapter 10.

Run:  python3 code/fig_ch10.py
  figures/ch10_leja.pdf      (Fig. 10.1)
"""
import math
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

# reuse Listing 10.3 from the check script without running its checks
src = (HERE / "ch10_checks.py").read_text()
ns = {"__file__": str(HERE / "ch10_checks.py")}
exec(src.split("# ===================================================================== helpers")[0], ns)
leja_interval = ns["leja_interval"]

n = 400
A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
lam, Q = np.linalg.eigh(A)
v = np.random.default_rng(0).standard_normal(n)
v /= np.linalg.norm(v)

fig, ax = plt.subplots(figsize=(4.6, 2.8))
for colour, t in zip(INK, [1, 25, 300, 3000]):
    m = int(14 * math.sqrt(t)) + 60
    xi = leja_interval(0.0, 4.0, m)
    d = [math.exp(-t * x) for x in xi]               # table, Leja order
    coef = [d[0]]
    for k in range(1, m):
        d = [(d[i + 1] - d[i]) / (xi[i + k] - xi[i]) for i in range(m - k)]
        coef.append(d[0])
    ref = Q @ (np.exp(-t * lam) * (Q.T @ v))
    r, w, errs = v.copy(), coef[0] * v, []
    for k in range(1, m):
        r = A @ r - xi[k - 1] * r
        w = w + coef[k] * r
        errs.append(max(np.linalg.norm(w - ref) / np.linalg.norm(ref), 1e-17))
    ax.semilogy(range(1, m), errs, lw=1.1, color=colour, label=f"$t={t}$")
    est = math.sqrt(2 * (2 * t) * math.log(1e12))
    ax.axvline(est, color=colour, lw=0.7, ls=(0, (3, 3)))
ax.set_xlabel("products with $A$")
ax.set_ylabel("relative error")
ax.set_ylim(1e-16, 1e3)
ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "ch10_leja.pdf")
fig.savefig(OUT / "ch10_leja.png", dpi=200)
print("Fig 10.1 done")
