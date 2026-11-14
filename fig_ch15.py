"""Figure for Chapter 15.

Run:  python3 code/fig_ch15.py
  figures/ch15_activity.pdf     (Fig. 15.1)
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
src = (HERE / "ch15_checks.py").read_text()
ns = {"__file__": str(HERE / "ch15_checks.py")}
exec(src.split("\n# ===================================================================== helpers\n")[0], ns)
winding_tower = ns["winding_tower"]


def delta(q, x, beta=1.0):
    A = x / (-beta)**q
    om = winding_tower([0.0] * q, A, beta)
    w1 = x / math.factorial(q - 1)          # omega_1 for equal energies at 0
    return om / w1 - 1


xs = np.geomspace(1e-2, 4e2, 120)
fig, ax = plt.subplots(figsize=(4.2, 2.7))
for colour, q in zip(INK, [2, 3]):
    pos = [abs(delta(q, x)) for x in xs]
    neg = [abs(delta(q, -x)) for x in xs]
    bound = [sum(math.exp((n - 1) * math.log(x) + math.lgamma(q + 1) - math.lgamma(n * q + 1))
                 for n in range(2, 60))
             for x in xs]
    ax.loglog(xs, pos, color=colour, lw=1.1, label=f"$q={q}$, $x_W>0$")
    ax.loglog(xs, neg, "o", ms=2, mew=0, color=colour, label=f"$q={q}$, $x_W<0$")
    ax.loglog(xs, bound, color=colour, lw=0.7, ls=(0, (3, 3)))
ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (1, 2)))
ax.set_xlabel(r"$|x_W|$")
ax.set_ylabel(r"$|\delta_W|$")
ax.set_ylim(1e-4, 1e3)
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "ch15_activity.pdf")
fig.savefig(OUT / "ch15_activity.png", dpi=200)
print("Fig 15.1 done")
