"""Figure for Chapter 14.

Run:  python3 code/fig_ch14.py
  figures/ch14_phasor.pdf     (Fig. 14.1)
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

K, dtdE = 8, 0.3
u = dtdE / (2 * K)
qs = np.arange(1, 31)
# equally spaced nodes: |e - e_K| / (dt^q/q!) = |e_K|/(dt^q/q!) |1 - (sin u/u)^q|,
# with |e_K| / (dt^q/q!) = (sin(dt dE/2) / (K sin u))^q  (Prop. 14.2)
r = math.sin(u) / u
mod = (math.sin(dtdE / 2) / (K * math.sin(u)))
err = [mod**q * abs(1 - r**q) for q in qs]
fig, ax = plt.subplots(figsize=(4.0, 2.6))
ax.plot(qs, err, "o", ms=3, mew=0, color=INK[0], label="error")
ax.plot(qs, [u * u] * len(qs), color=INK[1], lw=0.9, ls=(0, (3, 3)), label=r"$u^2$ (source)")
ax.plot(qs, [q * u * u / 6 for q in qs], color=INK[2], lw=1.0, label=r"$qu^2/6$ (Thm 14.3)")
ax.axvline(7, color=MUTED, lw=0.6, ls=(0, (1, 2)))
ax.set_xlabel("walk length $q$")
ax.set_ylabel(r"error $/\,(\Delta t^q/q!)$")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "ch14_phasor.pdf")
fig.savefig(OUT / "ch14_phasor.png", dpi=200)
print("Fig 14.1 done")
