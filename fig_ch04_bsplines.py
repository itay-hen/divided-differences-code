"""Figure 4.1: B-splines M(t; x_0..x_q) as Peano kernels of divided differences.

Run:  python3 code/fig_ch04_bsplines.py   ->  figures/ch04_bsplines.pdf

M is the Curry--Schoenberg normalised B-spline, f[x_0..x_q] = (1/q!) int M f^(q),
int M = 1.  It is computed from scipy's Schoenberg-normalised basis element N
by M = q N / (x_q - x_0).
"""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

INK = "#2a78d6"        # single series per panel
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

cases = [
    ((0, 1), r"$x=(0,1)$"),
    ((0, 1, 3), r"$x=(0,1,3)$"),
    ((0, 1, 2, 3), r"$x=(0,1,2,3)$"),
    ((0, 1, 1, 3), r"$x=(0,1,1,3)$"),
    ((0, 0.4, 1.3, 2.1, 3), r"$x=(0,0.4,1.3,2.1,3)$"),
    ((0, 3, 3, 3), r"$x=(0,3,3,3)$"),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "mathtext.fontset": "cm",
})
fig, axes = plt.subplots(2, 3, figsize=(5.3, 3.0), sharex=True, sharey=True)
for ax, (knots, title) in zip(axes.flat, cases):
    q = len(knots) - 1
    N = BSpline.basis_element(np.array(knots, float), extrapolate=False)
    t = np.linspace(knots[0], knots[-1], 1201)
    M = np.nan_to_num(N(t)) * q / (knots[-1] - knots[0])
    if q == 1:  # indicator: draw the jumps explicitly
        t = np.array([-0.2, 0, 0, 1, 1, 3.2]); M = np.array([0, 0, 1, 1, 0, 0])
    else:
        t = np.concatenate([[-0.2], t, [3.2]]); M = np.concatenate([[0], M, [0]])
    ax.plot(t, M, color=INK, lw=1.4, solid_capstyle="round")
    # knots with multiplicities
    for k, m in Counter(knots).items():
        ax.plot([k], [0], marker="|", ms=6, mew=1.0, color=TEXT, clip_on=False)
        if m > 1:
            right = k == knots[-1]
            ax.annotate(f"×{m}", (k, 0), xytext=(-4 if right else 0, 5), textcoords="offset points",
                        ha="right" if right else "center", fontsize=6.5, color=MUTED)
    ax.set_title(title, fontsize=7.5, color=TEXT, pad=3)
    ax.set_xlim(-0.2, 3.2); ax.set_ylim(0, 1.15)
    ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("$t$")
for ax in axes[:, 0]:
    ax.set_ylabel("$M(t)$")
fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
out = Path(__file__).resolve().parent.parent / "figures" / "ch04_bsplines.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=150)
print("wrote", out)
