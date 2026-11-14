"""Figures for Chapter 7.

Run:  python3 code/fig_ch07.py
  figures/ch07_table_error.pdf     (Fig. 7.1)
  figures/ch07_diagnostic.pdf      (Fig. 7.2)
"""
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

mp.mp.dps = 80
U = 2.0**-53
OUT = Path(__file__).resolve().parent.parent / "figures"
INK = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "mathtext.fontset": "cm",
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.linewidth": 0.6,
})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def table(fvals, xs):
    d = list(fvals)
    n = len(xs)
    for k in range(1, n):
        d = [(d[i + 1] - d[i]) / (xs[i + k] - xs[i]) for i in range(n - k)]
    return d[0]


def ref_exp(xs):
    q = len(xs) - 1
    Z = mp.matrix(q + 1, q + 1)
    for i, x in enumerate(xs):
        Z[i, i] = mp.mpf(x)
        if i < q:
            Z[i, i + 1] = 1
    return mp.expm(Z)[0, q]


# Fig 7.1: error of the table at equally spaced nodes vs spacing h
fig, ax = plt.subplots(figsize=(4.6, 2.7))
hs = np.logspace(-4, 0.5, 46)
for colour, q in zip(INK, (4, 8, 16)):
    errs, model = [], []
    for h in hs:
        xs = [0.3 + j * h for j in range(q + 1)]
        ex = ref_exp(xs)
        comp = table([math.exp(x) for x in xs], xs)
        errs.append(max(float(abs(comp - ex) / ex), 1e-17))
        eh = math.exp(h)
        model.append(min(q * U * ((eh + 1) / (eh - 1))**q, 1e30))
    ax.loglog(hs, errs, "o", ms=2.6, color=colour, mew=0)
    ax.loglog(hs, model, color=colour, lw=1.2, label=f"$q={q}$")
ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 3)))
ax.annotate("100% error", (3.0, 1), xytext=(0, 3), textcoords="offset points", color=MUTED, fontsize=7, ha="right")
ax.set_ylim(1e-17, 1e20)
ax.set_xlabel("node spacing $h$")
ax.set_ylabel("relative error of the table")
ax.legend(frameon=False, fontsize=7, loc="upper right")
style(ax)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "ch07_table_error.pdf"); fig.savefig(OUT / "ch07_table_error.png", dpi=150)
plt.close(fig)

# Fig 7.2: translation-spread estimate vs true error
random.seed(72)
est, true = [], []
for trial in range(300):
    q = random.choice([6, 10, 14])
    h = 10**random.uniform(-1.9, 0.3)
    xs = sorted(random.uniform(0, h) for _ in range(q + 1))
    ex = float(ref_exp(xs))
    vals = []
    for s in (0.0, 0.37, -0.81, 1.3):
        ys = [x + s for x in xs]
        vals.append(table([math.exp(y) for y in ys], ys) * math.exp(-s))
    e = abs(vals[0] - ex) / ex
    if e == 0:
        continue
    true.append(e)
    est.append(max((max(vals) - min(vals)) / ex, 1e-17))
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.loglog(true, est, "o", ms=2.8, color=INK[0], alpha=0.6, mew=0)
lim = [1e-16, 1e8]
ax.loglog(lim, lim, color=MUTED, lw=0.8)
ax.annotate("estimate = error", (1e-3, 1e-3), xytext=(6, -12), textcoords="offset points", color=MUTED, fontsize=7)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("true relative error of the table")
ax.set_ylabel("translation-spread estimate")
style(ax)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "ch07_diagnostic.pdf"); fig.savefig(OUT / "ch07_diagnostic.png", dpi=150)
plt.close(fig)
print("wrote figures")
