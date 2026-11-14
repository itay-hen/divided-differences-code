"""Figures for Chapter 9.

Run:  python3 code/fig_ch09.py
  figures/ch09_spread.pdf     (Fig. 9.1)
  figures/ch09_lambda.pdf     (Fig. 9.2)
"""
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
U = 2.0**-53
INK = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "mathtext.fontset": "cm",
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.linewidth": 0.6,
})

# reuse the listings and helpers from the check script without running its checks
src = (HERE / "ch09_checks.py").read_text()
ns = {"__file__": str(HERE / "ch09_checks.py")}
exec(src.split("# ============================================================== 9.1 Taylor")[0], ns)
dd_exp_taylor, dd_exp_squaring = ns["dd_exp_taylor"], ns["dd_exp_squaring"]
dd_exp_chebyshev, ExpDD = ns["dd_exp_chebyshev"], ns["ExpDD"]
ref_exp, rel_log, nodes = ns["ref_exp"], ns["rel_log"], ns["nodes"]


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def squaring_log(xs):
    lo, th, G = dd_exp_squaring(xs)
    return lo + math.log(G[0, -1]) - (len(xs) - 1) * math.log(th)


def adds_log(xs):
    s = max(xs) - min(xs)
    D = ExpDD(min(xs), max(1, math.ceil(math.e * s + math.log(1 / U))))
    for x in xs:
        D.add(x)
    return D.log_value()


def timed(fn, xs, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(xs)
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[reps // 2]


# ---------------------------------------------------------------- Fig 9.1
METHODS = [("Taylor (9.1)", dd_exp_taylor), ("squaring (9.2)", squaring_log),
           ("Chebyshev (9.3)", dd_exp_chebyshev), ("adds (9.4)", adds_log)]
q = 20
spreads = np.logspace(-2, math.log10(500), 21)
err = {k: [] for k, _ in METHODS}
tim = {k: [] for k, _ in METHODS}
for s in spreads:
    rng = random.Random(int(1000 * s))
    xs = [0.0, float(s)] + [s * rng.random() for _ in range(q - 1)]
    F = ref_exp(xs)
    for k, fn in METHODS:
        e = rel_log(fn(xs), F)
        err[k].append(min(max(e, 1e-17), 1e3))
        tim[k].append(1e3 * timed(fn, xs))
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
for colour, (k, _) in zip(INK, METHODS):
    axes[0].loglog(spreads, err[k], "o-", ms=2.5, lw=1.0, color=colour, mew=0, label=k)
    axes[1].loglog(spreads, tim[k], "o-", ms=2.5, lw=1.0, color=colour, mew=0, label=k)
axes[0].loglog(spreads, [s * U for s in spreads], color=MUTED, lw=0.8, ls=(0, (3, 3)))
axes[0].annotate("$s\\,u$", (spreads[-3], spreads[-3] * U), xytext=(-4, 5),
                 textcoords="offset points", color=MUTED, fontsize=7)
axes[0].set_xlabel("spread $s$")
axes[0].set_ylabel("relative error")
axes[0].set_ylim(1e-17, 1e-8)
axes[1].set_xlabel("spread $s$")
axes[1].set_ylabel("time (ms)")
axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
for ax in axes:
    style(ax)
fig.tight_layout()
fig.savefig(OUT / "ch09_spread.pdf")
fig.savefig(OUT / "ch09_spread.png", dpi=200)
plt.close(fig)
for k, _ in METHODS:
    print(f"{k:16s} err", " ".join(f"{e:.0e}" for e in err[k][::4]))
    print(f"{k:16s} ms ", " ".join(f"{t:.2f}" for t in tim[k][::4]))
print("Fig 9.1 done")

# ---------------------------------------------------------------- Fig 9.2
rng = random.Random(21)
FAM = ["unif", "low-cluster", "high-cluster", "two"]
pts = {f: [] for f in FAM}
for t in range(160):
    fam = FAM[t % 4]
    q = rng.choice([1, 3, 10, 30, 60])
    w = 10**rng.uniform(-1, math.log10(400))
    xs = nodes(fam, q, w, rng)
    F = ref_exp(xs)
    try:
        e = rel_log(dd_exp_chebyshev(xs), F)
    except (ValueError, OverflowError, RuntimeError):
        e = 1e30
    with mp.workdps(30):
        lam = float(mp.e**max(xs) / (mp.factorial(q) * F))
    pts[fam].append((lam, min(max(e, 1e-17), 1e30)))
fig, ax = plt.subplots(figsize=(4.0, 2.8))
labels = {"unif": "uniform", "low-cluster": "cluster at bottom, one at top",
          "high-cluster": "cluster at top, one at bottom", "two": "two clusters"}
for colour, f in zip(INK, FAM):
    L, E = zip(*pts[f])
    ax.loglog(L, E, "o", ms=2.6, color=colour, mew=0, label=labels[f])
lam = np.logspace(0, 80, 50)
ax.loglog(lam, U * lam, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.loglog(lam, 1e3 * U * lam, color=MUTED, lw=0.8, ls=(0, (1, 2)))
ax.annotate("$u\\Lambda$", (1e30, U * 1e30), xytext=(6, -6), textcoords="offset points",
            color=MUTED, fontsize=7)
ax.annotate("$10^3u\\Lambda$", (1e22, 1e3 * U * 1e22), xytext=(-34, 2), textcoords="offset points",
            color=MUTED, fontsize=7)
ax.set_xlim(0.5, 1e60)
ax.set_ylim(1e-17, 1e31)
ax.set_xlabel(r"$\Lambda(x)=\mathrm{e}^{\max x}/(q!\,\mathrm{e}^{[x]})$")
ax.set_ylabel("relative error")
ax.legend(frameon=False, fontsize=6, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "ch09_lambda.pdf")
fig.savefig(OUT / "ch09_lambda.png", dpi=200)
plt.close(fig)
small = [e for f in FAM for (L, e) in pts[f] if L <= 10]
print(f"Lambda <= 10: {len(small)} cases, max error {max(small):.1e}")
ratio = max(e / (U * L) for f in FAM for (L, e) in pts[f] if e < 1e30)
print(f"max error/(u Lambda) = {ratio:.1e}")
print("Fig 9.2 done")
