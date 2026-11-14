"""Figures for Chapter 8.

Run:  python3 code/fig_ch08.py
  figures/ch08_newton_order.pdf    (Fig. 8.1)
  figures/ch08_contour.pdf         (Fig. 8.2)
  figures/ch08_clusters.pdf        (Fig. 8.3)
"""
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

sys.argv = sys.argv[:1]
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
src = (HERE / "ch08_checks.py").read_text()
ns = {}
exec(src.split("# ============================================================== 8.1 ordering")[0], ns)
leja_order, dd_contour, dd_taylor = ns["leja_order"], ns["dd_contour"], ns["dd_taylor"]
table, table_prefix, newton_eval = ns["table"], ns["table_prefix"], ns["newton_eval"]
opitz = ns["opitz"]


def ref_exp(xs, dps=60):
    with mp.workdps(dps):
        return mp.expm(opitz(xs))[0, len(xs) - 1]


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- Fig 8.1
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5), sharey=True)
qs = list(range(10, 151, 10))
t = np.linspace(-1, 1, 2001)
for ax, (title, f) in zip(axes, [(r"$f(x)=\mathrm{e}^x$", np.exp),
                                 (r"$f(x)=1/(1+4x^2)$", lambda x: 1 / (1 + 4 * x * x))]):
    curves = {"increasing": [], "Leja": [], "random": []}
    for q in qs:
        x = np.cos(np.pi * (np.arange(q + 1) + 0.5) / (q + 1))
        rnd = list(x)
        random.Random(q).shuffle(rnd)
        for name, o in [("increasing", sorted(x)), ("Leja", leja_order(list(x))), ("random", rnd)]:
            o = np.array(o)
            c = np.array(table_prefix(f(o), o))
            with np.errstate(all="ignore"):
                e = np.max(np.abs(newton_eval(c, o, t) - f(t)))
            curves[name].append(min(max(e, 1e-17), 1e30) if np.isfinite(e) else 1e30)
    for colour, (name, ys) in zip(INK, curves.items()):
        ax.semilogy(qs, ys, "o-", ms=2.5, lw=1.1, color=colour, label=name, mew=0)
    ax.set_title(title, fontsize=8, color=TEXT)
    ax.set_xlabel("degree $q$")
    ax.set_ylim(1e-17, 1e31)
    style(ax)
axes[0].set_ylabel(r"$\max_{[-1,1]}|p_q-f|$")
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "ch08_newton_order.pdf")
fig.savefig(OUT / "ch08_newton_order.png", dpi=200)
plt.close(fig)
print("Fig 8.1 done")


# ---------------------------------------------------------------- Fig 8.2
def trap(xs, c, R, N):
    th = 2 * np.pi * np.arange(N) / N
    w = np.exp(1j * th)
    z = c + R * w
    tt = np.exp(z) * w
    for x in xs:
        tt = tt / ((z - x) / R)
    return tt.mean() / R**(len(xs) - 1), np.abs(tt).mean() / R**(len(xs) - 1)


q = 20
rr = random.Random(3)
xs = [0.2 + 1e-3 * rr.random() for _ in range(q + 1)]
F = float(ref_exp(xs))
c = sum(xs) / (q + 1)
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
ax = axes[0]
Ns = list(range(4, 129, 4))
for colour, R in zip(INK, [q / 2 + 1, q + 1, 2 * q + 2]):
    errs = [max(abs(trap(xs, c, R, N)[0] - F) / F, 1e-17) for N in Ns]
    ax.semilogy(Ns, errs, "o-", ms=2.2, lw=1.0, color=colour, mew=0, label=f"$R={R:.0f}$")
ax.set_xlabel("number of points $N$")
ax.set_ylabel("relative error")
ax.set_ylim(1e-17, 1e3)
ax.legend(frameon=False, fontsize=7)
style(ax)
ax = axes[1]
Rs = np.logspace(math.log10(2), math.log10(120), 40)
errs, model = [], []
for R in Rs:
    T, S = trap(xs, c, R, 1024)
    errs.append(max(abs(T - F) / F, 1e-17))
    model.append(U * S / F)
ax.loglog(Rs, errs, "o", ms=2.6, color=INK[0], mew=0, label="error, $N=1024$")
ax.loglog(Rs, model, color=MUTED, lw=1.0, label=r"$u\,S/|F|$")
ax.axvline(q + 1, color=INK[1], lw=0.7, ls=(0, (3, 3)))
ax.annotate("$R=q+1$", (q + 1, 1e-8), xytext=(3, 0), textcoords="offset points", color=INK[1], fontsize=7)
ax.set_xlabel("radius $R$")
ax.set_ylim(1e-17, 1e3)
ax.legend(frameon=False, fontsize=7, loc="upper center")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "ch08_contour.pdf")
fig.savefig(OUT / "ch08_contour.png", dpi=200)
plt.close(fig)
print("Fig 8.2 done")

# ---------------------------------------------------------------- Fig 8.3


def clusters(gap, m=10, seed=4):
    r = random.Random(seed)
    xs, cl = [], []
    for k in range(3):
        lo = len(xs)
        xs += sorted(k * gap + r.uniform(0, 1e-6) for _ in range(m))
        cl.append((lo, len(xs)))
    return np.array(xs), cl


def exp_coef(sigma, K):
    out = [math.exp(sigma)]
    for m in range(1, K + 1):
        out.append(out[-1] / m)
    return out


def block_parlett(xs, cl, K=80):
    n = len(xs)
    Fm = np.zeros((n, n))
    lab = np.empty(n, int)
    for k, (lo, hi) in enumerate(cl):
        lab[lo:hi] = k
        sig = xs[lo]
        C = list(xs[lo:hi])
        for j in range(hi - lo):
            Fm[lo:lo + j + 1, lo + j] = dd_taylor(exp_coef(sig, K), C[:j + 1], sig)
    for d in range(1, n):
        for i in range(n - d):
            j = i + d
            if lab[i] != lab[j]:
                Fm[i, j] = (Fm[i + 1, j] - Fm[i, j - 1]) / (xs[j] - xs[i])
    return Fm[0, n - 1]


def cluster_sum(xs, cl):
    tot = 0
    for (lo, hi) in cl:
        C = xs[lo:hi]
        O = np.concatenate([xs[:lo], xs[hi:]])
        g = lambda z, O=O: np.exp(z) / np.prod([(z - o) for o in O], axis=0)
        cc = np.mean(C)
        rad = min(0.4 * min(abs(o - cc) for o in O), np.abs(C - cc).max() + len(C))
        tot += dd_contour(g, C, R=rad)
    return tot.real


def taylor_scaled(xs):
    q = len(xs) - 1
    s = xs[-1] - xs[0]
    nt = 1
    while nt * math.log(s) - math.lgamma(nt + 1) > math.log(1e-17) or nt < 3 * s:
        nt += 1
    K = q + nt
    coef = [math.exp(xs[0])]
    for m in range(1, K + 1):
        coef.append(coef[-1] * s / m)
    return dd_taylor(coef, list((xs - xs[0]) / s), 0.0)[0] / s**q


gaps = np.logspace(-0.5, 2.5, 25)
res = {k: [] for k in ["cluster Parlett", "sum over clusters", "contour (one circle)", "Taylor"]}
for gap in gaps:
    xs, cl = clusters(gap)
    F = float(ref_exp(list(xs), dps=60 + int(2 * gap / 2.3) + 40))
    vals = {
        "cluster Parlett": block_parlett(xs, cl),
        "sum over clusters": cluster_sum(xs, cl),
        "contour (one circle)": dd_contour(np.exp, xs).real,
        "Taylor": taylor_scaled(xs),
    }
    for k, v in vals.items():
        e = abs(v - F) / F if np.isfinite(v) else 1e30
        res[k].append(min(max(e, 1e-17), 1e30))
fig, ax = plt.subplots(figsize=(4.6, 2.8))
colours = [INK[1], INK[2], INK[0], INK[3]]
marks = ["s", "^", "o", "d"]
for colour, mk, (k, ys) in zip(colours, marks, res.items()):
    ax.loglog(gaps, ys, marker=mk, ms=2.8, lw=1.0, color=colour, label=k, mew=0.6 if mk == "x" else 0)
ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 3)))
ax.set_ylim(1e-17, 1e31)
ax.set_xlabel("gap between clusters")
ax.set_ylabel("relative error")
ax.legend(frameon=False, fontsize=6.5, loc="center right")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "ch08_clusters.pdf")
fig.savefig(OUT / "ch08_clusters.png", dpi=200)
plt.close(fig)
for k, ys in res.items():
    print(f"{k:22s}", " ".join(f"{y:.0e}" for y in ys[::4]))
print("Fig 8.3 done")
