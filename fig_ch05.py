"""Figures for Chapter 5.

Run:  python3 code/fig_ch05.py
  figures/ch05_chebyshev_convergence.pdf   (Fig. 5.2)
  figures/ch05_sandwich.pdf                (Fig. 5.1)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

mp.mp.dps = 60
OUT = Path(__file__).resolve().parent.parent / "figures"
INK = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, fixed order
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


# ---------------------------------------------------------------------------
# Chebyshev--Bessel convergence
def cheb_column(ys, N):
    q = len(ys) - 1
    prev = [mp.mpf(0)] * (q + 1); prev[0] = mp.mpf(1)            # T_0
    cur = [mp.mpf(0)] * (q + 1); cur[0] = ys[0]
    if q >= 1:
        cur[1] = mp.mpf(1)                                         # T_1
    col = [prev[q], cur[q]]
    for n in range(1, N):
        nxt = [2 * (ys[k] * cur[k] + (cur[k - 1] if k else 0)) - prev[k] for k in range(q + 1)]
        prev, cur = cur, nxt
        col.append(cur[q])
    return col                                                     # T_n[y_0..y_q], n=0..N


def cb_partial_sums(xs, N):
    a, b = min(xs), max(xs)
    d, c = (a + b) / 2, (b - a) / 2
    ys = [(x - d) / c for x in xs]
    q = len(xs) - 1
    D = cheb_column(ys, N)
    I0 = mp.besseli(0, c)
    pref = mp.e**d * I0 / c**q
    sums, acc = [], D[0]
    sums.append(pref * acc)
    for n in range(1, N + 1):
        acc += 2 * mp.besseli(n, c) / I0 * D[n]
        sums.append(pref * acc)
    return sums


q = 10
rng = np.random.default_rng(11)
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for colour, c in zip(INK, (2, 20, 200)):
    u = np.sort(rng.uniform(-1, 1, q + 1)); u[0], u[-1] = -1, 1
    xs = [mp.mpf(c) * mp.mpf(v) for v in u]
    Nmax = int(c + 12 * np.sqrt(c) + 60)
    sums = cb_partial_sums(xs, Nmax + 40)
    ref = sums[-1]
    err = [float(abs(s - ref) / abs(ref)) for s in sums[: Nmax + 1]]
    err = np.clip(err, 1e-17, 1.0)            # partial sums can exceed the value by many orders first
    k = int(np.argmax(np.array(err) < 1e-16))
    ax.semilogy(range(k + 1), err[: k + 1], color=colour, lw=1.4)
    ax.annotate(f"$c={c}$", (k, 1e-16), xytext=(3, 3), textcoords="offset points",
                color=TEXT, fontsize=7.5)
ax.axvline(q, color=MUTED, lw=0.6, ls=(0, (3, 3)))
ax.annotate("$n=q$", (q, 1e-3), xytext=(3, 0), textcoords="offset points", color=MUTED, fontsize=7)
ax.set_xlim(0, 160); ax.set_ylim(1e-16, 3)
ax.set_xlabel("truncation index $N$")
ax.set_ylabel("relative error")
style(ax)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "ch05_chebyshev_convergence.pdf"); fig.savefig(OUT / "ch05_chebyshev_convergence.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Mean--variance sandwich (Zeng et al.)
mp.mp.dps = 30


def LM(n, sig):
    a0 = mp.sqrt(n) * sig
    a = (n + 1) * sig / mp.sqrt(n)
    L = n * mp.e**(-a0) * mp.quad(lambda s: s**(n - 1) * mp.e**(a * s), [0, 1])
    M = n * mp.e**(a0) * mp.quad(lambda s: s**(n - 1) * mp.e**(-a * s), [0, 1])
    return float(L), float(M)


def ee(xs):
    q = len(xs) - 1
    Z = mp.matrix(q + 1, q + 1)
    for i, x in enumerate(xs):
        Z[i, i] = mp.mpf(x)
        if i < q:
            Z[i, i + 1] = 1
    return mp.expm(Z)[0, q]


n = 4
sig_grid = np.linspace(0.01, 3, 120)
Ls, Ms = zip(*(LM(n, s) for s in sig_grid))
pts_s, pts_v = [], []
for _ in range(400):
    x = rng.normal(size=n + 1) * rng.uniform(0.05, 1.4)
    if rng.random() < 0.3:                       # include some one-outlier sets
        x = np.r_[x[:1] * 3, x[1:] * 0.2]
    s = np.std(x)
    if s > 3:
        continue
    pts_s.append(s)
    pts_v.append(float(mp.factorial(n) * ee(list(x)) / mp.e**np.mean(x)))
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.scatter(pts_s, pts_v, s=9, color=MUTED, alpha=0.45, lw=0, label="random node sets")
ax.plot(sig_grid, Ms, color=INK[0], lw=1.4, label=r"upper bound $M_n(\sigma)$")
ax.plot(sig_grid, Ls, color=INK[1], lw=1.4, label=r"lower bound $L_n(\sigma)$")
ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 3)))
ax.annotate("Jensen lower bound", (3.0, 1), xytext=(0, 3), textcoords="offset points", color=MUTED,
            fontsize=7, ha="right")
ax.set_xlabel(r"population standard deviation $\sigma$ of the nodes")
ax.set_ylabel(r"$n!\,\mathrm{e}^{[x_0,\ldots,x_n]}\,\mathrm{e}^{-\mu}$")
ax.legend(frameon=False, fontsize=7, loc="upper left")
style(ax)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "ch05_sandwich.pdf"); fig.savefig(OUT / "ch05_sandwich.png", dpi=150)
print("wrote figures")
