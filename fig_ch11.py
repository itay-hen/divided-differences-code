"""Figures for Chapter 11.

Run:  python3 code/fig_ch11.py
  figures/ch11_slicing.pdf    (Fig. 11.1)
  figures/ch11_driven.pdf     (Fig. 11.2)
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl

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
src = (HERE / "ch11_checks.py").read_text()
ns = {"__file__": str(HERE / "ch11_checks.py")}
exec(src.split("# ============================================================== 11.1")[0], ns)
dd_expi_squaring, dyson_driven = ns["dd_expi_squaring"], ns["dyson_driven"]


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def U2_series(a, b, g, t, Q):
    U2, ell1 = np.zeros((2, 2), complex), 0.0
    for col in (0, 1):
        nodes = [(a, b)[(col + r) % 2] for r in range(Q + 1)]
        mid, th, G = dd_expi_squaring(nodes, t)
        w = [(-1j * t / th)**q * np.exp(-1j * t * mid) * G[0, q] * g**q
             for q in range(Q + 1)]
        for q in range(Q + 1):
            U2[(col + q) % 2, col] += w[q]
        ell1 = max(ell1, sum(abs(v) for v in w))
    return U2, ell1


# ---------------------------------------------------------------- Fig 11.1
gts = np.geomspace(0.25, 48, 24)
fig, ax = plt.subplots(figsize=(4.4, 2.7))
for colour, Delta in zip(INK, [0.0, 1.0]):
    one, sliced, l1s = [], [], []
    for gt in gts:
        t = float(gt)
        Uex = sl.expm(-1j * t * np.array([[0.0, 1.0], [1.0, Delta]]))
        U1, l1 = U2_series(0.0, Delta, 1.0, t, int(math.e * t + 40))
        m = max(1, math.ceil(t))
        Us, _ = U2_series(0.0, Delta, 1.0, t / m, 43)
        one.append(max(np.abs(U1 - Uex).max(), 1e-17))
        sliced.append(max(np.abs(np.linalg.matrix_power(Us, m) - Uex).max(), 1e-17))
        l1s.append(U * l1)
    ax.loglog(gts, one, "o-", ms=2.5, lw=1.0, mew=0, color=colour,
              label=rf"one series, $\Delta={Delta:g}$")
    ax.loglog(gts, l1s, color=colour, lw=0.7, ls=(0, (3, 3)))
    ax.loglog(gts, sliced, "s-", ms=2.3, lw=0.8, mew=0, color=colour, alpha=0.55,
              label=rf"$\lceil gt\rceil$ slices, $\Delta={Delta:g}$")
ax.annotate(r"$u\,\sum_q|w_q|$", (20, U * math.exp(20) * 0.5), xytext=(-40, 6),
            textcoords="offset points", color=MUTED, fontsize=7)
ax.set_xlabel(r"$g\,t$")
ax.set_ylabel("max. entry error")
ax.set_ylim(1e-17, 1e6)
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "ch11_slicing.pdf")
fig.savefig(OUT / "ch11_slicing.png", dpi=200)
plt.close(fig)
print("Fig 11.1 done")

# ---------------------------------------------------------------- Fig 11.2
D3 = np.array([0.0, 1.0, 2.3])
V0 = np.array([[0, 0.2, 0], [0.2, 0, 0.1], [0, 0.1, 0]], float)
X3 = np.zeros((3, 3))
X3[0, 1] = X3[1, 0] = 0.5
Om3 = 0.9
Vs, ws = [V0, X3, X3], [0.0, Om3, -Om3]
Gam = sum(np.abs(Vm).sum(axis=0).max() for Vm in Vs)
K = 40
nk = 2 * K + 1
ks = np.arange(-K, K + 1)
HF = (np.kron(np.diag(D3), np.eye(nk)) - np.kron(np.eye(3), np.diag(ks * Om3))
      + np.kron(V0, np.eye(nk)) + np.kron(X3, np.eye(nk, k=-1))
      + np.kron(X3, np.eye(nk, k=1)))
fig, ax = plt.subplots(figsize=(4.0, 2.6))
Qs = [1, 3, 5, 7]
for colour, t in zip(INK, [0.25, 0.5, 1.0]):
    UF = sl.expm(-1j * t * HF)
    ref = sum(np.exp(-1j * t * k * Om3) * UF[1 * nk + k + K, K] for k in ks)
    err = [abs(dyson_driven(D3, Vs, ws, t, 0, 1, Q) - ref) for Q in Qs]
    tail = [sum((t * Gam)**q / math.factorial(q) for q in range(Q + 1, 80)) for Q in Qs]
    ax.semilogy(Qs, err, "o", ms=3, mew=0, color=colour, label=f"$t={t:g}$")
    ax.semilogy(Qs, tail, lw=0.8, color=colour, ls=(0, (3, 3)))
ax.set_xlabel("largest order $Q$")
ax.set_ylabel(r"error in $\langle 1|U(t)|0\rangle$")
ax.legend(frameon=False, fontsize=7, loc="lower left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "ch11_driven.pdf")
fig.savefig(OUT / "ch11_driven.png", dpi=200)
plt.close(fig)
print("Fig 11.2 done")
