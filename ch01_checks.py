"""Numerical checks of the identities stated in Chapter 1.

Run:  python3 code/ch01_checks.py
Every check prints the maximum discrepancy; all should be at round-off level.
"""
import itertools

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

mp.mp.dps = 50
rng = np.random.default_rng(1)


def dd(f, xs):
    """Divided difference f[x_0,...,x_q] for distinct nodes (explicit formula,
    evaluated in 50-digit arithmetic so it can serve as a reference)."""
    xs = [mp.mpf(x) for x in xs]
    tot = mp.mpf(0)
    for j, xj in enumerate(xs):
        den = mp.mpf(1)
        for k, xk in enumerate(xs):
            if k != j:
                den *= xj - xk
        tot += f(xj) / den
    return float(tot)


def dd_opitz(f_mat, xs):
    """Divided difference via Opitz's formula: [f(Z)]_{0,q} with Z bidiagonal."""
    q = len(xs) - 1
    Z = np.diag(xs) + np.diag(np.ones(q), 1)
    return f_mat(Z)[0, q]


# ---------------------------------------------------------------------------
# 1. Newton form reproduces the interpolant (spot check with f = sin)
xs = np.sort(rng.uniform(-1, 1, 6))
coeffs = [dd(mp.sin, xs[: k + 1]) for k in range(len(xs))]
t = np.linspace(-1, 1, 7)
p = np.zeros_like(t)
w = np.ones_like(t)
for k, c in enumerate(coeffs):
    p += c * w
    w *= t - xs[k]
err_interp = max(abs(p[i] - np.sin(t[i])) for i in range(len(t)))
# error formula: f(t) - p(t) = f[x_0..x_q, t] * omega(t)
err_formula = max(
    abs(np.sin(ti) - pi - dd(mp.sin, list(xs) + [ti]) * np.prod(ti - xs))
    for ti, pi in zip(t, p)
)
print(f"[1] Newton error formula residual:        {err_formula:.2e}"
      f"   (interp. error itself {err_interp:.1e})")

# ---------------------------------------------------------------------------
# 2. Opitz agrees with the explicit formula for exp
xs = rng.uniform(-2, 2, 5)
d1 = dd(mp.exp, xs)
d2 = dd_opitz(expm, xs)
print(f"[2] Opitz vs explicit (exp, q=4):         {abs(d1 - d2):.2e}")

# ---------------------------------------------------------------------------
# 3. Kubo integral:  int_0^beta e^{-(beta-tau)E_n - tau E_m} dtau = -F[E_n,E_m]
#    with F(x) = e^{-beta x}
beta, En, Em = 1.7, 0.3, -0.8
lhs = quad(lambda tau: np.exp(-(beta - tau) * En - tau * Em), 0, beta)[0]
F = lambda x: mp.exp(-beta * x)
rhs = -dd(F, [En, Em])
print(f"[3] Kubo integral vs divided difference:  {abs(lhs - rhs):.2e}")

# ---------------------------------------------------------------------------
# 4. Partition function as sum over closed walks:
#    Z = sum_q sum_{z0..z_{q-1}} V_{z0 z1}...V_{z_{q-1} z0} F[E_z0,...,E_z_{q-1},E_z0]
#    (diagonal part D = diag(E), off-diagonal part V, F(x) = e^{-beta x}).
#    Repeated energies are handled with the Opitz formula.
n, beta = 3, 1.3
E = rng.uniform(-1, 1, n)
V = rng.normal(size=(n, n)) * 0.25
V = (V + V.T) / 2
np.fill_diagonal(V, 0.0)
H = np.diag(E) + V
Z_exact = np.trace(expm(-beta * H))

Fm = lambda M: expm(-beta * M)
Z_series, qmax = 0.0, 10
for q in range(qmax + 1):
    term = 0.0
    for walk in itertools.product(range(n), repeat=q):
        if q == 0:
            term = sum(np.exp(-beta * E))
            break
        zs = list(walk) + [walk[0]]
        amp = np.prod([V[zs[i], zs[i + 1]] for i in range(q)])
        if amp == 0.0:
            continue
        term += amp * dd_opitz(Fm, E[zs])
    Z_series += term
print(f"[4] Closed-walk series vs Tr e^(-bH):     {abs(Z_series - Z_exact):.2e}"
      f"   (q <= {qmax})")

# ---------------------------------------------------------------------------
# 5. Daleckii--Krein expansion to second order for f = exp(-beta x):
#    [f(D+eps V)]_{nm} = f(E_n) delta_nm + eps V_nm F[E_n,E_m]
#                        + eps^2 sum_k V_nk V_km F[E_n,E_k,E_m] + O(eps^3)
eps = 1e-3
lhs = expm(-beta * (np.diag(E) + eps * V))
rhs = np.diag(np.exp(-beta * E)).astype(float)
for a in range(n):
    for b in range(n):
        if a != b:
            rhs[a, b] += eps * V[a, b] * dd(F, [E[a], E[b]])
        for k in range(n):
            if V[a, k] * V[k, b] != 0:
                rhs[a, b] += eps**2 * V[a, k] * V[k, b] * dd_opitz(Fm, E[[a, k, b]])
print(f"[5] Daleckii-Krein 2nd order residual:    {np.abs(lhs - rhs).max():.2e}"
      f"   (expected O(eps^3) = {eps**3:.0e})")

# ---------------------------------------------------------------------------
# 6. Rayleigh--Schroedinger limit:  -(1/beta) ln Z -> E_0 + sum_k |V_0k|^2/(E_0-E_k)
#    at second order in V, as beta -> infinity (nondegenerate ground state).
E = np.array([0.0, 1.0, 1.7, 2.4])
V = rng.normal(size=(4, 4)); V = (V + V.T) / 2; np.fill_diagonal(V, 0)
eps = 1e-3
rs2 = sum(V[0, k] ** 2 / (E[0] - E[k]) for k in range(1, 4))
for beta in (5.0, 20.0, 80.0):
    # second-order coefficient of the free energy, via finite differences:
    Fe = lambda s: -np.log(np.trace(expm(-beta * (np.diag(E) + s * V)))) / beta
    c2 = (Fe(eps) - 2 * Fe(0) + Fe(-eps)) / (2 * eps**2)
    print(f"[6] beta={beta:5.1f}: 2nd-order free energy {c2:+.6f}   RS value {rs2:+.6f}")
