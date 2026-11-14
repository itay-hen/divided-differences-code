"""Numerical checks of the identities proved in Chapter 2.

Run:  python3 code/ch02_checks.py
"""
import itertools
import math

import mpmath as mp
import numpy as np

mp.mp.dps = 60
rng = np.random.default_rng(2)


def dd(f, xs):
    """Reference divided difference (explicit formula, 60 digits)."""
    xs = [mp.mpf(x) for x in xs]
    tot = mp.mpf(0)
    for j, xj in enumerate(xs):
        den = mp.fprod(xj - xk for k, xk in enumerate(xs) if k != j)
        tot += f(xj) / den
    return tot


def h(k, xs):
    """Complete homogeneous symmetric polynomial h_k(xs)."""
    if k < 0:
        return mp.mpf(0)
    return mp.fsum(mp.fprod(c) for c in
                   itertools.combinations_with_replacement([mp.mpf(x) for x in xs], k))


def table(xs, fs):
    """Divided-difference table; returns the Newton coefficients
    f[x0], f[x0,x1], ..., f[x0..xq] (in place, O(q^2))."""
    c = list(fs)
    q = len(xs) - 1
    for k in range(1, q + 1):
        for j in range(q, k - 1, -1):
            c[j] = (c[j] - c[j - 1]) / (xs[j] - xs[j - k])
    return c


def newton_eval(c, xs, t):
    """Evaluate the Newton form by nested multiplication (O(q))."""
    p = c[-1]
    for k in range(len(c) - 2, -1, -1):
        p = p * (t - xs[k]) + c[k]
    return p


def neville(xs, fs, t):
    """Neville--Aitken evaluation of the interpolant at t."""
    P = list(fs)
    q = len(xs) - 1
    for k in range(1, q + 1):
        for i in range(q - k + 1):
            j = i + k
            P[i] = ((t - xs[j]) * P[i] - (t - xs[i]) * P[i + 1]) / (xs[i] - xs[j])
    return P[0]


xs = list(rng.uniform(-1, 1, 6))
q = len(xs) - 1
f, g = mp.sin, (lambda x: mp.exp(x) / (2 + x))

# 1. recursion
r = abs(dd(f, xs) - (dd(f, xs[:-1]) - dd(f, xs[1:])) / (xs[0] - xs[-1]))
print(f"[1] recursion residual:                 {float(r):.1e}")

# 2. symmetry
perm = list(rng.permutation(xs))
print(f"[2] symmetry residual:                  {float(abs(dd(f, xs) - dd(f, perm))):.1e}")

# 3. Leibniz rule
lhs = dd(lambda x: f(x) * g(x), xs)
rhs = mp.fsum(dd(f, xs[: k + 1]) * dd(g, xs[k:]) for k in range(q + 1))
print(f"[3] Leibniz residual:                   {float(abs(lhs - rhs)):.1e}")

# 4. monomials -> complete homogeneous symmetric polynomials
r = max(abs(dd(lambda x: x**m, xs) - h(m - q, xs)) for m in range(0, 12))
print(f"[4] x^m[x0..xq] = h_(m-q) residual:     {float(r):.1e}")

# 5. h-recursion  h_k(x0..x_{q-1}) - h_k(x1..xq) = (x0-xq) h_{k-1}(x0..xq)
r = max(abs(h(k, xs[:-1]) - h(k, xs[1:]) - (xs[0] - xs[-1]) * h(k - 1, xs)) for k in range(6))
print(f"[5] h recursion residual:               {float(r):.1e}")

# 6. exponential series  e^[x] = sum_k h_k(x)/(k+q)!
ser = mp.fsum(h(k, xs) / mp.factorial(k + q) for k in range(25))
print(f"[6] exp series residual (25 terms):     {float(abs(ser - dd(mp.exp, xs))):.1e}")

# 7. affine change of variable  g(x)=f(ax+b): g[x] = a^q f[ax+b]
a, b = 1.7, -0.3
lhs = dd(lambda x: f(a * x + b), xs)
rhs = a**q * dd(f, [a * x + b for x in xs])
print(f"[7] affine change residual:             {float(abs(lhs - rhs)):.1e}")

# 8. mean value theorem: f[x]*q! lies in range of f^(q) on hull (f = sin, q = 5 -> cos)
val = float(dd(f, xs) * math.factorial(q))
grid = np.cos(np.linspace(min(xs), max(xs), 2001))
print(f"[8] MVT: q! f[x] = {val:+.4f} in [{grid.min():+.4f}, {grid.max():+.4f}]")

# 9. table, Newton evaluation, Neville agree with reference
fs = [float(f(x)) for x in xs]
c = table(xs, fs)
r = max(abs(c[k] - float(dd(f, xs[: k + 1]))) for k in range(q + 1))
t = 0.123
r2 = abs(newton_eval(c, xs, t) - neville(xs, fs, t))
print(f"[9] table vs reference: {r:.1e};  Newton vs Neville at t: {r2:.1e}")

# 10. worked example in the text: x^3 at 0,1,2,3
print("[10] table for x^3 at 0,1,2,3:", table([0, 1, 2, 3], [0, 1, 8, 27]))

# 11. 1/x and 1/(c-x)
r = abs(dd(lambda x: 1 / x, [1, 2, 3.5, 5]) - (-1) ** 3 / (1 * 2 * 3.5 * 5))
print(f"[11] 1/x formula residual:              {float(r):.1e}")

# 12. forward differences on equispaced nodes
x0, hh = 0.2, 0.3
nodes = [x0 + j * hh for j in range(q + 1)]
fd = mp.fsum((-1) ** (q - j) * mp.binomial(q, j) * f(nodes[j]) for j in range(q + 1))
r = abs(dd(f, nodes) - fd / (mp.factorial(q) * hh**q))
print(f"[12] forward-difference residual:       {float(r):.1e}")

# 13. cancellation preview: exp at clustered nodes via the double-precision table
xs_c = [0.5 + 1e-3 * j for j in range(8)]
c = table(xs_c, [math.exp(x) for x in xs_c])
ref = float(dd(mp.exp, xs_c))
print(f"[13] clustered nodes, q=7: table {c[-1]:.6e}  ref {ref:.6e}  rel.err {abs(c[-1]-ref)/ref:.1e}")
