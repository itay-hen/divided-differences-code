"""Numerical checks of the identities proved in Chapter 3 (repeated nodes).

Run:  python3 code/ch03_checks.py

Reference values for confluent divided differences use Opitz's formula
(top-right entry of f(Z), Z bidiagonal), evaluated with mpmath at 50 digits;
Opitz's formula is valid for repeated nodes (Chapter 4).
"""
import math
from collections import Counter

import mpmath as mp

mp.mp.dps = 50


def dd_opitz(fm, xs):
    """f[x0..xq] = [f(Z)]_{0q}, Z = bidiag(xs; ones). fm acts on mp matrices."""
    q = len(xs) - 1
    Z = mp.matrix(q + 1, q + 1)
    for i, x in enumerate(xs):
        Z[i, i] = mp.mpf(x)
        if i < q:
            Z[i, i + 1] = 1
    return fm(Z)[0, q]


expm = mp.expm


def sin_m(Z):
    return (mp.expm(1j * Z) - mp.expm(-1j * Z)) / 2j


def dd_confluent_explicit(f_derivs, xs):
    """Explicit (residue) formula for confluent divided differences:
       sum_i 1/(nu_i-1)! d^{nu_i-1}/dz^{nu_i-1} [ f(z) / prod_{k!=i}(z-z_k)^{nu_k} ] at z_i.
    f_derivs(z, r) returns f^{(r)}(z)."""
    mult = Counter(xs)
    zs = list(mult)
    tot = mp.mpf(0)
    for zi in zs:
        nu = mult[zi]
        g = lambda z: mp.fprod((z - zk) ** (-mult[zk]) for zk in zs if zk != zi)
        # Leibniz: d^{n}[f g] = sum_r C(n,r) f^{(r)} g^{(n-r)}
        n = nu - 1
        s = mp.fsum(mp.binomial(n, r) * f_derivs(zi, r) * mp.diff(g, zi, n - r)
                    for r in range(n + 1))
        tot += s / mp.factorial(n)
    return tot


exp_d = lambda z, r: mp.exp(z)

# 1. limits: f[x,...,x] = f^(q)(x)/q!
x = mp.mpf("0.37")
r = abs(dd_opitz(expm, [x] * 5) - mp.exp(x) / 24)
print(f"[1] e^[x x x x x] = e^x/4!:                      {float(r):.1e}")

# 2. Hermite-Genocchi for q=2 (2D quadrature over the simplex) at confluent nodes
xs = [0.2, 0.2, -0.7]
hg = mp.quad(lambda s1: mp.quad(lambda s2: mp.exp(xs[0] + s1 * (xs[1] - xs[0]) + s2 * (xs[2] - xs[0])),
                                 [0, 1 - s1]), [0, 1])
print(f"[2] Hermite-Genocchi vs Opitz (q=2, confluent):  {float(abs(hg - dd_opitz(expm, xs))):.1e}")

# 3. explicit confluent formula vs Opitz
xs = [0.1, 0.1, 0.1, -0.5, -0.5, 0.9]
r = abs(dd_confluent_explicit(exp_d, xs) - dd_opitz(expm, xs))
print(f"[3] explicit confluent formula vs Opitz:         {float(r):.1e}")

# 4. limit of distinct nodes -> confluent value
eps = mp.mpf("1e-12")
xs_d = [0.1, 0.1 + eps, 0.1 + 2 * eps, -0.5, -0.5 + eps, 0.9]
r = abs(dd_opitz(expm, xs_d) - dd_opitz(expm, xs))
print(f"[4] distinct -> confluent limit (eps=1e-12):     {float(r):.1e}")

# 5. f[a,a,b] formula and e^[0,0,1] = e - 2
r = abs(dd_opitz(expm, [0, 0, 1]) - (mp.e - 2))
print(f"[5] e^[0,0,1] = e-2:                              {float(r):.1e}")

# 6. node derivative: d/dx_j f[T] = f[T, x_j]
T = [mp.mpf(v) for v in (0.3, -0.4, 0.8, 0.1)]
for j in (0, 2):
    def F(t, j=j):
        TT = list(T); TT[j] = t
        return dd_opitz(sin_m, TT)
    r = abs(mp.diff(F, T[j]) - dd_opitz(sin_m, T + [T[j]]))
    print(f"[6] d/dx_{j} sin[T] = sin[T, x_{j}]:                 {float(r):.1e}")

# 7. higher node derivative: d^2/dx_j^2 f[T] = 2! f[T, x_j, x_j]
F = lambda t: dd_opitz(expm, [t] + T[1:])
r = abs(mp.diff(F, T[0], 2) - 2 * dd_opitz(expm, T + [T[0], T[0]]))
print(f"[7] d^2/dx_0^2 e^[T] = 2 e^[T,x0,x0]:            {float(r):.1e}")

# 8. shift identity: sum_j f[T, x_j] = f'[T]   (f = sin, f' = cos)
cos_m = lambda Z: (mp.expm(1j * Z) + mp.expm(-1j * Z)) / 2
lhs = mp.fsum(dd_opitz(sin_m, T + [t]) for t in T)
r = abs(lhs - dd_opitz(cos_m, T))
print(f"[8] sum_j sin[T,x_j] = cos[T]:                   {float(r):.1e}")
# ... and for exp: sum_j e^[T,x_j] = e^[T]
r = abs(mp.fsum(dd_opitz(expm, T + [t]) for t in T) - dd_opitz(expm, T))
print(f"    sum_j e^[T,x_j] = e^[T]:                     {float(r):.1e}")

# 9. scaling identity: sum_j x_j f[T,x_j] = (x f')[T] - q f[T]
q = len(T) - 1
lhs = mp.fsum(t * dd_opitz(sin_m, T + [t]) for t in T)
Zcos = lambda Z: Z * cos_m(Z)
r = abs(lhs - (dd_opitz(Zcos, T) - q * dd_opitz(sin_m, T)))
print(f"[9] scaling identity:                            {float(r):.1e}")

# 10. beta derivative of e^{-beta[E]}:
#     d/dbeta e^{-beta[E0..Eq]} = -(E_q e^{-beta[E0..Eq]} + e^{-beta[E0..E_{q-1}]})
E = [mp.mpf(v) for v in (0.5, -1.2, 0.5, 2.0, -0.3)]
Fb = lambda b, EE: dd_opitz(lambda Z: mp.expm(-b * Z), EE)
b0 = mp.mpf("1.3")
lhs = mp.diff(lambda b: Fb(b, E), b0)
rhs = -(E[-1] * Fb(b0, E) + Fb(b0, E[:-1]))
print(f"[10] beta-derivative identity:                   {float(abs(lhs - rhs)):.1e}")

# 11. Hermite table (nodes sorted so equal nodes are adjacent)
def hermite_table(xs, derivs):
    """Newton coefficients for possibly repeated nodes (equal nodes adjacent).
    derivs(x, k) = f^{(k)}(x)."""
    q = len(xs) - 1
    c = [derivs(x, 0) for x in xs]
    for k in range(1, q + 1):
        for j in range(q, k - 1, -1):
            if xs[j] == xs[j - k]:
                c[j] = derivs(xs[j], k) / math.factorial(k)
            else:
                c[j] = (c[j] - c[j - 1]) / (xs[j] - xs[j - k])
    return c

xs = [0.1, 0.1, 0.1, 0.4, 0.4, 0.9]
c = hermite_table(xs, lambda x, k: math.exp(x))
ref = float(dd_opitz(expm, xs))
print(f"[11] Hermite table (double) vs reference:        {abs(c[-1] - ref) / ref:.1e} (rel)")

# 12. perturbing confluent nodes to distinct ones in double precision
xs_p = [0.1, 0.1 + 1e-8, 0.1 + 2e-8, 0.4, 0.4 + 1e-8, 0.9]
from itertools import count
def table(xs, fs):
    c = list(fs); q = len(xs) - 1
    for k in range(1, q + 1):
        for j in range(q, k - 1, -1):
            c[j] = (c[j] - c[j - 1]) / (xs[j] - xs[j - k])
    return c
cp = table(xs_p, [math.exp(x) for x in xs_p])
print(f"[12] perturbed-distinct table (double):  {cp[-1]:.4e}   confluent ref {ref:.4e}")
