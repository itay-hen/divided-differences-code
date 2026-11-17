"""Numerical checks of the representations proved in Chapter 4.

Run:  python3 code/ch04_checks.py
"""
import itertools
import math

import mpmath as mp
import numpy as np

mp.mp.dps = 40
rng = np.random.default_rng(4)


def dd(f, xs):
    """Reference divided difference at distinct nodes (explicit formula)."""
    xs = [mp.mpmathify(x) for x in xs]
    return mp.fsum(f(xj) / mp.fprod(xj - xk for k, xk in enumerate(xs) if k != j)
                   for j, xj in enumerate(xs))


def opitz(fm, xs):
    q = len(xs) - 1
    Z = mp.matrix(q + 1, q + 1)
    for i, x in enumerate(xs):
        Z[i, i] = mp.mpmathify(x)
        if i < q:
            Z[i, i + 1] = 1
    return fm(Z), Z


xs = [-0.9, -0.2, 0.35, 0.6, 1.1]
q = len(xs) - 1

# 1. Dirichlet moments: int_Sigma s^alpha ds = alpha! / (q+|alpha|)!  (q=2, alpha=(2,1,3))
alpha = (2, 1, 3)
num = mp.quad(lambda s1: mp.quad(lambda s2: (1 - s1 - s2) ** alpha[0] * s1 ** alpha[1] * s2 ** alpha[2],
                                 [0, 1 - s1]), [0, 1])
exact = mp.fprod(mp.factorial(a) for a in alpha) / mp.factorial(2 + sum(alpha))
print(f"[1] Dirichlet moment formula:                  {float(abs(num - exact)):.1e}")

# 2. Peano kernel / B-spline: f[x] = (1/q!) int M(t) f^(q)(t) dt, int M = 1
def M(t):
    return q * dd(lambda x: (x - t) ** (q - 1) if x > t else mp.mpf(0), xs)
brk = sorted(xs)
intM = mp.quad(M, brk)
f, fq = mp.sin, mp.sin            # q = 4: sin'''' = sin
peano = mp.quad(lambda t: M(t) * fq(t), brk) / mp.factorial(q)
print(f"[2] int M = 1:                                 {float(abs(intM - 1)):.1e}")
print(f"    Peano/B-spline representation:             {float(abs(peano - dd(f, xs))):.1e}")
print(f"    min of M on a grid (should be >= 0):       {min(float(M(t)) for t in np.linspace(brk[0], brk[-1], 41)):.2e}")

# 3. B-spline is the density of sum U_j x_j, U ~ Dirichlet(1,...,1): compare means
mean_M = mp.quad(lambda t: t * M(t), brk)
print(f"[3] mean of M = mean of nodes:                 {float(abs(mean_M - sum(xs) / (q + 1))):.1e}")

# 4. Cauchy integral with trapezoidal rule on a circle (confluent nodes allowed)
xs_c = [0.2, 0.2, -0.5, 0.7, 0.7, 0.7]
c, R, N = 0.1, 2.0, 64
zs = [c + R * mp.exp(2j * mp.pi * k / N) for k in range(N)]
cau = mp.fsum(mp.exp(z) / mp.fprod(z - x for x in xs_c) * (z - c) for z in zs) / N
ref = opitz(mp.expm, xs_c)[0][0, len(xs_c) - 1]
print(f"[4] Cauchy (trapezoid, N={N}) vs Opitz:         {float(abs(cau - ref)):.1e}")

# 5. Resolvent of the Opitz matrix: [(z-Z)^{-1}]_{ij} = prod_{k=i..j} 1/(z-x_k)
_, Z = opitz(mp.expm, xs)
z0 = mp.mpf(3.3)
Rz = (z0 * mp.eye(q + 1) - Z) ** -1
err = max(abs(Rz[i, j] - mp.fprod(1 / (z0 - xs[k]) for k in range(i, j + 1)))
          for i in range(q + 1) for j in range(i, q + 1))
print(f"[5] resolvent of Opitz matrix:                 {float(err):.1e}")

# 6. All entries of f(Z) are divided differences of consecutive nodes
FZ, _ = opitz(mp.expm, xs)
err = max(abs(FZ[i, j] - dd(mp.exp, xs[i:j + 1])) for i in range(q + 1) for j in range(i, q + 1))
print(f"[6] [exp(Z)]_ij = e^[x_i..x_j]:                {float(err):.1e}")

# 7. Commutation Z f(Z) = f(Z) Z  <=> recursion
print(f"[7] ||Z f(Z) - f(Z) Z||:                       {float(mp.mnorm(Z * FZ - FZ * Z, 1)):.1e}")

# 8. Semigroup: e^{(s+t)[x]} = sum_k e^{s[x0..xk]} e^{t[xk..xq]}
s, t = mp.mpf("0.7"), mp.mpf("1.9")
lhs = dd(lambda x: mp.exp((s + t) * x), xs)
rhs = mp.fsum(dd(lambda x: mp.exp(s * x), xs[:k + 1]) * dd(lambda x: mp.exp(t * x), xs[k:]) for k in range(q + 1))
print(f"[8] semigroup identity:                        {float(abs(lhs - rhs)):.1e}")

# 9. Scaled Opitz matrix: theta on the superdiagonal gives theta^{j-i} f[x_i..x_j]
theta = mp.mpf(0.37)
Zt = Z.copy()
for i in range(q):
    Zt[i, i + 1] = theta
err = abs(mp.expm(Zt)[0, q] - theta**q * dd(mp.exp, xs))
print(f"[9] scaled Opitz matrix:                       {float(err):.1e}")

# 10. Real-part domination |e^[z]| <= e^[Re z] for complex nodes
worst = 0
for _ in range(200):
    zz = rng.normal(size=5) * 2 + 1j * rng.normal(size=5) * 3
    a = abs(dd(mp.exp, list(zz)))
    b = dd(mp.exp, list(zz.real))
    worst = max(worst, float(a / b))
print(f"[10] max |e^[z]| / e^[Re z] over 200 trials:   {worst:.4f}  (<= 1)")

# 11. Oscillatory bound |e^{-it[E]}| <= t^q / q!
worst = 0
for _ in range(200):
    E = rng.normal(size=6) * 3
    tt = rng.uniform(0.1, 5)
    val = abs(dd(lambda x: mp.exp(-1j * tt * x), list(E)))
    worst = max(worst, float(val / (tt**5 / math.factorial(5))))
print(f"[11] max |e^(-it[E])| q!/t^q over 200 trials:  {worst:.4f}  (<= 1)")

# 12. Jensen lower bound e^[x] >= e^{mean x}/q!
worst = min(float(dd(mp.exp, list(x)) * mp.factorial(5) / mp.exp(np.mean(x)))
            for x in rng.normal(size=(200, 6)) * 4)
print(f"[12] min e^[x] q! / e^(mean x) over 200 trials: {worst:.4f}  (>= 1)")

# 13. Bernstein mixture for 1/x:  (1/x)[x] = int_0^inf (e^{-t x})[x] dt,  sign (-1)^q
xp = [0.5, 1.2, 2.0, 3.1, 4.4]
lhs = dd(lambda x: 1 / x, xp)
rhs = mp.quad(lambda tt: dd(lambda x: mp.exp(-tt * x), xp), [0, 1, mp.inf])
print(f"[13] Bernstein mixture for 1/x:                {float(abs(lhs - rhs)):.1e}   sign (-1)^q*val>0: {(-1)**4*lhs>0}")

# 14. Contour bound |f[x]| <= R max|f| / (R-r)^{q+1} for exp, disk centred at c
xs_r = [0.1, -0.3, 0.25, 0.0, -0.15]
c, r = 0.0, 0.3
best = min((Rr * math.exp(c + Rr)) / (Rr - r) ** 5 for Rr in np.linspace(0.31, 10, 2000))
print(f"[14] e^[x] = {float(dd(mp.exp, xs_r)):.4e} <= contour bound {best:.4e}")


def mark(ok):
    return "" if ok else "FAIL "


# 15. Chain rule (Theorem 4.x) at q = 4, against the explicit formula
f_, g_ = (lambda y: mp.sin(y) + y**3), (lambda t: mp.exp(t) + t / 3)
ts = [0.1, 0.35, 0.7, 1.2, 1.9]
xs_c = [g_(mp.mpf(t)) for t in ts]
chain = 0
for k in range(1, 5):
    for mid in itertools.combinations(range(1, 4), k - 1):
        idx = (0,) + mid + (4,)
        term = dd(f_, [xs_c[i] for i in idx])
        for a, b in zip(idx, idx[1:]):
            term *= dd(g_, ts[a:b + 1])
        chain += term
err = abs(chain / dd(lambda t: f_(g_(t)), ts) - 1)
print(f"{mark(err < 1e-30)}[15] chain rule, q=4:                          {float(err):.1e}")

# 16. Inverse function, second order
t3 = [mp.mpf(v) for v in (0.2, 0.5, 1.1)]
ginv = lambda y: mp.findroot(lambda t: g_(t) - y, 0.5)
lhs = dd(ginv, [g_(t) for t in t3])
rhs = -dd(g_, t3) / (dd(g_, t3[:2]) * dd(g_, t3[1:]) * dd(g_, [t3[0], t3[2]]))
print(f"{mark(abs(lhs / rhs - 1) < 1e-30)}[16] inverse function, second order:            {float(abs(lhs / rhs - 1)):.1e}")

# 17. Chebyshev interpolation of the Runge function: bound of Theorem 4.x and rate rho^-(q+1)
runge = lambda x: 1 / (1 + 25 * x**2)
grid = np.linspace(-1, 1, 20001)


def interp_err(nodes):
    w = np.array([1 / np.prod(nodes[j] - np.delete(nodes, j)) for j in range(len(nodes))])
    d = grid[:, None] - nodes[None, :]
    with np.errstate(all="ignore"):
        p = (w * runge(nodes) / d).sum(1) / (w / d).sum(1)
    hit = np.isclose(d, 0).any(1)
    p[hit] = runge(grid[hit])
    return np.max(np.abs(p - runge(grid)))


rho_r = (1 + math.sqrt(26)) / 5
ok = True
for q in (20, 40, 80):
    ch = np.cos((2 * np.arange(q + 1) + 1) * np.pi / (2 * q + 2))
    e = interp_err(ch)
    ok &= abs(e / rho_r ** -(q + 1) - 1) < 0.03
    # the theorem's bound on a slightly smaller ellipse, where f is analytic
    rr = rho_r * 0.999
    a = (rr + 1 / rr) / 2
    zz = [(rr * np.exp(1j * th) + np.exp(-1j * th) / rr) / 2 for th in np.linspace(0, 2 * np.pi, 4001)]
    M = max(abs(1 / (1 + 25 * z**2)) for z in zz)
    ok &= e <= 4 * a * M / ((a - 1) * (rr ** (q + 1) - rr ** -(q + 1)))
    print(f"     q={q}: Chebyshev error {e:.2e}, rho^-(q+1) = {rho_r ** -(q + 1):.2e}")
eq20, eq40 = interp_err(np.linspace(-1, 1, 21)), interp_err(np.linspace(-1, 1, 41))
print(f"     equispaced: {eq20:.1f} (q=20), {eq40:.1e} (q=40)")
ok &= eq20 > 50 and eq40 > 1e5
print(f"{mark(ok)}[17] Chebyshev rate and bound; Runge phenomenon")


def lebesgue(nodes):
    w = np.array([1 / np.prod(nodes[j] - np.delete(nodes, j)) for j in range(len(nodes))])
    d = grid[:, None] - nodes[None, :]
    with np.errstate(all="ignore"):
        L = np.abs(w / d).sum(1) / np.abs((w / d).sum(1))
    return np.nanmax(L[np.isfinite(L)])


lc = lebesgue(np.cos((2 * np.arange(21) + 1) * np.pi / 42))
le = lebesgue(np.linspace(-1, 1, 21))
ok = lc <= 2 / math.pi * math.log(21) + 1 and 1.0e4 < le < 1.2e4
print(f"{mark(ok)}[18] Lebesgue constants q=20: Chebyshev {lc:.2f}, equispaced {le:.3g}")

# 19. Higher-order convexity examples
xp3 = lambda x: max(x, 0) ** 3
ax3 = lambda x: abs(x) ** 3
m3 = min(dd(xp3, sorted(rng.uniform(-1, 1, 4))) for _ in range(2000))
m4 = min(dd(ax3, sorted(rng.uniform(-1, 1, 5))) for _ in range(2000))
m3a = min(dd(ax3, sorted(rng.uniform(-1, 1, 4))) for _ in range(2000))
ok = m3 >= -1e-30 and m4 >= -1e-30 and m3a < -0.1
print(f"{mark(ok)}[19] x_+^3 convex of order 3, |x|^3 of order 4 but not 3: "
      f"{float(m3):.1e}, {float(m4):.1e}, {float(m3a):.2f}")
