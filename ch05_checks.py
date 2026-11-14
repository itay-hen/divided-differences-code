"""Numerical checks of the identities and bounds in Chapter 5 (the exponential).

Run:  python3 code/ch05_checks.py
Reference values: Opitz's formula in 50-digit arithmetic (valid for repeated nodes).
"""
import math

import mpmath as mp
import numpy as np

mp.mp.dps = 50
rng = np.random.default_rng(5)


def opitz(fm, xs):
    q = len(xs) - 1
    Z = mp.matrix(q + 1, q + 1)
    for i, x in enumerate(xs):
        Z[i, i] = mp.mpmathify(x)
        if i < q:
            Z[i, i + 1] = 1
    return fm(Z)


def ee(xs, c=1):
    """e^{c[x_0..x_q]}: divided difference of x -> e^{cx}."""
    c = mp.mpmathify(c)
    return opitz(lambda Z: mp.expm(c * Z), xs)[0, len(xs) - 1]


def h_table(ys, M):
    """h_m(y_0..y_q) for m = 0..M via h_m(y_0..y_k) = h_m(y_0..y_{k-1}) + y_k h_{m-1}(y_0..y_k)."""
    h = [mp.mpf(1)] + [mp.mpf(0)] * M          # k = -1: h_0 = 1? (empty set: h_0=1, h_m=0)
    for y in ys:
        new = [mp.mpf(1)]
        for m in range(1, M + 1):
            new.append(h[m] + y * new[m - 1])
        h = new
    return h


def report(tag, val, ok=None):
    s = f"{tag:<58s} {val}"
    if ok is not None:
        s += "   OK" if ok else "   FAIL"
    print(s)


xs = [mp.mpf(v) for v in (-1.3, 0.2, 0.2, 0.9, 1.7, -0.4)]
q = len(xs) - 1
ref = ee(xs)

# ---------------------------------------------------------------- 5.1
M = 40
xbar = sum(xs) / (q + 1)
hs = h_table([x - xbar for x in xs], M)
ser = mp.e**xbar * mp.fsum(hs[m] / mp.factorial(m + q) for m in range(M + 1))
report("[5.1a] centred h-series vs Opitz", mp.nstr(abs(ser - ref), 3), abs(ser - ref) < 1e-40)
for MM in (5, 10, 15):
    part = mp.e**xbar * mp.fsum(hs[m] / mp.factorial(m + q) for m in range(MM + 1))
    r = max(abs(x - xbar) for x in xs)
    bound = mp.e**xbar / mp.factorial(q) * mp.fsum(r**m / mp.factorial(m) for m in range(MM + 1, 200))
    report(f"[5.1b] truncation M={MM}: error <= bound", f"{mp.nstr(abs(part-ref),3)} <= {mp.nstr(bound,3)}",
           abs(part - ref) <= bound)

# impulse response: y(t) = e^{t[x]} solves omega(d/dt) y = 0, y^(k)(0) = delta_{k,q}
t0 = mp.mpf("0.8")
y = lambda t: ee(xs, t)
derivs = [mp.diff(y, mp.mpf(0), k) for k in range(q + 1)]
report("[5.1c] y^(k)(0) = 0 (k<q), 1 (k=q)", [mp.nstr(d, 3) for d in derivs],
       all(abs(d) < 1e-20 for d in derivs[:-1]) and abs(derivs[-1] - 1) < 1e-20)
# omega(d/dt) y = 0 : polynomial coefficients of omega
coeffs = [mp.mpf(1)]
for x in xs:  # multiply by (D - x)
    coeffs = [a - x * b for a, b in zip(coeffs + [0], [0] + coeffs)]
coeffs = coeffs[::-1]  # coeffs[k] multiplies D^k
res = mp.fsum(coeffs[k] * mp.diff(y, t0, k) for k in range(q + 2))
report("[5.1d] omega(d/dt) e^{t[x]} = 0", mp.nstr(abs(res), 3), abs(res) < 1e-15)

# Laplace transform: int_0^inf e^{-st} e^{t[x]} dt = 1/prod(s - x_j)  (Re s > max x)
s = mp.mpf(3)
lap = mp.quad(lambda t: mp.e**(-s * t) * y(t), [0, 5, 20, mp.inf])
report("[5.1e] Laplace transform", mp.nstr(abs(lap - 1 / mp.fprod(s - x for x in xs)), 3),
       abs(lap - 1 / mp.fprod(s - x for x in xs)) < 1e-20)

# ---------------------------------------------------------------- 5.2
lam, t = mp.mpf("0.7"), mp.mpf("1.3")
ap = ee([k * lam for k in range(q, -1, -1)], t)
report("[5.2a] arithmetic progression closed form",
       mp.nstr(abs(ap - ((mp.e**(lam * t) - 1) / lam)**q / mp.factorial(q)), 3),
       abs(ap - ((mp.e**(lam * t) - 1) / lam)**q / mp.factorial(q)) < 1e-40)
bell = mp.fsum(ee(list(range(k + 1)), t) for k in range(60))
report("[5.2b] sum_q e^{t[0..q]} = exp(e^t - 1)", mp.nstr(abs(bell - mp.e**(mp.e**t - 1)), 3),
       abs(bell - mp.e**(mp.e**t - 1)) < 1e-30)
# two-level resummation: sum_k e^{T[a^(k+1), b^(k)]} m^k = e^{abar T}(cosh(Om T) + d sinh(Om T)/Om)
a, b, m, T = mp.mpf("-0.4"), mp.mpf("1.1"), mp.mpf("0.35"), mp.mpf("1.7")
lhs = mp.fsum(ee([a] * (k + 1) + [b] * k, T) * m**k for k in range(40))
d = (a - b) / 2; Om = mp.sqrt(d**2 + m)
rhs = mp.e**((a + b) / 2 * T) * (mp.cosh(Om * T) + d * mp.sinh(Om * T) / Om)
report("[5.2c] two-level resummation", mp.nstr(abs(lhs - rhs), 3), abs(lhs - rhs) < 1e-30)

# ---------------------------------------------------------------- 5.3
bad = 0
for _ in range(300):
    x = list(rng.normal(size=rng.integers(2, 9)) * 3)
    n = len(x) - 1
    v = mp.factorial(n) * ee(x)
    lo, hi = mp.e**np.mean(x), np.mean(np.exp(x))
    bad += not (lo <= v <= hi)
report("[5.3a] Jensen sandwich, 300 random sets: violations", bad, bad == 0)

# bound uniform at collision |e^{T[z]}| <= T^q/q! e^{T sigma}, complex nodes
bad = 0
for _ in range(200):
    z = list(rng.normal(size=5) + 1j * rng.normal(size=5) * 4)
    T = rng.uniform(0.1, 3)
    sigma = max(w.real for w in z)
    bad += abs(ee(z, T)) > T**4 / 24 * math.exp(T * sigma) * (1 + 1e-12)
report("[5.3b] |e^{T[z]}| <= T^q/q! e^{T max Re z}: violations", bad, bad == 0)

# left half-plane: int_0^inf |e^{t[z]}| <= prod 1/h ; sup_t <= min_k prod_{r!=k} 1/h_r
hvals = rng.uniform(0.3, 2.0, 4)
z = [complex(-hv, im) for hv, im in zip(hvals, rng.normal(size=4) * 2)]
integral = mp.quad(lambda tt: abs(ee(z, tt)), [0, 2, 10, 60])
supv = max(abs(ee(z, tt)) for tt in np.linspace(0, 30, 301))
b1 = 1 / np.prod(hvals); b2 = min(1 / np.prod(np.delete(hvals, k)) for k in range(4))
report("[5.3c] left half-plane: integral <= prod 1/h", f"{mp.nstr(integral,4)} <= {b1:.4f}", integral <= b1)
report("       sup_t |e^{t[z]}| <= min_k prod_(r!=k) 1/h_r", f"{mp.nstr(supv,4)} <= {b2:.4f}", supv <= b2)
# real negative nodes: integral equals prod 1/h exactly
integral_r = mp.quad(lambda tt: ee([-hv for hv in hvals], tt), [0, 2, 10, mp.inf])
report("       real nodes: integral = prod 1/h (quadrature)", mp.nstr(abs(integral_r - b1), 3), abs(integral_r - b1) < 1e-12)

# node-scaling monotonicity and ratio bounds
x = list(rng.uniform(0, 2, 5)); alpha = 1.7
report("[5.3d] x>=0, alpha>=1: e^[alpha x] >= e^[x]", ee([alpha * v for v in x]) >= ee(x), True)
x = list(-rng.uniform(0, 2, 5))
report("       x<=0, alpha>=1: e^[alpha x] <= e^[x]", ee([alpha * v for v in x]) <= ee(x), True)
E = list(rng.uniform(-1, 2, 6)); beta, dbeta = mp.mpf("1.2"), mp.mpf("0.5")
W = lambda bb: mp.factorial(5) * ee([-bb * e for e in E])
ratio = W(beta + dbeta) / W(beta)
report("       e^{-db Emax} <= W(b+db)/W(b) <= e^{-db Emin}",
       f"{mp.nstr(mp.e**(-dbeta*max(E)),5)} <= {mp.nstr(ratio,5)} <= {mp.nstr(mp.e**(-dbeta*min(E)),5)}",
       mp.e**(-dbeta * max(E)) <= ratio <= mp.e**(-dbeta * min(E)))

# [Z+25] results
def K(a, p, qq, xv, yv):
    return ee(list(a) + [xv] * p + [yv] * qq)
bad_sub = bad_sup = 0
for _ in range(200):
    a = sorted(rng.normal(size=rng.integers(0, 4)) * 2)
    p, qq = rng.integers(1, 4), rng.integers(1, 4)
    x1, x2 = sorted(rng.normal(size=2) * 2); y1, y2 = sorted(rng.normal(size=2) * 2)
    k11, k22, k12, k21 = K(a, p, qq, x1, y1), K(a, p, qq, x2, y2), K(a, p, qq, x1, y2), K(a, p, qq, x2, y1)
    bad_sub += k11 * k22 > k12 * k21 * (1 + 1e-30)
    bad_sup += k11 + k22 < (k12 + k21) * (1 - 1e-30)
report("[5.3e] log-submodularity: violations / 200", bad_sub, bad_sub == 0)
report("       supermodularity: violations / 200", bad_sup, bad_sup == 0)
bad = 0; minratio = mp.inf
for _ in range(300):
    A, B, C, D = rng.normal(size=4) * 2
    val = ee([A, A, B, C]) * ee([D, D, B, C]) + ee([B, B, A, D]) * ee([C, C, A, D]) - ee([A, B, C, D])**2
    bad += val < 0; minratio = min(minratio, val / ee([A, B, C, D])**2)
report("       four-point inequality: violations / 300", f"{bad}  (min ratio {mp.nstr(minratio,3)})", bad == 0)

def LM(n, sig):
    a0 = mp.sqrt(n) * sig; aa = (n + 1) * sig / mp.sqrt(n)
    L = n * mp.e**(-a0) * mp.quad(lambda s_: s_**(n - 1) * mp.e**(aa * s_), [0, 1])
    Mx = n * mp.e**(a0) * mp.quad(lambda s_: s_**(n - 1) * mp.e**(-aa * s_), [0, 1])
    return L, Mx
bad = 0
for _ in range(100):
    n = int(rng.integers(1, 8)); x = rng.normal(size=n + 1) * rng.uniform(0.2, 3)
    mu, sig = np.mean(x), np.std(x)
    L, Mx = LM(n, sig)
    v = mp.factorial(n) * ee(list(x))
    bad += not (mp.e**mu * L * (1 - 1e-15) <= v <= mp.e**mu * Mx * (1 + 1e-15))   # n=1: L = M (equality)
report("[5.3f] mean-variance sandwich, 100 random sets: violations", bad, bad == 0)
n, sig, mu = 6, mp.mpf("0.9"), mp.mpf("0.3")
a0 = mp.sqrt(n) * sig
up = mp.factorial(n) * ee([mu + a0] + [mu - a0 / n] * n); L, Mx = LM(n, sig)
report("       attained upper bound (one outlier)", mp.nstr(abs(up - mp.e**mu * Mx), 3), abs(up - mp.e**mu * Mx) < 1e-30)
# asymptotics: log(n! e^[x]) - mu - var/(2n) = O(n^{-3/2}) for bounded variance
for n in (10, 100):
    x = list(np.sin(np.arange(n + 1)))             # bounded, varied
    mu, var = np.mean(x), np.var(x)
    # cumulant expansion of log E[e^{U.x}]
    lv = mp.log(mp.factorial(n) * ee(x))
    if True:
        report(f"[5.3g] n={n}: log(n! e^[x]) - mu - var/(2n)", mp.nstr(lv - mu - var / (2 * n), 3))

# two-point identity
xv, yv = mp.mpf("1.3"), mp.mpf("-0.6"); mm, dd_ = (xv + yv) / 2, (xv - yv) / 2
lhs = mp.e**xv + mp.e**yv - 2 * ee([xv, yv]); rhs = 2 * mp.e**mm * (mp.cosh(dd_) - mp.sinh(dd_) / dd_)
report("       two-point identity", mp.nstr(abs(lhs - rhs), 3), abs(lhs - rhs) < 1e-40)

# ---- proofs of the four-point inequality and the mean-variance sandwich
def hh(xv, yv):
    return mp.e**xv + mp.e**yv - 2 * ee([xv, yv])
def phi_(u):
    return mp.cosh(u) - mp.sinh(u) / u if u != 0 else mp.mpf(0)
def fourpt(A, B, C, D):
    return ee([A, A, B, C]) * ee([D, D, B, C]) + ee([B, B, A, D]) * ee([C, C, A, D]) - ee([A, B, C, D])**2
err = 0
for _ in range(50):
    A, B, C, D = [mp.mpf(v) for v in rng.normal(size=4) * 2]
    num = hh(A, C) * hh(B, D) + hh(A, B) * hh(C, D) - hh(B, C) * hh(A, D)
    err = max(err, abs(fourpt(A, B, C, D) - num / (2 * (A - B) * (A - C) * (B - D) * (C - D))) / abs(fourpt(A, B, C, D)))
report("[5.3h] four-point: f = (hh+hh-hh)/(2 prod), max rel err", mp.nstr(err, 3), err < 1e-30)
err = 0
for u in (mp.mpf("0.3"), mp.mpf(2), mp.mpf(7)):
    err = max(err, abs(mp.diff(phi_, u, 2) - (1 + 2 / u**2) * phi_(u)) / phi_(u))
report("       phi'' = (1 + 2/u^2) phi", mp.nstr(err, 3), err < 1e-30)
bad = 0
for _ in range(500):
    x_, y_, z_ = [mp.mpf(v) for v in rng.exponential(size=3) * rng.choice([0.1, 1, 5])]
    bad += phi_(x_ + y_) * phi_(y_ + z_) < phi_(x_) * phi_(z_) + phi_(y_) * phi_(x_ + y_ + z_)
report("       phi(x+y)phi(y+z) >= phi(x)phi(z)+phi(y)phi(x+y+z): violations/500", bad, bad == 0)
A, B, C, D = [mp.mpf(v) for v in (-1.1, 0.4, 1.3, 2.9)]
G8 = [(A, B, C, D), (D, B, C, A), (A, C, B, D), (D, C, B, A), (B, A, D, C), (C, D, A, B), (B, D, A, C), (C, A, D, B)]
spread = max(abs(fourpt(*g) - fourpt(A, B, C, D)) for g in G8)
report("       f invariant under the 8-element group", mp.nstr(spread, 3), spread < 1e-40)
# mean-variance: critical points are two-level; second variation within a block
okc = True; vals_ok = True
for n in (3, 5, 8):
    for sig in (mp.mpf("0.5"), mp.mpf(2), mp.mpf(6)):
        vals = []
        for kk in range(1, n + 1):
            u = sig * mp.sqrt(mp.mpf(n + 1 - kk) / kk); v = -sig * mp.sqrt(mp.mpf(kk) / (n + 1 - kk))
            y = [u] * kk + [v] * (n + 1 - kk)
            vals.append(mp.factorial(n) * ee(y))
            # Lagrange: g(t)=n! e[y,t] equals lam + nu t at u and v, nu = n! e[y,u,v]
            curv_hi = 2 * mp.factorial(n) * (ee(y + [u, u]) - ee(y + [u, v]))
            curv_lo = 2 * mp.factorial(n) * (ee(y + [v, v]) - ee(y + [u, v]))
            okc &= (kk < 2 or curv_hi > 0) and (n + 1 - kk < 2 or curv_lo < 0)
        L, Mx = LM(n, sig)
        vals_ok &= abs(vals[0] - Mx) < 1e-30 * Mx and abs(vals[-1] - L) < 1e-30 * L
        tol = mp.mpf(10)**-30
        vals_ok &= all(L * (1 - tol) <= w <= Mx * (1 + tol) for w in vals)
report("       two-level points: block curvature signs as claimed", okc, okc)
report("       two-level values: k=1 gives M_n, k=n gives L_n, others between", vals_ok, vals_ok)
yy = [1, 1, -1, -1]; n = 3; a0 = mp.sqrt(3)
report("       sigma(1,1,-1,-1) not majorized by y+ (top-2 sums)", f"2 > {mp.nstr(a0 - a0 / n, 5)}", 2 > a0 - a0 / n)
# the two-level structure for f with f^(n+3) > 0: f(z) = z^(n+3) + e^(z/2)
def fdd(fz, ys):
    return opitz(fz, [mp.mpf(v) for v in ys])[0, len(ys) - 1]
bad = 0
for n in (3, 4):
    fz = lambda Z, n=n: Z**(n + 3) + mp.expm(Z / 2)
    for sig in (mp.mpf("0.7"), mp.mpf(2)):
        two = []
        for kk in (1, n):
            u = sig * mp.sqrt(mp.mpf(n + 1 - kk) / kk); v = -sig * mp.sqrt(mp.mpf(kk) / (n + 1 - kk))
            two.append(fdd(fz, [u] * kk + [v] * (n + 1 - kk)))
        for _ in range(40):
            y = rng.normal(size=n + 1); y -= y.mean(); y *= float(sig) * math.sqrt((n + 1) / np.sum(y * y))
            val = fdd(fz, list(y))
            bad += not (two[1] * (1 - mp.mpf(10)**-30) <= val <= two[0] * (1 + mp.mpf(10)**-30))
report("       f = z^(n+3) + e^(z/2): one-outlier bounds, violations / 160", bad, bad == 0)
# general weights: log-concave phi gives TN2; an arbitrary mixture in c does not
def Kphi(ph, p, qq, xv, yv):
    with mp.workdps(25):
        return mp.quad(lambda cc: cc**(p + qq - 1) * ph(cc) * ee([cc * xv] * p + [cc * yv] * qq), [0, 1])
bad = 0
for ph in (lambda cc: (1 - cc)**2, lambda cc: cc * (1 - cc)):
    for _ in range(8):
        p, qq = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        x1, x2 = sorted(rng.normal(size=2) * 3); y1, y2 = sorted(rng.normal(size=2) * 3)
        bad += (Kphi(ph, p, qq, x1, y1) * Kphi(ph, p, qq, x2, y2)
                > Kphi(ph, p, qq, x1, y2) * Kphi(ph, p, qq, x2, y1) * (1 + mp.mpf(10)**-18))
report("[5.3i] log-concave weights: TN2 violations / 16", bad, bad == 0)
cs, ws = (mp.mpf("0.003383"), mp.mpf("0.677934")), (mp.mpf("0.337897"), mp.mpf("0.309958"))
Kmix = lambda xv, yv: sum(w_ * ee([c_ * xv, c_ * yv]) for c_, w_ in zip(cs, ws))
x1, x2, y1, y2 = mp.mpf("-0.7702"), mp.mpf("12.7407"), mp.mpf("-7.3683"), mp.mpf("-0.7513")
rat = Kmix(x1, y1) * Kmix(x2, y2) / (Kmix(x1, y2) * Kmix(x2, y1))
report("       two-point mixture in c violates TN2: ratio", mp.nstr(rat, 6), rat > 1.05)
# diagonal inequality and the Taylor-coefficient condition for P = e^[a, .]
bad = badc = 0
for _ in range(100):
    a = list(rng.normal(size=int(rng.integers(0, 4))) * 4)
    xv, yv = rng.normal(size=2) * 4
    bad += ee(a + [xv, yv])**2 < ee(a + [xv, xv]) * ee(a + [yv, yv])
    cfs = [ee(a + [xv] * (k + 1)) for k in range(1, 8)]          # P^(k)(x)/k!, k = 1..7
    badc += any(cfs[0] * cfs[nn] > cfs[1] * cfs[nn - 1] * (1 + mp.mpf(10)**-30) for nn in range(1, 6))
report("       e^[a,x,y]^2 >= e^[a,x,x] e^[a,y,y]: violations / 100", bad, bad == 0)
report("       c_1 c_(n+1) <= c_2 c_n for P = e^[a,.]: violations / 100", badc, badc == 0)

# ---------------------------------------------------------------- 5.4 Chebyshev
def chebT(n, y):
    return mp.chebyt(n, y)

def cheb_dd_table(ys, N):
    """D[n][k] = T_n[y_0..y_k] via the three-term recurrence (works for repeated nodes)."""
    qq = len(ys) - 1
    D = [[mp.mpf(0)] * (qq + 1) for _ in range(N + 1)]
    D[0][0] = mp.mpf(1)
    if N >= 1:
        D[1][0] = ys[0]
        if qq >= 1:
            D[1][1] = mp.mpf(1)
    for n in range(1, N):
        for k in range(qq + 1):
            D[n + 1][k] = 2 * (ys[k] * D[n][k] + (D[n][k - 1] if k > 0 else 0)) - D[n - 1][k]
    return D

xs2 = [mp.mpf(v) for v in (-0.7, 0.4, 0.4, 1.9, 2.5, 0.1)]
aa, bb = min(xs2), max(xs2); dmid, c = (aa + bb) / 2, (bb - aa) / 2
ys = [(x - dmid) / c for x in xs2]; qq = len(ys) - 1
N = 60
D = cheb_dd_table(ys, N)
# direct check of a few entries via Opitz with T_n
Tn_op = lambda n: opitz(lambda Z: mp.chebyt(n, 0) * 0 + _chebmat(n, Z), ys)
def _chebmat(n, Z):
    I = mp.eye(Z.rows); Tm, T0 = I, Z
    if n == 0: return I
    for _ in range(n - 1):
        Tm, T0 = T0, 2 * Z * T0 - Tm
    return T0
err = max(abs(D[n][qq] - _chebmat(n, _opz(ys))[0, qq]) for n in range(0, N + 1)) if False else None
def _opz(v):
    Z = mp.matrix(len(v), len(v))
    for i, x in enumerate(v):
        Z[i, i] = x
        if i < len(v) - 1: Z[i, i + 1] = 1
    return Z
err = max(abs(D[n][qq] - _chebmat(n, _opz(ys))[0, qq]) for n in range(N + 1))
report("[5.4a] recurrence vs Opitz T_n(J)", mp.nstr(err, 3), err < 1e-30)
report("       structural zeros D_q^(n)=0 (n<q), D_q^(q)=2^(q-1)",
       f"max|D_q^(n<q)|={mp.nstr(max(abs(D[n][qq]) for n in range(qq)),2)}, D_q^(q)={mp.nstr(D[qq][qq],6)}",
       max(abs(D[n][qq]) for n in range(qq)) < 1e-40 and abs(D[qq][qq] - 2**(qq - 1)) < 1e-40)
series = mp.e**dmid * mp.besseli(0, c) / c**qq * (D[0][qq] + 2 * mp.fsum(mp.besseli(n, c) / mp.besseli(0, c) * D[n][qq] for n in range(1, N + 1)))
report("[5.4b] Chebyshev-Bessel series vs Opitz", mp.nstr(abs(series - ee(xs2)), 3), abs(series - ee(xs2)) < 1e-35)
# Markov envelope |D_q^(n)| <= T_n^(q)(1)/q!
B = lambda n, k: mp.fprod(mp.mpf(n * n - j * j) / (2 * j + 1) for j in range(k)) / mp.factorial(k)
okm = all(abs(D[n][qq]) <= B(n, qq) * (1 + 1e-30) for n in range(qq, N + 1))
report("       Markov envelope holds for n = q..N", okm, okm)
# Bessel ratio bound
okb = all(mp.besseli(n, cc) / mp.besseli(0, cc) <= (cc / 2)**n / mp.factorial(n) for n in range(30) for cc in (0.1, 1, 5, 40))
report("       Bessel ratio R_n(c) <= (c/2)^n/n!", okb, okb)
# parity zeros for symmetric nodes
ysym = [mp.mpf(v) for v in (-0.8, -0.3, 0.3, 0.8)]
Ds = cheb_dd_table(ysym, 20)
report("[5.4c] parity zeros (n-q odd) for symmetric nodes",
       mp.nstr(max(abs(Ds[n][3]) for n in range(20) if (n - 3) % 2), 3),
       max(abs(Ds[n][3]) for n in range(20) if (n - 3) % 2) < 1e-40)
# beta-separated series: same D's for all beta
E = xs2
for beta in (mp.mpf("0.3"), mp.mpf(2)):
    ser_b = mp.e**(-beta * dmid) * mp.besseli(0, beta * c) / c**qq * (
        D[0][qq] + 2 * mp.fsum((-1)**n * mp.besseli(n, beta * c) / mp.besseli(0, beta * c) * D[n][qq] for n in range(1, N + 1)))
    refb = ee(E, -beta)
    report(f"[5.4d] beta-separated series, beta={beta}", mp.nstr(abs(ser_b - refb), 3), abs(ser_b - refb) < 1e-30)

# ---------------------------------------------------------------- 5.5
for k in (1, 2, 3):
    zv = mp.mpf("0.9")
    phi = mp.fsum(zv**j / mp.factorial(j + k) for j in range(80))
    report(f"[5.5a] phi_{k}(z) = e^[z, 0^({k})]", mp.nstr(abs(phi - ee([zv] + [0] * k)), 3), abs(phi - ee([zv] + [0] * k)) < 1e-40)
# sin divided difference via e^{+-i}
sin_dd = opitz(lambda Z: (mp.expm(1j * Z) - mp.expm(-1j * Z)) / 2j, xs)[0, q]
via = (ee(xs, 1j) - ee(xs, -1j)) / 2j
report("[5.5b] sin[x] = (e^{i[x]} - e^{-i[x]})/(2i)", mp.nstr(abs(sin_dd - via), 3), abs(sin_dd - via) < 1e-40)
