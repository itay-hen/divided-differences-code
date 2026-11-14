"""Numerical checks for Chapter 8 (algorithms for general functions).

Run:  python3 code/ch08_checks.py        (about one minute)

The functions leja_order, dd_contour, dd_taylor and dd_enclosure are the
listings of the chapter, copied verbatim.  Double-precision computations
use Python floats / numpy (IEEE binary64, u = 2^-53).  References use
mpmath at 60 digits unless stated otherwise.
"""
import math
import random
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.linalg as sl
import warnings

warnings.filterwarnings("ignore")
mp.mp.dps = 60
U = 2.0**-53
random.seed(8)
RESULTS = []


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:64s} {info}")


# ===================================================================== listings
# ---- Listing 8.1 ------------------------------------------------------------
def leja_order(xs):
    """Return the nodes xs (real or complex) in Leja order."""
    rest = list(xs)
    c = sum(rest) / len(rest)
    k = max(range(len(rest)), key=lambda i: abs(rest[i] - c))
    out = [rest.pop(k)]
    logp = [0.0] * len(rest)          # log prod_{j<k} |x - x_j|
    while rest:
        logp = [lp + (math.log(abs(x - out[-1])) if x != out[-1]
                      else -math.inf) for lp, x in zip(logp, rest)]
        k = max(range(len(rest)), key=logp.__getitem__)
        out.append(rest.pop(k))
        logp.pop(k)
    return out


# ---- Listing 8.2 ------------------------------------------------------------
def dd_contour(f, xs, R=None, tau=1.0, rtol=1e-14, N=16, Nmax=2**16):
    """f[x_0..x_q] by the trapezoid rule on a circle |z - c| = R.
    c is the mean node; f maps numpy complex arrays to arrays, and
    |f(z)| <~ exp(tau |z|)."""
    xs = np.asarray(xs, dtype=complex)
    q = len(xs) - 1
    c = xs.mean()
    if R is None:
        R = np.abs(xs - c).max() + (q + 1) / tau

    def terms(theta):                       # f(z)(z-c)/omega(z), times R^q
        w = np.exp(1j * theta)
        z = c + R * w
        t = f(z) * w
        for x in xs:
            t /= (z - x) / R
        return t.sum(), np.abs(t).sum()

    S, A = terms(2 * np.pi * np.arange(N) / N)
    T = S / N
    while N < Nmax:                         # double N, reusing old points
        s, a = terms(2 * np.pi * (np.arange(N) + 0.5) / N)
        S, A, N = S + s, A + a, 2 * N
        T, T_old = S / N, T
        floor = 4 * (q + 2) * 2.0**-53 * A / N     # rounding level
        if abs(T - T_old) <= max(rtol * abs(T), floor):
            return T / R**q
    raise RuntimeError("trapezoid rule did not converge")


# ---- Listing 8.3 ------------------------------------------------------------
def dd_taylor(coef, xs, sigma):
    """v[i] = p[x_i..x_q] for p(z) = sum_m coef[m] (z - sigma)^m, by
    Horner's rule on the Opitz matrix: v <- (Z - sigma I) v + c_m e_q."""
    y = [x - sigma for x in xs]
    q = len(y) - 1
    v = [0 * y[0]] * (q + 1)
    for c in reversed(coef):
        for i in range(q):
            v[i] = y[i] * v[i] + v[i + 1]
        v[q] = y[q] * v[q] + c
    return v                                   # v[0] = p[x_0..x_q]


# ---- Listing 8.4 ------------------------------------------------------------
iv = mp.iv


def dd_enclosure(f, xs, digits=16, prec=64):
    """Certified enclosure of f[x_0..x_q] at distinct real nodes by the
    table in interval arithmetic; f must accept mpmath intervals. The
    precision is doubled until the relative width is < 10^-digits."""
    while True:
        iv.prec = prec
        X = [iv.mpf(x) for x in xs]           # exact for binary floats
        d = [f(x) for x in X]
        for k in range(1, len(X)):
            d = [(d[i + 1] - d[i]) / (X[i + k] - X[i])
                 for i in range(len(X) - k)]
        F = d[0]
        if (F.a > 0 or F.b < 0) and \
                F.delta <= 10**-digits * min(abs(F.a), abs(F.b)):
            return F
        prec *= 2


# ===================================================================== helpers
def table(fv, xs):
    d = list(fv)
    n = len(xs)
    for k in range(1, n):
        d = [(d[i + 1] - d[i]) / (xs[i + k] - xs[i]) for i in range(n - k)]
    return d[0]


def table_prefix(fv, xs):
    """Newton coefficients d_{0,k}, k = 0..q, by the table."""
    d = list(fv)
    n = len(xs)
    out = [d[0]]
    for k in range(1, n):
        d = [(d[i + 1] - d[i]) / (xs[i + k] - xs[i]) for i in range(n - k)]
        out.append(d[0])
    return out


def newton_eval(c, xs, t):
    p = c[-1] * np.ones_like(t)
    for k in range(len(c) - 2, -1, -1):
        p = p * (t - xs[k]) + c[k]
    return p


def opitz(xs, theta=1):
    q = len(xs) - 1
    Z = mp.matrix(q + 1, q + 1)
    for i, x in enumerate(xs):
        Z[i, i] = mp.mpmathify(x)
        if i < q:
            Z[i, i + 1] = theta
    return Z


def ref_exp(xs):
    return mp.expm(opitz(xs))[0, len(xs) - 1]


def ref_cos(xs):
    Z = opitz(xs)
    q = len(xs) - 1
    return (mp.expm(1j * Z)[0, q] + mp.expm(-1j * Z)[0, q]) / 2


def trap_mp(f, xs, c, R, N):
    """Trapezoid rule (4.x) in mpmath."""
    s = 0
    for k in range(N):
        w = mp.expj(2 * mp.pi * k / N)
        z = c + R * w
        om = 1
        for x in xs:
            om *= z - x
        s += f(z) * R * w / om
    return s / N


def h_abs(y, n):
    """h_n(|y_0|,...,|y_q|) via the Opitz recursion in mpmath."""
    q = len(y) - 1
    v = [mp.mpf(0)] * (q + 1)
    v[q] = mp.mpf(1)
    for _ in range(n + q):
        v = [abs(y[i]) * v[i] + v[i + 1] for i in range(q)] + [abs(y[q]) * v[q]]
    return v[0]


# ============================================================== 8.1 ordering
print("== 8.1 node ordering and Leja sequences")
ok = True
for trial in range(60):
    q = random.randint(3, 25)
    if trial % 2:
        xs = [random.uniform(-2, 3) for _ in range(q + 1)]
    else:
        xs = [complex(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(q + 1)]
    o = leja_order(xs)
    if sorted(map(lambda z: (complex(z).real, complex(z).imag), o)) != \
            sorted(map(lambda z: (complex(z).real, complex(z).imag), xs)):
        ok = False
    for k in range(1, q + 1):
        wk = lambda x: math.prod(abs(x - o[j]) for j in range(k))
        if any(wk(o[m]) > wk(o[k]) * (1 + 1e-12) for m in range(k, q + 1)):
            ok = False
c0 = sum(xs) / len(xs)
report("Listing 8.1: Leja property |w_k(x_m)| <= |w_k(x_k)|, m >= k", ok)
report("Listing 8.1: first node is farthest from the mean",
       abs(o[0] - c0) >= max(abs(x - c0) for x in xs) - 1e-14)
o = leja_order([0.3, 0.3, 1.0, 1.0, -2.0])
report("Listing 8.1 handles repeated nodes", sorted(o) == [-2.0, 0.3, 0.3, 1.0, 1.0], str(o))

# capacity of [-1,1] = 1/2: Leja points from a fine grid
g = np.cos(np.linspace(0, np.pi, 200001))
logp = np.log(np.abs(g - 1.0) + 1e-300)
roots = {}
for k in range(1, 1001):
    i = int(np.argmax(logp))
    roots[k] = math.exp(logp[i] / k)
    logp += np.log(np.abs(g - g[i]) + 1e-300)
report("Leja products^(1/k) -> capacity 1/2 of [-1,1]",
       abs(roots[1000] - 0.5) < 0.005 and roots[100] > roots[1000],
       f"k=10: {roots[10]:.4f}, 100: {roots[100]:.4f}, 1000: {roots[1000]:.4f}")

# Newton form at Chebyshev points: increasing vs Leja vs random order
t = np.linspace(-1, 1, 2001)
newton_err = {}
for fname, f in [("exp", np.exp), ("runge4", lambda x: 1 / (1 + 4 * x * x))]:
    for q in [40, 60, 80, 120]:
        x = np.cos(np.pi * (np.arange(q + 1) + 0.5) / (q + 1))
        rnd = list(x)
        random.Random(q).shuffle(rnd)
        for oname, oo in [("inc", sorted(x)), ("leja", leja_order(list(x))), ("rand", rnd)]:
            oo = np.array(oo)
            c = table_prefix(f(oo), oo)
            newton_err[fname, q, oname] = np.max(np.abs(newton_eval(np.array(c), oo, t) - f(t)))
report("Newton form, q=80: increasing order fails (exp)", newton_err["exp", 80, "inc"] > 1e2,
       f"{newton_err['exp', 80, 'inc']:.1e}")
report("Newton form, q=40..120: Leja order error < 1e-14 (exp)",
       max(newton_err["exp", q, "leja"] for q in [40, 60, 80, 120]) < 1e-14,
       f"max {max(newton_err['exp', q, 'leja'] for q in [40, 60, 80, 120]):.1e}")
report("Newton form, q=120: Leja error < 1e-14 (1/(1+4x^2))", newton_err["runge4", 120, "leja"] < 1e-14,
       f"{newton_err['runge4', 120, 'leja']:.1e}; inc {newton_err['runge4', 120, 'inc']:.1e}")

# the coefficients themselves are inaccurate in Leja order, yet the interpolant is fine
q = 80
x = np.cos(np.pi * (np.arange(q + 1) + 0.5) / (q + 1))
oo = leja_order(list(x))
c = table_prefix(np.exp(oo), oo)
cm = [mp.mpf(0)] * 0
d = [mp.e**mp.mpf(v) for v in oo]
cm = [d[0]]
for k in range(1, q + 1):
    d = [(d[i + 1] - d[i]) / (mp.mpf(oo[i + k]) - mp.mpf(oo[i])) for i in range(q + 1 - k)]
    cm.append(d[0])
worst_coef = max(float(abs(mp.mpf(a) - b) / abs(b)) for a, b in zip(c, cm))
report("Leja order, q=80: some coefficients have relative error > 1e20", worst_coef > 1e20, f"{worst_coef:.1e}")

# the forward bound (path sums) is far too pessimistic for the Newton form
A = [abs(v) for v in np.exp(oo)]
Apref = [A[0]]
for k in range(1, q + 1):
    A = [(A[i + 1] + A[i]) / abs(oo[i + k] - oo[i]) for i in range(q + 1 - k)]
    Apref.append(A[0])
om = np.ones_like(t)
B = np.zeros_like(t)
for k in range(q + 1):
    B += Apref[k] * np.abs(om)
    om = om * (t - oo[k])
report("forward bound gamma_{3q+1} sum_k A_k |w_k(t)| exceeds the error by > 1e20",
       3 * q * U * B.max() > 1e20 * newton_err["exp", 80, "leja"], f"bound {3 * q * U * B.max():.1e}")

# ============================================================== 8.2 contour
print("== 8.2 contour quadrature")
# aliasing formula: T_N - f[x] = sum_m R^{mN} f[x, c^(mN)] + sum_m R^{-mN} ((z-c)^{mN} f)[x]
a = mp.mpf(5)
fres = lambda z: 1 / (a - z)
xs = [mp.mpf("0.3"), mp.mpf("-0.2"), mp.mpc("0.1", "0.2"), mp.mpf("0.3")]
c, R, N = mp.mpf("0.05"), mp.mpf(2), 8
exact = 1 / mp.fprod([a - x for x in xs])
T = trap_mp(fres, xs, c, R, N)
pos = sum(R**(m * N) * exact / (a - c)**(m * N) for m in range(1, 80))   # f[x,c^(n)] for resolvent
neg = sum(R**(-m * N) * trap_mp(lambda z, m=m: (z - c)**(m * N) * fres(z), xs, c, mp.mpf(1.5), 300)
          for m in range(1, 12))
report("aliasing formula, resolvent, complex + repeated nodes",
       abs((T - exact) - (pos + neg)) < mp.mpf(10)**-40 * abs(T - exact), mp.nstr(T - exact, 6))

xs = [mp.mpf(v) for v in ["0.1", "0.4", "0.4", "-0.3", "0.25"]]
c, R, N = mp.mpf("0.1"), mp.mpf(3), 10
exact = ref_exp(xs)
T = trap_mp(mp.exp, xs, c, R, N)


def dd_with_c(n):     # e[x, c^(n)] = [e^Z]_{0,q+n} for the extended Opitz matrix
    return ref_exp(xs + [c] * n)


pos = sum(R**(m * N) * dd_with_c(m * N) for m in range(1, 6))
neg = sum(R**(-m * N) * trap_mp(lambda z, m=m: (z - c)**(m * N) * mp.exp(z), xs, c, mp.mpf(1), 300)
          for m in range(1, 8))
report("aliasing formula, exp, repeated nodes", abs((T - exact) - (pos + neg)) < mp.mpf(10)**-40,
       f"pos {mp.nstr(pos, 4)}, neg {mp.nstr(neg, 4)}")
report("aliasing, exp, real data: positive part > 0", pos > 0)

# geometric bound (Corollary): resolvent, nodes in |z-c|<=r, f analytic in |z-c|<rho
xs = [mp.mpf(v) for v in ["0.5", "-0.4", "0.1", "0.45"]]
c = mp.mpf(0)
r = max(abs(x) for x in xs)
rho = a
exact = 1 / mp.fprod([a - x for x in xs])
ok = True
for R in [mp.mpf(1), mp.mpf(1.5), mp.mpf(2.5)]:
    for N in [4, 8, 16, 32]:
        err = abs(trap_mp(fres, xs, c, R, N) - exact)
        best = mp.inf
        for s2 in mp.linspace(R * 1.01, rho * 0.999, 25):
            for s1 in mp.linspace(r * 1.001 + 1e-9, R * 0.999, 25):
                K2 = s2 * mp.mpf(1) / (rho - s2) / (s2 - r)**len(xs)
                K1 = s1 * mp.mpf(1) / (rho - s1) / (s1 - r)**len(xs)
                b = K2 * (R / s2)**N / (1 - (R / s2)**N) + K1 * (s1 / R)**N / (1 - (s1 / R)**N)
                best = min(best, b)
        if err > best:
            ok = False
report("geometric aliasing bound holds (resolvent, 12 (R,N) pairs)", ok)

# exponential: explicit bounds on the two alias sums, real nodes
ok = True
for trial in range(20):
    q = random.randint(1, 8)
    xs = [mp.mpf(random.uniform(-1, 1)) for _ in range(q + 1)]
    c = sum(xs) / (q + 1)
    r = max(abs(x - c) for x in xs)
    R = r + q + 1
    for N in [8, 16, 24]:
        T = trap_mp(mp.exp, xs, c, R, N)
        F = ref_exp(xs)
        sig = max(xs)
        b1 = mp.e**sig * R**N / mp.factorial(q + N) / (1 - R**N * mp.factorial(q + N) / mp.factorial(q + 2 * N))
        b2 = mp.e**sig * (r + N)**q * r**(N - q) / R**N / mp.factorial(q) * 2 if N > q else mp.inf
        if not (abs(T - F) <= b1 + b2):
            ok = False
report("exp alias estimates e^max R^N/(q+N)! + 2e^max (r+N)^q r^(N-q)/(q! R^N)", ok)

# rounding: computed T vs exact trapezoid sum, relative to S = mean |terms|
ok, worst = True, 0
for trial in range(40):
    q = random.choice([3, 8, 15, 25])
    xs = [random.uniform(-1, 1) * 10**random.uniform(-6, 0) for _ in range(q + 1)]
    c = sum(xs) / (q + 1)
    r = max(abs(x - c) for x in xs)
    R = r + q + 1
    N = 4 * q + 32
    th = 2 * np.pi * np.arange(N) / N
    z = c + R * np.exp(1j * th)
    terms = np.exp(z) * R * np.exp(1j * th)
    for x in xs:
        terms = terms / (z - x)
    Tc = terms.mean()
    Tm = trap_mp(mp.exp, [mp.mpf(x) for x in xs], mp.mpf(c), mp.mpf(R), N)
    S = np.abs(terms).mean()
    ratio = abs(complex(Tm) - Tc) / (U * S)
    worst = max(worst, ratio / (q + N))
    F = ref_exp([mp.mpf(x) for x in xs])
    cauchy_ratio = mp.factorial(q) * R * mp.e**R / (R - r)**(q + 1)
    if S > float(cauchy_ratio * F) * (1 + 1e-12) * math.exp(c) / float(mp.e**mp.mpf(c)):
        ok = False
report("rounding error of the sum <= (q+N) u S", worst <= 1.0, f"max |err|/((q+N)uS) = {worst:.2f}")
report("S <= q! R e^R / (R-r)^(q+1) * F  (real nodes, c = mean)", ok)

# the default radius R = r + q + 1 is near-optimal: compare R = q/2, q, 2q for a cluster
q = 30
xs = [0.2 + 1e-6 * random.random() for _ in range(q + 1)]
F = ref_exp(xs)
errR = {}
for R in [q / 2 + 1, q + 1, 2 * q + 2]:
    errR[R] = abs(dd_contour(np.exp, xs, R=R) - complex(F)) / float(F)
report("cluster q=30: R=q+1 beats R=q/2+1 and R=2q+2",
       errR[q + 1] < errR[q / 2 + 1] and errR[q + 1] < errR[2 * q + 2],
       " ".join(f"R={k:.0f}:{v:.0e}" for k, v in errR.items()))
report("cluster q=30: R=q+1 error < 1e-14", errR[q + 1] < 1e-14)

# Listing 8.2 battery
worst = {}
for trial in range(80):
    q = random.choice([0, 1, 3, 8, 15, 30])
    kind = random.choice(["cluster", "unit", "spread10", "complex", "repeat"])
    if kind == "cluster":
        xs = [0.3 + 1e-7 * random.random() for _ in range(q + 1)]
    elif kind == "unit":
        xs = [random.uniform(-1, 1) for _ in range(q + 1)]
    elif kind == "spread10":
        xs = [random.uniform(-10, 10) for _ in range(q + 1)]
    elif kind == "complex":
        xs = [complex(random.uniform(-1, 1), random.uniform(-3, 3)) for _ in range(q + 1)]
    else:
        b = [random.uniform(-1, 1) for _ in range(3)]
        xs = [random.choice(b) for _ in range(q + 1)]
    for fname, f, rf in [("exp", np.exp, ref_exp), ("cos", np.cos, ref_cos)]:
        F = complex(rf(xs))
        T = dd_contour(f, xs)
        scale = abs(F)
        if fname == "cos":   # cos[x] can vanish: measure against max|cos^(q)|/q! on the hull
            scale = max(abs(F), math.cosh(max(abs(complex(x).imag) for x in xs)) / math.factorial(q))
        key = (fname, kind)
        worst[key] = max(worst.get(key, 0), abs(T - F) / scale)
w_local = max(v for k, v in worst.items() if k[1] != "spread10")
w_spread = max(v for k, v in worst.items() if k[1] == "spread10")
report("Listing 8.2: 160 cases, error < 1e-14 unless spread", w_local < 1e-14, f"worst {w_local:.1e}")
report("Listing 8.2: nodes in [-10,10], error < 1e-12", w_spread < 1e-12, f"worst {w_spread:.1e}")

# oscillatory kernel e^{-i t x} at real nodes (complex values, exact zeros possible)
ok, worst_osc = True, 0
for trial in range(10):
    q = random.choice([5, 10, 20])
    tt = random.uniform(1, 15)
    xs = [random.uniform(-1, 1) for _ in range(q + 1)]
    F = complex(mp.expm(-1j * tt * opitz(xs))[0, q])
    T = dd_contour(lambda z: np.exp(-1j * tt * z), xs, tau=tt)
    r = max(abs(x - sum(xs) / len(xs)) for x in xs)
    ratio = abs(T - F) / (U * math.exp(tt * r) * tt**q / math.factorial(q))
    worst_osc = max(worst_osc, ratio)
    if ratio > 10 * math.sqrt(q):
        ok = False
report("e^{-itx}, tau = t: error <= 10 sqrt(q) u e^{t r} t^q/q!", ok, f"max ratio {worst_osc:.1e}")

# a single circle degrades for widely spread nodes (factor e^r in Prop. 8.x)
spread_err = []
for q in [5, 10, 30]:
    rr = random.Random(50 + q)
    xs = [rr.uniform(-100, 100) for _ in range(q + 1)]
    with mp.workdps(150):
        F = float(ref_exp(xs))
    spread_err.append(abs(dd_contour(np.exp, xs) - F) / F)
report("single circle, nodes in [-100,100]: digits are lost (error > 1e-12)",
       max(spread_err) > 1e-12, " ".join(f"{e:.0e}" for e in spread_err))

# sum over clusters: f[x] = sum_k (f / omega_{other})[x_{C_k}]
xs = [mp.mpf(v) for v in ["0", "0.01", "0.02", "3", "3.001", "7", "7.5", "7.25"]]
cl = [(0, 3), (3, 5), (5, 8)]
tot = 0
for (lo, hi) in cl:
    C = xs[lo:hi]
    O = xs[:lo] + xs[hi:]
    g = lambda z, O=O: mp.exp(z) / mp.fprod([z - o for o in O])
    cc = sum(C) / len(C)
    tot += trap_mp(g, C, cc, mp.mpf("0.6"), 200)
report("sum over clusters reproduces e[x] (exact identity)", abs(tot - ref_exp(xs)) < mp.mpf(10)**-40)


def cluster_sum(xs, cl):
    tot, absum = 0, 0
    for (lo, hi) in cl:
        C = xs[lo:hi]
        O = np.concatenate([xs[:lo], xs[hi:]])
        g = lambda z, O=O: np.exp(z) / np.prod([(z - o) for o in O], axis=0)
        cc = np.mean(C)
        rad = min(0.4 * min(abs(o - cc) for o in O), np.abs(C - cc).max() + len(C))
        v = dd_contour(g, C, R=rad)
        tot, absum = tot + v, absum + abs(v)
    return tot, absum


def three_clusters(gap, m=10, seed=4):
    rr = random.Random(seed)
    xs, cl = [], []
    for k in range(3):
        lo = len(xs)
        xs += sorted(k * gap + rr.uniform(0, 1e-6) for _ in range(m))
        cl.append((lo, len(xs)))
    return np.array(xs), cl


amp = {}
for gap in [1, 3, 10, 30, 100]:
    xs, cl = three_clusters(gap)
    with mp.workdps(150):
        F = float(ref_exp(list(xs)))
    v, ab = cluster_sum(xs, cl)
    amp[gap] = (ab / F, abs(v - F) / F)
report("cluster sum: amplification > 1e30 at gap 1, ~1 at gaps 30, 100",
       amp[1][0] > 1e30 and amp[30][0] < 1.01 and amp[100][0] < 1.01,
       " ".join(f"{g}:{a[0]:.0e}" for g, a in amp.items()))
report("cluster sum: error < 1e-14 at gaps 30 and 100", amp[30][1] < 1e-14 and amp[100][1] < 1e-14,
       f"{amp[30][1]:.0e}, {amp[100][1]:.0e}")
print("   cluster sum errors:", " ".join(f"{g}:{a[1]:.0e}" for g, a in amp.items()))
report("cluster sum: error <= 100 u * amplification (all gaps)",
       all(a[1] <= 100 * U * max(a[0], 1) for a in amp.values()))

# ============================================================== 8.3 Opitz / matrix functions
print("== 8.3 Opitz matrix and matrix-function algorithms")
ok_same, info = True, []
for q, w in [(5, 1e-3), (10, 1e-3), (20, 1e-2), (20, 3.0), (30, 1e-6)]:
    rr = random.Random(q)
    xs = sorted(rr.uniform(0, w) for _ in range(q + 1))
    F = float(ref_exp(xs))
    Zd = np.diag(xs) + np.diag(np.ones(q), 1)
    fm, est = sl.funm(Zd, np.exp, disp=False)
    tb = table(np.exp(xs), xs)
    e_fm, e_tb = abs(fm[0, q] - F) / F, abs(tb - F) / F
    info.append(f"{e_fm:.0e}/{e_tb:.0e}")
    if not (0.5 < e_fm / e_tb < 2):
        ok_same = False
report("scipy funm on Z has the table's error (ratio in [0.5,2])", ok_same, " ".join(info))
report("funm's own error estimate is normwise (q=30: est < 1e-5, error > 1e100)",
       est < 1e-5 and e_fm > 1e100, f"est {est:.0e}, err {e_fm:.0e}")

ok_amp, ok_theta, rows = True, True, []
for q, w in [(10, 1e-3), (20, 1e-2), (20, 3.0), (30, 1e-6), (30, 5.0), (60, 1e-3), (60, 10.0)]:
    rr = random.Random(100 + q)
    xs = sorted(rr.uniform(0, w) for _ in range(q + 1))
    F = float(ref_exp(xs))
    E1 = sl.expm(np.diag(xs) + np.diag(np.ones(q), 1))
    Eq = sl.expm(np.diag(xs) + q * np.diag(np.ones(q), 1))
    e1 = abs(E1[0, q] - F) / F
    eq = abs(Eq[0, q] / q**q - F) / F
    a1 = np.abs(E1).max() / F
    aq = np.abs(Eq).max() / (F * q**q)
    rows.append((q, w, e1, a1, eq, aq))
    if e1 > 10 * q * U * a1:
        ok_amp = False
    if not (aq < 3 and eq < 1e-9):
        ok_theta = False
report("expm(Z) corner error <= 10 q u ||e^Z||_max / F", ok_amp,
       f"q=30,w=1e-6: err {rows[3][2]:.0e}, amp {rows[3][3]:.0e}")
report("superdiagonal theta = q: amplification < 3, error < 1e-9", ok_theta,
       f"max err {max(r[4] for r in rows):.0e}")
EXPM_ROWS = rows
for r_ in rows:
    print(f"   q={r_[0]:3d} w={r_[1]:7.0e}  expm err {r_[2]:.0e}  amp {r_[3]:.0e}  theta=q err {r_[4]:.0e}  amp {r_[5]:.1f}")

# Taylor through the Opitz matrix
ok = True
for trial in range(30):
    q = random.randint(0, 6)
    y = [Fraction(random.randint(-9, 9), random.randint(1, 5)) for _ in range(q + 1)]
    for m in range(q, q + 6):
        Y = [[Fraction(0)] * (q + 1) for _ in range(q + 1)]
        for i in range(q + 1):
            Y[i][i] = y[i]
            if i < q:
                Y[i][i + 1] = Fraction(1)
        P = [[Fraction(int(i == j)) for j in range(q + 1)] for i in range(q + 1)]
        for _ in range(m):
            P = [[sum(P[i][k] * Y[k][j] for k in range(q + 1)) for j in range(q + 1)] for i in range(q + 1)]
        coef = [Fraction(0)] * m + [Fraction(1)]
        v = dd_taylor(coef, y, Fraction(0))
        # h_{m-q}(y) by brute force over multisets
        def h(n, ys):
            if not ys:
                return Fraction(int(n == 0))
            return sum(ys[0]**k * h(n - k, ys[1:]) for k in range(n + 1))
        if not (P[0][q] == h(m - q, y) == v[0]):
            ok = False
report("[(Z-s)^m]_{0q} = h_{m-q}(x-s) = dd_taylor(z^m)  (exact rationals)", ok)


def exp_coef(sigma, K):
    c = [math.exp(sigma)]
    for m in range(1, K + 1):
        c.append(c[-1] / m)
    return c


ok, worst, worst_trunc = True, 0, 0
for trial in range(60):
    q = random.choice([0, 3, 10, 30, 60])
    s = random.choice([1e-8, 1e-3, 0.5, 5, 30])
    xs = sorted(random.uniform(-1, -1 + s) for _ in range(q + 1))
    sigma = xs[0]
    n_tail = 1
    while s**n_tail / math.factorial(n_tail) * math.exp(s) > 1e-17 or n_tail < s:
        n_tail += 1
    K = q + n_tail
    v = dd_taylor(exp_coef(sigma, K), xs, sigma)
    F = ref_exp(xs)
    e = float(abs(v[0] - F) / F)
    worst = max(worst, e / (U * (4 * K + 2)))
    if e > (4 * K + 2) * U + 1e-17:
        ok = False
report("Listing 8.3, exp, sigma = min node: rel. error <= (4K+2)u", ok, f"max err/((4K+2)u) = {worst:.2f}")

# truncation bound: relative tail <= sum_{n>K-q} s^n/n!
ok = True
for trial in range(30):
    q = random.choice([2, 5, 10])
    s = random.choice([0.5, 2.0, 6.0])
    xs = sorted(random.uniform(0, s) for _ in range(q + 1))
    xs[0], xs[-1] = 0.0, s
    F = ref_exp(xs)
    for nt in [2, 5, 10, 20]:
        coef = [mp.mpf(1) / mp.factorial(m) for m in range(q + nt + 1)]
        y = [mp.mpf(x) for x in xs]
        v = dd_taylor(coef, y, mp.mpf(0))[0]
        tail = sum(mp.mpf(s)**n / mp.factorial(n) for n in range(nt + 1, nt + 200))
        if (F - v) / F > tail or v > F:
            ok = False
report("truncation: 0 <= (F - p_K[x])/F <= sum_{n>K-q} s^n/n!", ok)

# general f: rounding bound |err| <= gamma_{2K+1} sum |c_m| h_{m-q}(|y|) (cos, complex nodes)
ok, worst = True, 0
for trial in range(30):
    q = random.choice([2, 6, 12])
    sigma = complex(random.uniform(-1, 1), random.uniform(-1, 1))
    xs = [sigma + 0.5 * complex(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(q + 1)]
    K = q + 45
    cs = [mp.cos(sigma), -mp.sin(sigma)]
    coef_mp = [(cs[0] if m % 4 == 0 else cs[1] if m % 4 == 1 else -cs[0] if m % 4 == 2 else -cs[1])
               / mp.factorial(m) for m in range(K + 1)]
    coef = [complex(c) for c in coef_mp]
    v = dd_taylor(coef, xs, sigma)[0]
    y = [mp.mpc(x - sigma) for x in xs]
    pK = dd_taylor([mp.mpc(c) for c in coef], y, mp.mpc(0))[0]      # same coefficients, exact-ish
    bnd = sum(abs(mp.mpc(coef[m])) * h_abs(y, m - q) for m in range(q, K + 1))
    gam = (2 * K + 1) * 4 * U          # complex arithmetic: constant 4 (Higham, Sec. 3.6)
    err = abs(mp.mpc(v) - pK)
    worst = max(worst, float(err / (gam * bnd)))
    if err > gam * bnd:
        ok = False
report("Taylor rounding bound, cos at complex nodes", ok, f"max err/bound = {worst:.1e}")

# block Parlett: cross-cluster recursion vs gap (Figure 8.3 data)
def block_parlett(xs, cl, K=80):
    n = len(xs)
    F = np.zeros((n, n))
    lab = np.empty(n, int)
    for k, (lo, hi) in enumerate(cl):
        lab[lo:hi] = k
        sig = xs[lo]
        C = list(xs[lo:hi])
        for j in range(hi - lo):
            F[lo:lo + j + 1, lo + j] = dd_taylor(exp_coef(sig, K), C[:j + 1], sig)
    for d in range(1, n):
        for i in range(n - d):
            j = i + d
            if lab[i] != lab[j]:
                F[i, j] = (F[i + 1, j] - F[i, j - 1]) / (xs[j] - xs[i])
    return F[0, n - 1]


bp = {}
for gap in [1, 3, 10, 30, 100]:
    xs, cl = three_clusters(gap)
    with mp.workdps(150):
        F = float(ref_exp(list(xs)))
    bp[gap] = abs(block_parlett(xs, cl) - F) / F
report("block Parlett: error > 1 at gap 1, < 1e-12 at gap 100", bp[1] > 1 and bp[100] < 1e-12,
       " ".join(f"{g}:{e:.0e}" for g, e in bp.items()))

# Taylor with sigma = min handles all gaps (positive terms)
ok, info = True, []
for gap in [1, 3, 10, 30, 100]:
    xs, cl = three_clusters(gap)
    s = xs[-1] - xs[0]
    n_tail = 1
    while n_tail * math.log(s) - math.lgamma(n_tail + 1) > math.log(1e-17) or n_tail < 3 * s:
        n_tail += 1
    K = len(xs) - 1 + n_tail
    q = len(xs) - 1
    # scale nodes to [0,1] (Eq. 2.affine): p[x] = s^{-q} p~[(x - sigma)/s], p~(w) = p(sigma + s w)
    coef = [math.exp(xs[0])]                 # c_m = e^sigma s^m / m!, relative error <= gamma_{2m+1}
    for m in range(1, K + 1):
        coef.append(coef[-1] * s / m)
    v = dd_taylor(coef, list((xs - xs[0]) / s), 0.0)[0] / s**q
    with mp.workdps(150):
        F = float(ref_exp(list(xs)))
    e = abs(v - F) / F
    info.append(f"{gap}:{e/((5 * K + 2) * U):.2f}")
    if e > (5 * K + 2) * U:
        ok = False
report("Taylor (sigma = min, scaled) on three clusters: error <= (5K+2)u", ok, "err/bound " + " ".join(info))

# ============================================================== 8.4 references
print("== 8.4 arbitrary-precision references")
ok, info = True, []
for q, w in [(7, 1e-3), (20, 1e-6), (40, 1.0), (30, 1e-10)]:
    rr = random.Random(q)
    xs = sorted(0.5 + rr.uniform(0, w) for _ in range(q + 1))
    if q == 7:
        xs = [0.5 + 1e-3 * j for j in range(8)]          # the nodes of Table 2.x
    Fi = dd_enclosure(iv.exp, xs)
    with mp.workdps(int(-q * math.log10(w)) + 80):
        R = ref_exp(xs)
        inside = (mp.mpf(Fi.a) - R) <= mp.mpf(10)**-60 * R and (R - mp.mpf(Fi.b)) <= mp.mpf(10)**-60 * R
        rel = mp.mpf(Fi.delta.b) / R
    info.append(f"q={q}:{iv.prec}b")
    if not (inside and rel <= 1e-16):
        ok = False
report("Listing 8.4: enclosure contains the value, width <= 1e-16", ok, " ".join(info))

# exact rational arithmetic for polynomial f
xs = [Fraction(1, 3), Fraction(2, 7), Fraction(-5, 4), Fraction(1, 3) + Fraction(1, 10**12), Fraction(3, 2)]
pcoef = [Fraction(k * k - 3, k + 1) for k in range(9)]           # p(z) = sum c_k z^k
pv = lambda z: sum(cc * z**k for k, cc in enumerate(pcoef))
tb = table([pv(x) for x in xs], xs)
ty = dd_taylor(pcoef, xs, Fraction(0))[0]
report("exact rationals: table = Opitz-Horner for a polynomial", tb == ty, f"{float(tb):.6g}")

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
