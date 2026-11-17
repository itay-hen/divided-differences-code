"""Numerical checks for Chapter 9 (algorithms for the exponential).

Run:  python3 code/ch09_checks.py

The functions dd_exp_taylor, dd_exp_squaring, bessel_ratios,
dd_exp_chebyshev and the class ExpDD are the listings of the chapter,
copied verbatim (dd_taylor is Listing 8.3).  Double-precision computations
use Python floats / numpy (IEEE binary64, u = 2^-53).  Reference values
come from Horner's rule on the Opitz matrix about the smallest node in
mpmath, whose terms are all positive (Corollary 8.9), at a working
precision chosen from the spread of the nodes.
"""
import math
import random
import time
import warnings

import mpmath as mp
import numpy as np
import scipy.linalg as sl
from scipy.special import ive

warnings.filterwarnings("ignore")
U = 2.0**-53
RESULTS = []


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:64s} {info}")


# ===================================================================== listings
# ---- Listing 8.3 (from Chapter 8) -------------------------------------------
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


# ---- Listing 9.1 ------------------------------------------------------------
def dd_exp_taylor(xs, eps=2.0**-53):
    """log exp[x_0..x_q] for real nodes: Horner's rule on the Opitz
    matrix about the smallest node, nodes scaled to [0, 1]."""
    q, lo = len(xs) - 1, min(xs)
    s = max(xs) - lo
    if s == 0:
        return lo - math.lgamma(q + 1)
    n = math.ceil(math.e * s + math.log(1 / eps))       # Lemma 9.1
    K = q + n
    m0 = min(max(q, int(s)), K)          # c_m = s^(m - m0) m0!/m!
    c = [0.0] * (K + 1)
    c[m0] = 1.0
    for m in range(m0 + 1, K + 1):
        c[m] = c[m - 1] * s / m
    for m in range(m0, 0, -1):
        c[m - 1] = c[m] * m / s
    v = dd_taylor(c, [(x - lo) / s for x in xs], 0.0)[0]
    return (lo + (m0 - q) * math.log(s) - math.lgamma(m0 + 1)
            + math.log(v))


# ---- Listing 9.2 ------------------------------------------------------------
def dd_exp_squaring(xs):
    """Return (mid, th, G) with exp[x_i..x_j] = e^mid th^(i-j) G[i, j]
    for i <= j: scaling and squaring on the Opitz matrix, Thm 9.4."""
    x = np.asarray(xs, dtype=float)
    n, lo = len(x), x.min()
    s = x.max() - lo
    k = max(0, math.ceil(math.log2(s))) if s > 0 else 0
    y = (x - lo) / 2**k                # spread at most 1
    th = 2.0**round(math.log2(max(1.0, n / math.e)))   # range
    d = [1.0]                          # d_m = th^m/m!: no underflow
    for m in range(1, n + 40):         # Lemma 9.1 with s = 1
        d.append(d[-1] * th / m)
    w = y / th                         # p(Y_th) = sum d_m (Y_th/th)^m
    G = np.zeros((n, n))               # Horner's rule
    for dm in reversed(d):
        G = w[:, None] * G + np.vstack([G[1:], np.zeros(n)])
        G += dm * np.eye(n)
    G *= math.exp(-s / 2**(k + 1))     # centre the range
    i, j = np.indices((n, n))
    P = np.where(j >= i, 2.0**(i - j), 0.0)
    for _ in range(k):                 # G <- R G^2 R^-1, R=diag(2^i)
        G = (G @ G) * P
    return lo + s / 2, th, G


# ---- Listing 9.3 ------------------------------------------------------------
def bessel_ratios(c, N):
    """r[n] = I_n(c)/I_{n-1}(c) for n = 1..N, from the backward
    recurrence r_n = 1/(2n/c + r_{n+1}), started beyond N and c."""
    r = [0.0] * (N + 1)
    t = 0.0
    for n in range(int(max(N, c) + 10 * math.sqrt(c)) + 60, 0, -1):
        t = 1.0 / (2.0 * n / c + t)
        if n <= N:
            r[n] = t
    return r


def dd_exp_chebyshev(xs, tol=2.0**-53):
    """log exp[x_0..x_q] at real nodes by the Chebyshev-Bessel series,
    with W[k] = k! R_n(c) T_n[y_0..y_k]/c^k; a posteriori stop."""
    x = np.asarray(xs, dtype=float)
    q, a, b = len(x) - 1, x.min(), x.max()
    d, c = (a + b) / 2, (b - a) / 2
    if c == 0:
        return d - math.lgamma(q + 1)
    y = (x - d) / c
    Nmax = int(q + 2 * c + 60)
    r = bessel_ratios(c, Nmax + 1)
    kk = np.arange(1, q + 1)
    W_old = np.zeros(q + 1); W_old[0] = 1.0    # n = 0
    W = np.zeros(q + 1); W[0] = r[1] * y[0]    # n = 1
    if q > 0:
        W[1] = r[1] / c
    S, logR = W_old[q] + 2 * W[q], math.log(r[1])
    for n in range(1, Nmax):
        W_new = 2 * r[n + 1] * y * W - r[n + 1] * r[n] * W_old
        W_new[1:] += 2 * r[n + 1] * kk * W[:-1] / c
        W_old, W = W, W_new
        S += 2 * W[q]
        logR += math.log(r[n + 1])             # log R_{n+1}(c)
        m = n + 1
        if m > q:                              # tail bound, Lemma 9.7
            rho = r[m + 1] * (m + 1) * (m + q) / ((m - q + 1) * m)
            if rho < 1:
                logB = sum(math.log((m * m - k * k) / (2 * k + 1))
                           for k in range(q))  # log q! B(m, q)
                tail = 2 * math.exp(logR + logB - q * math.log(c))
                if tail * rho / (1 - rho) <= tol * abs(S):
                    return (d + c + math.log(ive(0, c)) + math.log(S)
                            - math.lgamma(q + 1))
    raise RuntimeError("Chebyshev series did not converge")


# ---- Listing 9.4 ------------------------------------------------------------
class ExpDD:
    """exp[x_0..x_q] for a multiset of real nodes x >= sigma, kept as
    w[m] = q! h_m(x - sigma)/(m + q)!, m = 0..M, so that
    exp[x] = e^sigma sum(w)/q!; e[m] bounds the error of w[m]."""

    def __init__(self, sigma, M, tol=1e-12):
        self.sigma, self.M, self.tol = sigma, M, tol
        self.nodes = []
        self.refresh()

    def refresh(self):                      # recompute from the nodes
        old, self.nodes = self.nodes, []
        self.w = [1.0] + [0.0] * self.M
        self.e = [0.0] * (self.M + 1)
        for x in old:
            self.add(x)

    def add(self, x):
        y = x - self.sigma
        self.nodes.append(x)
        q = len(self.nodes) - 1             # order after adding
        w, e = self.w, self.e
        pw = pe = 0.0
        for m in range(self.M + 1):         # eq. (9.16)
            if q == 0 and m == 0:
                pw, pe = w[0], e[0]
            elif q == 0:
                pw, pe = y * pw / m, y * pe / m
            else:
                pw = (q * w[m] + y * pw) / (m + q)
                pe = (q * e[m] + y * pe) / (m + q)
            pe += 3 * U * pw
            w[m], e[m] = pw, pe

    def remove(self, x):
        y = x - self.sigma
        self.nodes.remove(x)
        q = len(self.nodes)                 # order before removing
        if q == 0:
            return self.refresh()
        w, e = self.w, self.e
        for m in range(self.M, 0, -1):      # eq. (9.17)
            a, b = (m + q) * w[m], y * w[m - 1]
            w[m] = (a - b) / q
            e[m] = ((m + q) * e[m] + y * e[m - 1] + 3*U*(a + b)) / q
        if sum(e) > self.tol * sum(w):
            self.refresh()

    def log_value(self):
        q, s = len(self.nodes) - 1, math.fsum(self.w)
        return self.sigma - math.lgamma(q + 1) + math.log(s)


# ===================================================================== helpers
def ref_exp(xs, dps=30):
    """exp[x_0..x_q] in mpmath: Horner on the Opitz matrix about min x.
    All terms are positive, so dps digits are correct up to ~K ulps."""
    with mp.workdps(dps + 10):
        X = [mp.mpf(x) for x in xs]
        lo = min(X)
        y = [x - lo for x in X]
        q, s = len(y) - 1, max(y)
        n = int(math.e * float(s) + (dps + 10) * math.log(10)) + 2
        c = [mp.mpf(1)]
        for m in range(1, q + n + 1):
            c.append(c[-1] / m)
        v = dd_taylor(c, y, 0)[0]
        return +(v * mp.exp(lo))


def rel_log(lv, ref):
    """relative error of exp(lv) against the mpmath value ref > 0"""
    return abs(float(mp.expm1(mp.mpf(lv) - mp.log(ref))))


def gamma_(n):
    return n * U / (1 - n * U)


def nodes(kind, q, w, rng):
    if kind == "unif":
        return [rng.uniform(0, w) for _ in range(q + 1)]
    if kind == "low-cluster":
        return [1e-3 * rng.random() for _ in range(q)] + [w]
    if kind == "high-cluster":
        return [w - 1e-3 * rng.random() for _ in range(q)] + [0.0]
    if kind == "two":
        return [rng.choice([0.0, w]) + 1e-4 * rng.random() for _ in range(q + 1)]
    if kind == "confluent":
        vals = [rng.uniform(0, w) for _ in range(max(1, (q + 1) // 5))]
        return sorted((vals * 6)[:q + 1])
    raise ValueError(kind)


KINDS = ["unif", "low-cluster", "high-cluster", "two", "confluent"]

# ============================================================== 9.1 Taylor
print("--- 9.1 Taylor-type methods")
ok = True
for s in [0.0, 0.01, 0.5, 1, 3, 10, 50, 300, 2000]:
    for eps in [1e-8, 2.0**-53, 1e-30]:
        n = math.ceil(math.e * s + math.log(1 / eps))
        with mp.workdps(50):
            tail = mp.nsum(lambda j: mp.mpf(s)**j / mp.factorial(j), [n + 1, mp.inf]) if s else 0
        ok &= tail <= eps / (math.e - 1)
report("Lemma 9.1: n >= e s + ln(1/eps) gives tail <= eps/(e-1)", ok)

# centre choice: A(sigma) = e^sigma exp[|x - sigma|]/exp[x]
rng = random.Random(1)
ok, ok2, worst_mean = True, True, 0
for t in range(40):
    q = rng.choice([3, 10, 30]); w = 10**rng.uniform(-2, 1.5)
    xs = nodes(rng.choice(KINDS), q, w, rng)
    F = ref_exp(xs)
    xbar = sum(xs) / len(xs)
    for sig in [min(xs), xbar, max(xs), min(xs) - 1.0]:
        with mp.workdps(40):
            y = [abs(mp.mpf(x) - sig) for x in xs]
            Aabs = mp.e**sig * ref_exp(y) / F
        if sig <= min(xs):
            ok &= abs(Aabs - 1) < 1e-20
        bound = math.exp(sig - xbar + max(abs(x - sig) for x in xs))
        ok2 &= float(Aabs) <= bound * (1 + 1e-12)
    # Horner about the mean: error vs u * A * K
    sig = xbar
    smax = max(abs(x - sig) for x in xs)
    K = q + math.ceil(math.e * smax + math.log(1 / U)) + 5
    coef = [math.exp(sig)]
    for m in range(1, K + 1):
        coef.append(coef[-1] / m)
    v = dd_taylor(coef, xs, sig)[0]
    with mp.workdps(40):
        A = float(mp.e**sig * ref_exp([abs(x - sig) for x in xs]) / F)
    worst_mean = max(worst_mean, abs(v - float(F)) / float(F) / (A * (4 * K + 2) * U))
report("Prop 9.2: A(sigma) = 1 for sigma <= min x", ok)
report("Prop 9.2: A(sigma) <= exp(sigma - mean + max|x - sigma|)", ok2)
report("Horner about the mean: err <= (4K+2) u A(mean)", worst_mean <= 1, f"worst ratio {worst_mean:.2f}")

rng = random.Random(2)
ok, worst = True, 0
for t in range(80):
    q = rng.choice([0, 1, 5, 20, 60, 200]); w = 10**rng.uniform(-9, math.log10(1500))
    xs = nodes(rng.choice(KINDS), q, w, rng)
    lv = dd_exp_taylor(xs)
    e = rel_log(lv, ref_exp(xs))
    s = max(xs) - min(xs)
    K = q + math.ceil(math.e * s + math.log(1 / U))
    lsize = abs(min(xs)) + abs((max(q, int(s)) - q) * math.log(max(s, 1e-300))) + math.lgamma(max(q, int(s)) + 1) + 1
    bound = (4 * K + 2) * U + 4 * U * lsize
    worst = max(worst, e / bound)
report("Listing 9.1: relative error <= (4K+2)u + 4u|log terms|, 80 cases", worst <= 1, f"worst ratio {worst:.2f}")

# ============================================================== 9.2 scaling and squaring
print("--- 9.2 Scaling and squaring")
with mp.workdps(40):
    xs = [mp.mpf(v) for v in ["0.3", "1.7", "-0.4", "2.2", "0.9"]]
    n = len(xs)
    Zop = lambda nd: mp.matrix([[nd[i] if i == j else (1 if j == i + 1 else 0) for j in range(n)] for i in range(n)])
    ok = True
    for sfac in [2, 3, 7]:
        F = mp.expm(Zop(xs))
        G = mp.expm(Zop([x / sfac for x in xs]))**sfac
        for i in range(n):
            for j in range(i, n):
                ok &= abs(F[i, j] - G[i, j] / mp.mpf(sfac)**(j - i)) < mp.mpf(10)**-30 * abs(F[i, j])
report("Identity 7.2 (Prop 9.3): exp[x_i..x_j] = s^(i-j) [F(x/s)^s]_ij", ok)

rng = random.Random(3)
worst, worst_abs, rows = 0, 0, []
for t in range(40):
    q = rng.choice([5, 10, 20, 40, 80]); w = 10**rng.uniform(-8, math.log10(500))
    xs = nodes(rng.choice(KINDS), q, w, rng)
    lo, th, G = dd_exp_squaring(xs)
    s = max(xs) - min(xs)
    k = max(0, math.ceil(math.log2(s))) if s > 0 else 0
    K = q + 40
    step = max(1, q // 4)
    e = 0
    for a in range(0, q + 1, step):
        for b in range(a, q + 1, step):
            Fab = ref_exp(xs[a:b + 1])
            Gab = mp.mpf(G[a, b]) * mp.e**mp.mpf(lo) * mp.mpf(th)**(a - b)
            e = max(e, float(abs(Gab - Fab) / Fab))
    bound = 2**k * ((4 * K + 3) * U + gamma_(q + 1))
    worst = max(worst, e / bound)
    rows.append((s, e))
report("Thm 9.4: every entry has rel. error <= 2^k((4K+3)u + g_{q+1}), 40 cases", worst <= 1, f"worst ratio {worst:.3f}")

# similarity invariance: theta-scaled squaring gives the same relative errors
rng = random.Random(4)
ok = True
for t in range(10):
    q = 30; xs = sorted(rng.uniform(0, 1e-4) for _ in range(q + 1))
    F = float(ref_exp(xs))
    lo, th, G = dd_exp_squaring(xs)
    E1 = sl.expm(np.diag(xs) + np.diag(np.ones(q), 1))[0, q]
    ok &= abs(G[0, q] * math.exp(lo) / th**q - F) / F < 1e-13 and abs(E1 - F) / F > 1e-10
report("Thm 9.4 vs expm: squaring accurate where normwise expm fails", ok)

# Identity 4.44 (scaled log form for the thermal kernel)
rng = random.Random(5)
ok = True
with mp.workdps(50):
    for t in range(6):
        q = rng.choice([1, 3, 6]); beta = mp.mpf(rng.uniform(0.5, 5))
        E = [mp.mpf(rng.uniform(-2, 2)) for _ in range(q + 1)]
        Emin = min(E)
        M = mp.matrix(q + 1, q + 1)
        for i in range(q + 1):
            M[i, i] = -q - beta * (E[i] - Emin)
            if i + 1 <= q:
                M[i, i + 1] = q                 # upper convention
        lhs = mp.log(beta**q * ref_exp([-beta * e for e in E], dps=45))
        rhs = q - beta * Emin + q * mp.log(beta / q) + mp.log(mp.expm(M)[0, q])
        ok &= abs(lhs - rhs) < mp.mpf(10)**-35
report("Identity 4.44: ln K = q - b Emin + q ln(b/q) + ln[e^M]_corner", ok)

# ============================================================== 9.3 Chebyshev
print("--- 9.3 Chebyshev")
ok = True
for c in [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]:
    N = int(2 * c + 100)
    r = bessel_ratios(c, N)
    with mp.workdps(30):
        for nn in sorted(set([1, 2, int(c / 2) + 1, int(c) + 1, N])):
            ex = mp.besseli(nn, c) / mp.besseli(nn - 1, c)
            ok &= abs(r[nn] - ex) <= 1e-13 * ex
report("bessel_ratios: relative error < 1e-13 for c up to 5000", ok)

# Jacobi-Anger at y = 1, differentiated q times:  e^d I0(c)/c^q * 2 sum R_n B(n,q) = e^b/q!
ok = True
with mp.workdps(40):
    for c in [0.5, 3.0, 20.0]:
        for q in [1, 2, 5]:
            ssum = 2 * sum(mp.besseli(n, c) * mp.fprod((n * n - k * k) / mp.mpf(2 * k + 1) for k in range(q)) / mp.factorial(q) for n in range(q, 200))
            lhs = ssum / c**q
            ok &= abs(lhs - mp.e**c / mp.factorial(q)) < mp.mpf(10)**-25 * lhs
report("sum_n 2 I_n(c) B(n,q) = c^q e^c/q! (q-th derivative at y = 1)", ok)


def cheb_terms_mp(xs, N, dps):
    """partial sums S_0..S_{N-1} (times the prefactor) and the sum of
    |terms| of the Chebyshev-Bessel series, in mpmath at dps digits"""
    with mp.workdps(dps):
        X = [mp.mpf(x) for x in xs]
        q, a, b = len(X) - 1, min(X), max(X)
        d, c = (a + b) / 2, (b - a) / 2
        y = [(x - d) / c for x in X]
        pref = mp.e**d * mp.besseli(0, c) / c**q
        # Bessel ratios r_n = I_n/I_{n-1} by the backward recurrence
        top = int(max(N, c) + 10 * math.sqrt(float(c)) + 60 + dps)
        r = [mp.mpf(0)] * (N + 2)
        t = mp.mpf(0)
        for n in range(top, 0, -1):
            t = 1 / (2 * n / c + t)
            if n <= N + 1:
                r[n] = t
        Dm = [mp.mpf(0)] * (q + 1); Dm[0] = mp.mpf(1)
        D = [mp.mpf(0)] * (q + 1); D[0] = y[0]
        if q >= 1:
            D[1] = mp.mpf(1)
        R = r[1]
        S = Dm[q] + 2 * R * D[q]
        A = abs(Dm[q]) + abs(2 * R * D[q])
        partial = [S]
        for n in range(1, N):
            Dn = [2 * (y[k] * D[k] + (D[k - 1] if k else 0)) - Dm[k] for k in range(q + 1)]
            Dm, D = D, Dn
            R *= r[n + 1]
            tt = 2 * R * D[q]
            S += tt
            A += abs(tt)
            partial.append(S)
        return [pref * p_ for p_ in partial], pref * A


rng = random.Random(6)
ok_amp, worst, rows_cheb = True, 0, []
for t in range(50):
    q = rng.choice([1, 3, 10, 30]); w = 10**rng.uniform(-2, math.log10(300))
    xs = nodes(rng.choice(KINDS), q, w, rng)
    if max(xs) == min(xs):
        continue
    F = ref_exp(xs)
    lv = dd_exp_chebyshev(xs)
    e = rel_log(lv, F)
    with mp.workdps(30):
        Lam = float(mp.e**max(xs) / (mp.factorial(q) * F))
    N = int(q + 2 * w + 60)
    parts, Aabs = cheb_terms_mp(xs, min(N, 700), dps=int(30 + w / 2))
    A = float(Aabs / F)
    ok_amp &= A <= Lam * (1 + 1e-10)
    worst = max(worst, e / (U * Lam))
    rows_cheb.append((Lam, A, e))
report("Thm 9.8: amplification A <= Lambda = e^max/(q! exp[x]), 50 cases", ok_amp)
report("Listing 9.3: error <= 1e3 u Lambda (observed), 50 cases", worst <= 1e3, f"worst ratio {worst:.2e}")
small = [e for (L, A, e) in rows_cheb if L <= 10]
report("Listing 9.3: error < 1e-13 whenever Lambda <= 10", max(small) < 1e-13, f"max {max(small):.1e} over {len(small)}")

# a priori bound (Identity 4.8) and the a posteriori stopping rule
rng = random.Random(7)
ok1, ok2 = True, True
for t in range(8):
    q = rng.choice([1, 4, 12]); w = rng.choice([3.0, 20.0, 80.0])
    xs = sorted(rng.uniform(0, w) for _ in range(q + 1))
    xs[0], xs[-1] = 0.0, w
    dps = int(40 + w / 2)
    parts, _ = cheb_terms_mp(xs, int(q + 2 * w + 120), dps)
    with mp.workdps(dps):
        F = parts[-1]
        c, d = mp.mpf(w) / 2, mp.mpf(w) / 2
        R = lambda n: mp.besseli(n, c) / mp.besseli(0, c)
        B = lambda n: mp.fprod((n * n - k * k) / mp.mpf(2 * k + 1) for k in range(q)) / mp.factorial(q)
        for Ntr in [q + 2, q + 10, int(q + w / 2), int(q + w)]:
            if Ntr >= len(parts) - 1:
                continue
            err = abs(parts[Ntr] - F) / F          # parts[n] includes terms <= n
            rhs = 2 * mp.factorial(q) * mp.e**(d - mp.fsum(xs) / len(xs)) * mp.besseli(0, c) / c**q * \
                mp.nsum(lambda n: R(n) * B(n), [Ntr + 1, Ntr + 400])
            ok1 &= err <= rhs
    # a posteriori: stop index of Listing 9.3 at tol 1e-10, then true error
    lv = dd_exp_chebyshev(xs, tol=1e-10)
    ok2 &= rel_log(lv, ref_exp(xs)) <= 1.2e-10
report("Prop 9.6 (Identity 4.8 with the mean): a priori bound holds", ok1)
report("A posteriori rule (tol 1e-10): true error <= 1.2e-10", ok2)

# truncation length actually needed (cf. compendium amendment D.2), eps = 1e-10
print("    N needed for relative accuracy 1e-10 (exact arithmetic), vs rules:")
NEED = []
for c in [200, 2000]:
    for q in [1, int(math.sqrt(c)), int(c**0.75)]:
        Ns = []
        for seed in range(3):
            r_ = random.Random(100 + seed)
            xs = [0.0, 2.0 * c] + [r_.uniform(0, 2 * c) for _ in range(q - 1)]
            dps = int(30 + c / 2.3)                # Lambda <= e^(max - mean) ~ e^c
            parts, _ = cheb_terms_mp(xs, int(q + 1.2 * c + 200), dps)
            with mp.workdps(dps):
                F = parts[-1]
                Nn = next(n for n in range(len(parts)) if
                          all(abs(parts[m] - F) <= mp.mpf(1e-10) * F for m in range(n, len(parts))))
            Ns.append(Nn)
        heur = math.sqrt(2 * c * math.log(1e10))
        NEED.append((c, q, Ns, heur))
        print(f"    c={c:5d} q={q:4d}  N = {Ns}   sqrt(2c ln(1/eps)) = {heur:.0f}")
ok = NEED[0][2][0] >= 80 and min(NEED[-1][2]) > 3 * NEED[-1][3]
report("Amendment D.2: heuristic fails for q >> 1 (N >> sqrt(2c ln 1/eps))", ok)

# ============================================================== 9.4 add / remove
print("--- 9.4 Adding and removing nodes")
from fractions import Fraction as Fr


def h_direct(ys, m):
    """complete homogeneous symmetric polynomial by the recurrence (5.x)"""
    hs = [Fr(1)] + [Fr(0)] * m
    for y in ys:
        for j in range(1, m + 1):
            hs[j] += y * hs[j - 1]
    return hs


# exactness of the update formulas in rational arithmetic
ok = True
rng = random.Random(9)
for t in range(5):
    ys = [Fr(rng.randint(0, 20), rng.randint(1, 7)) for _ in range(rng.randint(1, 7))]
    M = 8
    w = [Fr(1)] + [Fr(0)] * M
    for qq, y in enumerate(ys):                     # add, q = qq after adding
        if qq == 0:
            for m in range(1, M + 1):
                w[m] = w[m - 1] * y / m
        else:
            new = []
            prev = Fr(0)
            for m in range(M + 1):
                prev = (qq * w[m] + y * prev) / (m + qq)
                new.append(prev)
            w = new
    q = len(ys) - 1
    hs = h_direct(ys, M)
    ok &= all(w[m] == math.factorial(q) * hs[m] / math.factorial(m + q) for m in range(M + 1))
    if q >= 1:                                      # remove ys[0]
        y = ys[0]
        w2 = [w[0]] + [((m + q) * w[m] - y * w[m - 1]) / q for m in range(1, M + 1)]
        hs2 = h_direct(ys[1:], M)
        ok &= all(w2[m] == math.factorial(q - 1) * hs2[m] / math.factorial(m + q - 1) for m in range(M + 1))
report("Prop 9.9: add/remove recurrences exact, closed under truncation", ok)


def M_for(s, eps=U):
    return max(1, math.ceil(math.e * s + math.log(1 / eps)))


def walk(s, steps, tol, seed, refresh=True):
    rng = random.Random(seed)
    D = ExpDD(0.0, M_for(s), tol=tol if refresh else math.inf)
    for _ in range(20):
        D.add(rng.uniform(0, s))
    nref = [0]
    orig = D.refresh

    def counted():
        nref[0] += 1
        orig()
    D.refresh = counted
    worst, viol = 0.0, False
    t0 = time.time()
    for step in range(1, steps + 1):
        if (rng.random() < 0.5 and len(D.nodes) > 2) or len(D.nodes) > 80:
            D.remove(D.nodes[rng.randrange(len(D.nodes))])
        else:
            D.add(rng.uniform(0, s))
        if step % (steps // 10) == 0:
            q = len(D.nodes) - 1
            with mp.workdps(40):
                R = ref_exp(D.nodes) * mp.factorial(q)
                e = float(abs(mp.fsum(D.w) - R) / R)
            worst = max(worst, e)
            viol |= e > sum(D.e) / abs(sum(D.w)) + 1e-15
    return worst, nref[0], viol, (time.time() - t0) / steps


WALK = []
ok_b, ok_e = True, True
for s in [2, 10, 40, 100]:
    worst, nref, viol, tpu = walk(s, 20000, 1e-12, 11)
    WALK.append((s, M_for(s), nref, worst, tpu))
    ok_b &= not viol
    ok_e &= worst <= 1e-12
    print(f"    s={s:4d} M={M_for(s):4d} refreshes={nref:5d}/20000  max err={worst:.1e}  {1e6 * tpu:.0f} us/update")
report("running error bound never violated on random walks", ok_b)
report("with refresh at tol 1e-12, errors stay below 1e-12", ok_e)
DRIFT = {}
for s_ in [2, 10, 40, 100]:
    DRIFT[s_] = max(walk(s_, 20000, 1e-12, seed, refresh=False)[0] for seed in range(4))
    print(f"    s={s_:4d}  max error without recomputation over 4 walks: {DRIFT[s_]:.1e}")
report("without recomputation, removal drifts (s = 40: error > 1e-3)", DRIFT[40] > 1e-3, f"{DRIFT[40]:.1e}")
report("without recomputation, s = 2 stays accurate (error < 1e-12)", DRIFT[2] < 1e-12, f"{DRIFT[2]:.1e}")

# removal ratios exp[x \ x_j]/exp[x] (energy estimator of Chapter 3)
rng = random.Random(12)
ok = True
for t in range(5):
    s = rng.choice([1.0, 5.0, 20.0]); xs = [rng.uniform(0, s) for _ in range(rng.randint(3, 25))]
    D = ExpDD(0.0, M_for(s))
    for x in xs:
        D.add(x)
    q = len(xs) - 1
    for j in [0, len(xs) // 2]:
        y = xs[j]
        w2 = [D.w[0]] + [((m + q) * D.w[m] - y * D.w[m - 1]) / q for m in range(1, D.M + 1)]
        ratio = q * math.fsum(w2) / math.fsum(D.w)
        ex = ref_exp(xs[:j] + xs[j + 1:]) / ref_exp(xs)
        ok &= abs(ratio - ex) / ex < 1e-10
report("removal ratio q sum(w^(-j))/sum(w) = exp[x\\x_j]/exp[x]", ok)

# ============================================================== 9.5 benchmark
print("--- 9.5 Benchmark")
_src = open(__file__.replace("ch09_checks.py", "ch08_checks.py")).read()
_ns = {}
_lst = _src.split("# ---- Listing 8.2")[1].split("# ---- Listing 8.3")[0]
exec("import numpy as np\n" + _lst.split("\n", 1)[1], _ns)    # Listing 8.2, verbatim
dd_contour = _ns["dd_contour"]


def benchmark_sets():
    r = random.Random(2026)
    B = {}
    B["B1 tight cluster"] = [1e-8 * r.random() for _ in range(51)]
    B["B2 unit interval"] = [r.random() for _ in range(51)]
    B["B3 spread 30"] = [30 * r.random() for _ in range(51)]
    B["B4 spread 300"] = [300 * r.random() for _ in range(51)]
    B["B5 q = 500"] = [10 * r.random() for _ in range(501)]
    vals = [5 * r.random() for _ in range(10)]
    B["B6 confluent"] = sorted(vals * 5 + vals[:1])
    B["B7 low cluster + outlier"] = [1e-3 * r.random() for _ in range(30)] + [100.0]
    B["B8 QMC-like"] = [-10.0 * r.randint(-5, 5) for _ in range(101)]
    return B


def table_dd(xs):
    xs = sorted(xs)
    d = [math.exp(x) for x in xs]
    n = len(xs)
    for k in range(1, n):
        d = [(d[i + 1] - d[i]) / (xs[i + k] - xs[i]) if xs[i + k] != xs[i] else math.nan
             for i in range(n - k)]
    return math.log(d[0]) if d[0] > 0 else math.nan


def expm_theta(xs):
    q = len(xs) - 1
    th = float(max(q, 1))
    E = sl.expm(np.diag(xs) + th * np.diag(np.ones(q), 1))[0, q]
    return math.log(E) - q * math.log(th) if E > 0 else math.nan


def contour_log(xs):
    v = dd_contour(np.exp, xs).real
    return math.log(v) if v > 0 else math.nan


def adds_log(xs):
    s = max(xs) - min(xs)
    D = ExpDD(min(xs), M_for(s))
    for x in xs:
        D.add(x)
    return D.log_value()


def squaring_log(xs):
    lo, th, G = dd_exp_squaring(xs)
    return lo + math.log(G[0, -1]) - (len(xs) - 1) * math.log(th)


METHODS = [("table", table_dd), ("Taylor (9.1)", dd_exp_taylor),
           ("squaring (9.2)", squaring_log), ("Chebyshev (9.3)", dd_exp_chebyshev),
           ("adds (9.4)", adds_log), ("contour (8.2)", contour_log),
           ("expm, theta = q", expm_theta)]
BENCH = {}
for name, xs in benchmark_sets().items():
    F = ref_exp(xs)
    row = []
    for mname, fn in METHODS:
        try:
            lv = fn(xs)
            e = rel_log(lv, F) if np.isfinite(lv) else math.inf
        except Exception:
            e = math.inf
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            try:
                fn(xs)
            except Exception:
                pass
            ts.append(time.perf_counter() - t0)
        row.append((min(e, 1e99), sorted(ts)[1]))
    BENCH[name] = row
    print(f"    {name:26s}" + "".join(f" {e:8.0e}" for e, _ in row))
print("    times (ms):")
for name, row in BENCH.items():
    print(f"    {name:26s}" + "".join(f" {1e3 * t:8.2f}" for _, t in row))
ok = all(BENCH[n][1][0] < 1e-12 and BENCH[n][2][0] < 1e-12 and BENCH[n][4][0] < 1e-12 for n in BENCH)
report("benchmark: Taylor, squaring and adds accurate (< 1e-12) on all sets", ok)
ok = BENCH["B7 low cluster + outlier"][3][0] > 1e-8 and BENCH["B1 tight cluster"][0][0] > 1
report("benchmark: Chebyshev fails on B7, table fails on B1", ok)

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
