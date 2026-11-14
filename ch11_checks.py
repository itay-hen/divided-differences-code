"""Numerical checks for Chapter 11 (time-ordered integrals and the
Dyson series).

Run:  python3 code/ch11_checks.py

The functions dd_expi_squaring, compositions, ordered_poly_integral and
dyson_driven are the listings of the chapter, copied verbatim.  dd_push
(Proposition 9.9) is reused from code/ch10_checks.py.  Ordered integrals
are computed independently and exactly, by closed-form antiderivatives
of exponential polynomials in mpmath with dyadic (exactly representable)
rates; divided differences are referenced in mpmath by the Taylor series
about the centre of the nodes at a precision chosen from their spread.
"""
import itertools
import math
import random
import time
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy.linalg as sl
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")
U = 2.0**-53
RESULTS = []
HERE = Path(__file__).resolve().parent


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:64s} {info}")


# dd_push (Proposition 9.9, used by Listing 10.4) from Chapter 10
_ns = {"__file__": str(HERE / "ch10_checks.py")}
exec((HERE / "ch10_checks.py").read_text().split(
    "# ===================================================================== helpers")[0], _ns)
dd_push = _ns["dd_push"]


# ===================================================================== listings
# ---- Listing 11.1 -----------------------------------------------------------
def dd_expi_squaring(E, t):
    """Return (mid, th, G) with e^{-it[E_i..E_j]} =
    (-it)^(j-i) e^(-it mid) th^(i-j) G[i, j] for i <= j, E real:
    scaling and squaring on the Opitz matrix of the nodes -itE."""
    E = np.asarray(E, dtype=float)
    n, mid = len(E), (E.max() + E.min()) / 2
    s = t * (E.max() - E.min())        # spread of the nodes -itE
    k = max(0, math.ceil(math.log2(s))) if s > 0 else 0
    y = -1j * t * (E - mid) / 2**k     # centred, |y| <= 1/2
    th = 2.0**round(math.log2(max(1.0, n / math.e)))
    c = [1.0]
    for m in range(1, n + 40):         # Lemma 9.1 with s = 1
        c.append(c[-1] / m)
    G = np.zeros((n, n), dtype=complex)
    for cm in reversed(c):             # G = p(Y_th), Horner's rule
        G = y[:, None] * G + th * np.vstack([G[1:], np.zeros(n)])
        G += cm * np.eye(n)
    i, j = np.indices((n, n))
    P = np.where(j >= i, 2.0**(i - j), 0.0)
    for _ in range(k):                 # G <- R G^2 R^-1, R=diag(2^i)
        G = (G @ G) * P
    return mid, th, G


# ---- Listing 11.2 -----------------------------------------------------------
def compositions(n, k):
    """All k-tuples of nonnegative integers with sum n."""
    if k == 1:
        yield (n,)
        return
    for a in range(n + 1):
        for rest in compositions(n - a, k - 1):
            yield (a,) + rest


def ordered_poly_integral(gam, p, T):
    """Integral over 0 < t_1 < ... < t_q < T of
    prod_k t_k^p_k e^{-i gam_k t_k}, by the allocation formula
    (Theorem 11.5) and Listing 11.1 at the repeated nodes."""
    q = len(gam)
    x = [sum(gam[j:]) for j in range(q)]       # x_j = gam_{j+1}+...
    terms = {(0,) * q: 1.0}         # multiplicity vector M -> weight
    for k in range(1, q + 1):       # event k gives p_k to j < k
        new = {}
        for M, w in terms.items():
            for m in compositions(p[k - 1], k):
                M2 = tuple(M[j] + (m[j] if j < k else 0)
                           for j in range(q))
                c = math.factorial(p[k - 1]) / math.prod(
                    math.factorial(a) for a in m)
                new[M2] = new.get(M2, 0.0) + w * c
        terms = new
    total = 0j
    for M, w in terms.items():
        nodes = [xj for xj, Mj in zip(x, M) for _ in range(Mj + 1)]
        mid, th, G = dd_expi_squaring(nodes + [0.0], T)
        n = len(nodes)
        w *= math.prod(math.factorial(a) for a in M)
        total += w * (T / th)**n * np.exp(-1j * T * mid) * G[0, n]
    return total


# ---- Listing 11.3 -----------------------------------------------------------
def dyson_driven(D, Vs, ws, t, x, y, Q):
    """<y|U(t)|x> for H(t) = diag(D) + sum_r e^(-i ws[r] t) Vs[r],
    summed over the event sequences of length <= Q (Thm 11.16)."""
    c = (D.max() + D.min()) / 2            # centre of the nodes
    rho = t * (np.ptp(D) / 2 + Q * max(map(abs, ws)))
    M = math.ceil(math.e * rho + 37)       # Lemma 9.1
    total = 0j

    def dfs(z, q, amp, P, w):
        nonlocal total
        if z == y:            # amp e^{-it[E_u - P]} e^{-itP}
            total += (amp * (-1j * t)**q * w.sum() / math.factorial(q)
                      * np.exp(-1j * t * P))
        if q == Q:
            return
        for V, om in zip(Vs, ws):
            for z2 in np.nonzero(V[:, z])[0]:
                dfs(z2, q + 1, amp * V[z2, z], P + om,
                    dd_push(w, q, -1j * t * (D[z2] - P - om - c)))

    w0 = np.zeros(M + 1, dtype=complex)
    dfs(x, 0, 1.0, 0.0, dd_push(w0, -1, -1j * t * (D[x] - c)))
    return np.exp(-1j * t * c) * total


# ===================================================================== helpers
def mp_ddexp(z, dps=None):
    """e^{[z_0..z_q]} for complex nodes (repeats allowed), in mpmath, by
    the Taylor series about the centre (Theorem 5.1)."""
    q = len(z) - 1
    zz = [mp.mpc(v) for v in z]
    c = sum(zz) / (q + 1)
    r = max(abs(v - c) for v in zz)
    if dps is None:
        dps = int(40 + float(r) / 2.3)
    with mp.workdps(dps):
        y = [v - c for v in zz]
        K = q + int(math.e * float(r)) + 60
        coef = [mp.mpf(1)]
        for m in range(1, K + 1):
            coef.append(coef[-1] / m)
        v = [mp.mpc(0)] * (q + 1)
        for cm in reversed(coef):
            for a in range(q):
                v[a] = y[a] * v[a] + v[a + 1]
            v[q] = y[q] * v[q] + cm
        return +(mp.exp(c) * v[0])


def mp_ddc(T, z, dps=None):
    """e^{T[z]} = T^q e^{[Tz]}."""
    q = len(z) - 1
    return mp.mpf(T)**q * mp_ddexp([mp.mpf(T) * mp.mpc(v) for v in z], dps)


def expi_from_listing(E, t, i=0, j=None):
    """e^{-it[E_i..E_j]} from Listing 11.1 (complex double)."""
    mid, th, G = dd_expi_squaring(E, t)
    j = len(E) - 1 if j is None else j
    d = j - i
    return (-1j * t / th)**d * np.exp(-1j * t * mid) * G[i, j]


class EP:
    """Exponential polynomial sum c t^p e^{mu t}, exact in mpmath when the
    rates are dyadic."""

    def __init__(self, terms=None):
        self.t = dict(terms or {})

    @staticmethod
    def key(mu):
        mu = mp.mpc(mu)
        return (mu.real, mu.imag)

    def add(self, mu, p, c):
        k = (EP.key(mu), p)
        self.t[k] = self.t.get(k, 0) + c

    def mul_exp_poly(self, lam, poly):
        """multiply by e^{lam t} sum_p poly[p] t^p"""
        out = EP()
        for ((mr, mi), p), c in self.t.items():
            for pp, cc in poly.items():
                if cc:
                    out.add(mp.mpc(mr, mi) + lam, p + pp, c * cc)
        return out

    def integrate(self):
        """t -> int_0^t (self)(s) ds"""
        out = EP()
        for ((mr, mi), p), c in self.t.items():
            mu = mp.mpc(mr, mi)
            if mu == 0:
                out.add(0, p + 1, c / (p + 1))
                continue
            a0 = None
            for k in range(p + 1):
                a = ((-1)**(p - k) * mp.factorial(p)
                     / (mp.factorial(k) * mu**(p - k + 1)))
                out.add(mu, k, c * a)
                if k == 0:
                    a0 = a
            out.add(0, 0, -c * a0)
        return out

    def __call__(self, T):
        T = mp.mpf(T)
        return mp.fsum(c * T**p * mp.exp(mp.mpc(mr, mi) * T)
                       for ((mr, mi), p), c in self.t.items())


def ordered_exact(lams, polys, T=None):
    """int_{Delta_q(T)} prod_k poly_k(t_k) e^{lam_k t_k} dt as an EP in T
    (or its value at T)."""
    g = EP({((mp.mpf(0), mp.mpf(0)), 0): mp.mpf(1)})
    for lam, poly in zip(lams, polys):
        g = g.mul_exp_poly(mp.mpc(lam), poly).integrate()
    return g if T is None else g(T)


def dyadic(rng, scale=8):
    return mp.mpc(rng.randint(-24, 24) / scale, rng.randint(-24, 24) / scale)


mp.mp.dps = 50
# ============================================================== 11.1 ordered simplices
print("--- 11.1 Ordered simplices")
rng = random.Random(11)
worst = 0
for trial in range(6):
    q = rng.choice([1, 2, 3, 4, 5])
    lams = [dyadic(rng) for _ in range(q)]
    T = mp.mpf(rng.choice([0.5, 1.25, 2]))
    lhs = ordered_exact(lams, [{0: 1}] * q, T)
    S = [mp.fsum(lams[j:]) for j in range(q)] + [mp.mpc(0)]
    rhs = mp_ddc(T, S)
    worst = max(worst, float(abs(lhs - rhs) / abs(rhs)))
report("Thm 11.2: ordered exponential integral = e^T[S_0..S_q]",
       worst < 1e-35, f"{worst:.0e}")

# Corollary 11.3, oscillatory form, with Listing 11.1 in double
worst = 0
for trial in range(6):
    q = rng.choice([1, 2, 3, 4])
    gam = [rng.randint(-16, 16) / 8 for _ in range(q)]
    t = rng.choice([0.5, 1.0, 3.0])
    lhs = (-1j)**q * complex(ordered_exact([mp.mpc(0, -g) for g in gam],
                                           [{0: 1}] * q, t))
    x = [sum(gam[j:]) for j in range(q)] + [0.0]
    worst = max(worst, abs(lhs - expi_from_listing(x, t)))
report("Cor 11.3: (-i)^q int e^{-i sum gam t} = e^{-it}[x_0..x_q]",
       worst < 1e-14, f"{worst:.0e}")

# Proposition 11.4: segment lengths, q = 2, m = (1, 0, 2)
x = [mp.mpc(0.25, 1), mp.mpc(-0.5, 0.5), mp.mpc(0.75, -1)]
m = [1, 0, 2]
T = mp.mpf(1.5)
with mp.workdps(25):
    f = lambda u0, u1: (u0**m[0] / mp.factorial(m[0]) * mp.exp(x[0] * u0)
                        * u1**m[1] / mp.factorial(m[1]) * mp.exp(x[1] * u1)
                        * (T - u0 - u1)**m[2] / mp.factorial(m[2])
                        * mp.exp(x[2] * (T - u0 - u1)))
    lhs = mp.quad(lambda u0: mp.quad(lambda u1: f(u0, u1), [0, T - u0]), [0, T])
rhs = mp_ddc(T, [x[0]] * 2 + [x[1]] + [x[2]] * 3)
report("Prop 11.4: segment-length powers raise multiplicities",
       abs(lhs - rhs) / abs(rhs) < 1e-18, f"{float(abs(lhs - rhs) / abs(rhs)):.0e}")


def allocation_mp(lams, p, T):
    """allocation formula of Theorem 11.5 in mpmath"""
    q = len(lams)
    S = [mp.fsum(lams[j:]) for j in range(q)] + [mp.mpc(0)]
    total = 0
    count = {}
    for alloc in itertools.product(*[list(compositions(p[k - 1], k))
                                     for k in range(1, q + 1)]):
        M = [sum(alloc[k - 1][j] for k in range(j + 1, q + 1)) for j in range(q)]
        c = mp.mpf(1)
        for k in range(1, q + 1):
            c *= mp.factorial(p[k - 1]) / mp.fprod(mp.factorial(a) for a in alloc[k - 1])
        c *= mp.fprod(mp.factorial(a) for a in M)
        count[tuple(M)] = count.get(tuple(M), 0) + c
    for M, c in count.items():
        nodes = [S[j] for j in range(q) for _ in range(M[j] + 1)] + [S[q]]
        total += c * mp_ddc(T, nodes)
    return total, count


lams = [mp.mpc(0.5, -1), mp.mpc(-0.25, 0.75), mp.mpc(1, 0.5)]
p = [1, 0, 2]
T = mp.mpf(1.25)
exact = ordered_exact(lams, [{1: 1}, {0: 1}, {2: 1}], T)
alloc, _ = allocation_mp(lams, p, T)
report("Thm 11.5: allocation formula, q = 3, p = (1, 0, 2)",
       abs(exact - alloc) / abs(exact) < 1e-35,
       f"{float(abs(exact - alloc) / abs(exact)):.0e}")
# worked example q = 2, weight t_2^2
l2 = [mp.mpc(0.5, 0.25), mp.mpc(-0.75, 1)]
S = [l2[0] + l2[1], l2[1], mp.mpc(0)]
ex = ordered_exact(l2, [{0: 1}, {2: 1}], T)
rhs = 2 * (mp_ddc(T, [S[0]] * 3 + [S[1], S[2]]) + mp_ddc(T, [S[0]] * 2 + [S[1]] * 2 + [S[2]])
           + mp_ddc(T, [S[0]] + [S[1]] * 3 + [S[2]]))
report("Thm 11.5: example q = 2, weight t_2^2 (three terms)",
       abs(ex - rhs) / abs(ex) < 1e-35)
# derivative form: d^p/d lam^p e^T[S(lam)], central differences in mpmath


def F_lam(a, c):
    lm = [mp.mpc(a, lams[0].imag), lams[1], mp.mpc(c, lams[2].imag)]
    return mp_ddc(T, [lm[0] + lm[1] + lm[2], lm[1] + lm[2], lm[2], mp.mpc(0)],
                  dps=80)


h = mp.mpf(10)**-10
a0, c0 = lams[0].real, lams[2].real
der = (F_lam(a0 + h, c0 + h) - 2 * F_lam(a0 + h, c0) + F_lam(a0 + h, c0 - h)
       - F_lam(a0 - h, c0 + h) + 2 * F_lam(a0 - h, c0) - F_lam(a0 - h, c0 - h)) / (2 * h**3)
report("Thm 11.5: = d^p/dlam^p e^T[S(lam)] (multishift form)",
       abs(der - exact) / abs(exact) < 1e-18, f"{float(abs(der - exact) / abs(exact)):.0e}")

# Listing 11.2 against the exact ordered integral
worst = 0
for trial in range(8):
    q = rng.choice([1, 2, 3, 4])
    gam = [rng.randint(-16, 16) / 8 for _ in range(q)]
    p = [rng.randint(0, 2) for _ in range(q)]
    if trial == 0:
        gam = [0.5, -0.5, 0.5, 0.0][:q]          # repeated suffix sums
    t = rng.choice([0.5, 1.0, 2.0])
    ex = complex(ordered_exact([mp.mpc(0, -g) for g in gam],
                               [{pk: 1} for pk in p], t))
    got = ordered_poly_integral(gam, p, t)
    scale = t**(q + sum(p)) / math.factorial(q)
    worst = max(worst, abs(got - ex) / scale)
report("Listing 11.2: error / (T^(q+|p|)/q!) small, 8 cases",
       worst < 1e-13, f"{worst:.0e}")

# Proposition 11.6: multiplicity patterns and Fuss-Catalan count
ok = True
for P in range(0, 3):
    for q in range(1, 6):
        pats = set()
        for p in itertools.product(range(P + 1), repeat=q):
            for alloc in itertools.product(*[list(compositions(p[k - 1], k))
                                             for k in range(1, q + 1)]):
                pats.add(tuple(sum(alloc[k - 1][j] for k in range(j + 1, q + 1))
                               for j in range(q)))
        char = {M for M in itertools.product(range(P * q + 1), repeat=q)
                if all(sum(M[r:]) <= P * (q - r) for r in range(q))}
        fc = math.comb((P + 1) * (q + 1), q + 1) // (P * (q + 1) + 1)
        ok &= (pats == char and len(pats) == fc)
report("Prop 11.7: patterns = characterization, count = Fuss-Catalan", ok,
       "P <= 2, q <= 5")

# Proposition 11.7: dimension of the span, P = 1, q = 7
q = 7
lam7 = [mp.mpc(0.5, 0.25 * k) - mp.mpc(0.125 * k * k, 0) for k in range(1, q + 1)]
keys, rows = set(), []
t0 = time.time()
for p in itertools.product(range(2), repeat=q):
    g = ordered_exact(lam7, [{pk: 1} for pk in p])
    g.t = {k: c for k, c in g.t.items() if abs(c) > mp.mpf(10)**-40}
    keys |= set(g.t)
    rows.append(g)
klist = sorted(keys, key=str)
npat = len({tuple(sum(alloc[k - 1][j] for k in range(j + 1, q + 1)) for j in range(q))
            for p in itertools.product(range(2), repeat=q)
            for alloc in itertools.product(*[list(compositions(p[k - 1], k))
                                             for k in range(1, q + 1)])})
report("Prop 11.8: P = 1, q = 7: 1430 patterns, basis size <= 36",
       npat == 1430 and len(klist) <= 36,
       f"patterns {npat}, basis functions used {len(klist)}")

# ============================================================== 11.2 integral-free Dyson
print("--- 11.2 Integral-free Dyson series")
rng_np = np.random.default_rng(3)
n = 3
D = np.array([0.0, 0.7, 1.9])
V = rng_np.standard_normal((n, n)) + 1j * rng_np.standard_normal((n, n))
V = 0.3 * (V + V.conj().T)
np.fill_diagonal(V, 0)
H = np.diag(D) + V
ok, info = True, []
G1 = np.abs(V).sum(axis=0).max()
for t in [0.5, 1.0]:
    Uex = sl.expm(-1j * t * H)
    for Q in [4, 8, 12]:
        Uq = np.array([[dyson_driven(D, [V], [0.0], t, x, y, Q)
                        for x in range(n)] for y in range(n)])
        err1 = np.abs(Uq - Uex).sum(axis=0).max()      # l1 operator norm
        tail = sum((t * G1)**q / math.factorial(q) for q in range(Q + 1, 120))
        ok &= err1 <= tail * (1 + 1e-9) + 1e-14
        info.append(f"{err1 / tail:.2f}")
report("Thm 11.9/11.11: path sum = expm, l1 error <= tail(t Gamma)", ok,
       "ratios " + " ".join(info))
# operator-norm form with H = D + sum_j D_j P_j (bit flips on 2 qubits)
X = np.array([[0, 1], [1, 0]])
I2 = np.eye(2)
Ps = [np.kron(X, I2), np.kron(I2, X), np.kron(X, X)]
Ds = [np.diag(rng_np.uniform(-0.4, 0.4, 4)) for _ in Ps]
D0 = np.diag([0.0, 0.6, 1.1, 2.0])
H = D0 + sum(Dj @ Pj for Dj, Pj in zip(Ds, Ps))
Vop = H - D0
Gam = sum(np.abs(np.diag(Dj)).max() for Dj in Ds)
t, ok, rat = 1.0, True, []
Uex = sl.expm(-1j * t * H)
for Q in [3, 6, 9]:
    Uq = np.array([[dyson_driven(np.diag(D0).copy(), [Vop], [0.0], t, x, y, Q)
                    for x in range(4)] for y in range(4)])
    err = np.linalg.norm(Uq - Uex, 2)
    tail = sum((t * Gam)**q / math.factorial(q) for q in range(Q + 1, 120))
    ok &= err <= tail
    rat.append(f"{err / tail:.2f}")
report("Thm 11.11: operator-norm error <= tail(t sum ||D_j||)", ok,
       "ratios " + " ".join(rat))


def U2_series(a, b, g, t, Q):
    """2x2 propagator of [[a, g], [g, b]] from the integral-free Dyson
    series up to order Q, weights from Listing 11.1; also returns the
    largest l1 sum of the terms of a column."""
    U, ell1 = np.zeros((2, 2), complex), 0.0
    for col in (0, 1):
        nodes = [(a, b)[(col + r) % 2] for r in range(Q + 1)]
        mid, th, G = dd_expi_squaring(nodes, t)
        w = [(-1j * t / th)**q * np.exp(-1j * t * mid) * G[0, q] * g**q
             for q in range(Q + 1)]
        for q in range(Q + 1):
            U[(col + q) % 2, col] += w[q]
        ell1 = max(ell1, sum(abs(v) for v in w))
    return U, ell1


print("    two-level system [[0, g], [g, Delta]], g = 1:")
print("    Delta  g t   Q  one-shot  u*l1sum   sliced (m = ceil(g t))")
TAB1 = []
for Delta in [0.0, 1.0]:
    for gt in [1, 4, 16, 32, 48]:
        g, t = 1.0, float(gt)
        Hm = np.array([[0.0, g], [g, Delta]])
        Uex = sl.expm(-1j * t * Hm)
        Q = int(math.e * g * t + 40)
        U1, l1 = U2_series(0.0, Delta, g, t, Q)
        mslice = max(1, math.ceil(g * t))
        Us, _ = U2_series(0.0, Delta, g, t / mslice, 43)
        Um = np.linalg.matrix_power(Us, mslice)
        e1, es = np.abs(U1 - Uex).max(), np.abs(Um - Uex).max()
        TAB1.append((Delta, gt, Q, e1, U * l1, mslice, es))
        print(f"    {Delta:4.1f} {gt:4d} {Q:4d}   {e1:.1e}   {U * l1:.1e}   {es:.1e}")
report("one-shot error <= 10 u (l1 sum of terms); grows like e^{gt}",
       all(r[3] <= 10 * r[4] + 1e-15 for r in TAB1)
       and max(r[3] for r in TAB1 if r[1] == 48) > 1e-3)
report("time slicing with g dt <= 1 keeps the error below 1e-14",
       all(r[6] < 1e-14 for r in TAB1))

# ============================================================== 11.3 confluent nodes
print("--- 11.3 Degeneracies and confluent arguments")
# secular terms
ok = True
for k in range(0, 6):
    for E in [0.3, -1.7]:
        t = 2.5
        v = expi_from_listing([E] * (k + 1), t)
        ok &= abs(v - (-1j * t)**k / math.factorial(k) * np.exp(-1j * t * E)) < 1e-15 * max(1, t**k / math.factorial(k))
report("Prop 11.12: e^{-it}[E^(k+1)] = (-it)^k e^{-itE}/k!", ok)
# derivative bound |d/dE_j e^{-it[E]}| <= t^(q+1)/(q+1)!
ok = True
for trial in range(20):
    q = rng.randint(1, 6)
    E = [rng.uniform(-2, 2) for _ in range(q + 1)]
    t = rng.uniform(0.2, 5)
    j = rng.randint(0, q)
    d = expi_from_listing(E + [E[j]], t)
    ok &= abs(d) <= t**(q + 1) / math.factorial(q + 1) * (1 + 1e-12)
report("Prop 11.12: |d/dE_j e^{-it}[E]| <= t^(q+1)/(q+1)!", ok)


def env_err(E, t, i=0, j=None):
    """|Listing 11.1 - reference| / (t^d/d!) for the entry (i, j)"""
    j = len(E) - 1 if j is None else j
    d = j - i
    ref = complex((-1j)**d * mp_ddc(t, [mp.mpc(0, -e) for e in E[i:j + 1]]))
    got = expi_from_listing(E, t, i, j)
    return abs(got - ref) / (t**d / math.factorial(d)), abs(ref) / (t**d / math.factorial(d))


print("    Listing 11.1, error relative to the envelope t^q/q!:")
worst_ratio, rows = 0, []
for q in [1, 3, 10, 30, 100]:
    for ts in [0.1, 1, 10, 100, 1000]:
        E = [0.0, 1.0] + [rng.random() for _ in range(q - 1)]
        e, size = env_err(E, ts)
        k = max(0, math.ceil(math.log2(ts))) if ts > 0 else 0
        K = q + 1 + 40
        bound = 2**k * ((4 * K + 3) * U * math.sqrt(math.e) + (q + 1) * U / (1 - (q + 1) * U))
        worst_ratio = max(worst_ratio, e / bound)
        rows.append((q, ts, e, size))
for q in [1, 10, 100]:
    print("    q=%3d " % q + "  ".join(f"ts={ts:g}: {e:.0e} (|w|/env {s:.0e})"
                                     for (qq, ts, e, s) in rows if qq == q))
report("Thm 11.13: Listing 11.1 error <= 2^k(delta_0 + gamma) envelope", worst_ratio <= 1,
       f"worst ratio {worst_ratio:.1e}, max {max(r[2] for r in rows):.1e}")

print("    near-degenerate nodes, q = 4, t = 2:  explicit formula vs Listing 11.1")
TAB2 = []
for d in [1e-1, 1e-3, 1e-6, 1e-9, 1e-12]:
    E = [0.3 + d * rng.random() for _ in range(4)] + [1.0]
    t = 2.0
    ref = complex((-1j)**4 * mp_ddc(t, [mp.mpc(0, -e) for e in E]))
    expl = sum(np.exp(-1j * t * E[j]) / math.prod(E[j] - E[k] for k in range(5) if k != j)
               for j in range(5))
    env = t**4 / 24
    TAB2.append((d, abs(expl - ref) / env, abs(expi_from_listing(E, t) - ref) / env))
    print(f"    gap {d:.0e}:  explicit {TAB2[-1][1]:.1e}   Listing 11.1 {TAB2[-1][2]:.1e}")
report("explicit formula loses ~u/gap^q; Listing 11.1 stays below 1e-14",
       TAB2[-1][1] > 1 and all(r[2] < 1e-14 for r in TAB2))

# ============================================================== 11.4 time-dependent
print("--- 11.4 Time-dependent Hamiltonians")
# Lemma 11.14 concatenation
worst = 0
for trial in range(3):
    a = [mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(rng.randint(1, 3))]
    b = [mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(rng.randint(1, 3))]
    if trial == 2:
        b = [a[0]] + b                                     # shared node
    T = mp.mpf(1.3)
    with mp.workdps(25):
        lhs = mp.quad(lambda tau: mp_ddc(tau, a, 30) * mp_ddc(T - tau, b, 30), [0, T])
    rhs = mp_ddc(T, a + b)
    worst = max(worst, float(abs(lhs - rhs) / abs(rhs)))
report("Lemma 11.14: int_0^T e^tau[a] e^(T-tau)[b] dtau = e^T[a,b]", worst < 1e-18,
       f"{worst:.0e}")
# Corollary 11.14 input table
Dn = [mp.mpc(-0.4, 0.3), mp.mpc(0.2, -0.5), mp.mpc(-1, 0)]
Dr = [mp.mpf(-0.4), mp.mpf(0.2), mp.mpf(-1.1)]
t, gm, Om, tau, mm = mp.mpf(1.7), mp.mpc(0.3, 0.8), mp.mpf(2.1), mp.mpf(0.6), 2
with mp.workdps(25):
    conv = lambda Dl, u, a=0: mp.quad(lambda s: mp_ddc(t - s, Dl, 30) * u(s), [a, t])
    rows_ok = [
        abs(conv(Dn, lambda s: 1) - mp_ddc(t, Dn + [0])),
        abs(conv(Dn, lambda s: s**mm / mp.factorial(mm)) - mp_ddc(t, Dn + [0] * (mm + 1))),
        abs(conv(Dn, lambda s: mp.exp(gm * s)) - mp_ddc(t, Dn + [gm])),
        abs(conv(Dn, lambda s: s**mm * mp.exp(gm * s) / mp.factorial(mm))
            - mp_ddc(t, Dn + [gm] * (mm + 1))),
        abs(conv(Dr, lambda s: mp.cos(Om * s)) - mp_ddc(t, Dr + [mp.mpc(0, Om)]).real),
        abs(conv(Dn, lambda s: (s - tau)**mm / mp.factorial(mm), tau)
            - mp_ddc(t - tau, Dn + [0] * (mm + 1))),
    ]
    zero = abs(mp.quad(lambda s: mp_ddc(s, Dn, 30), [0, t]) - mp_ddc(t, [0] + Dn))
    scal = abs(mp.quad(lambda a: a**len(Dn[1:]) * mp_ddexp([a * v for v in Dn], 30), [0, 1])
               - mp_ddexp([0] + Dn))
report("Cor 11.15: all rows of the input table", max(rows_ok) < 1e-18,
       f"max {float(max(rows_ok)):.0e}")
report("Cor 11.15: time integral appends 0; scaling integral",
       zero < 1e-18 and scal < 1e-18)

# Theorem 11.15 rate-shifted nodes (exact, q = 3) and forced response
Eu = [mp.mpc(0.25, -1), mp.mpc(-0.5, 0.5), mp.mpc(0, 0.25), mp.mpc(0.5, 0)]
lr = [mp.mpc(0, 0.75), mp.mpc(-0.25, -0.5), mp.mpc(0.125, 1)]
q = 3
t = mp.mpf(1.5)
c = EP({((mp.mpf(0), mp.mpf(0)), 0): mp.mpf(1)})
c = EP({(EP.key(Eu[0]), 0): mp.mpf(1)})          # c_0 = e^{E_u0 t}


def conv_ep(Er, lam, cprev):
    """t -> int_0^t e^{Er (t - s)} e^{lam s} cprev(s) ds, exactly"""
    g = cprev.mul_exp_poly(lam - Er, {0: 1}).integrate()
    return g.mul_exp_poly(Er, {0: 1})


for r in range(1, q + 1):
    c = conv_ep(Eu[r], lr[r - 1], c)
y = [Eu[j] + mp.fsum(lr[j:]) for j in range(q + 1)]
rel = abs(c(t) - mp_ddc(t, y)) / abs(c(t))
gm = mp.mpc(-0.375, 0.25)
LamP = mp.fsum(lr)
forced = c.mul_exp_poly(mp.mpc(0), {0: 1})
# int_0^t e^{s LamP} e^{(t-s)[y]} e^{gm s} ds = e^t[y, gm + LamP]
with mp.workdps(25):
    lhs = mp.quad(lambda s: mp.exp(s * LamP) * mp_ddc(t - s, y, 30) * mp.exp(gm * s), [0, t])
rel2 = abs(lhs - mp_ddc(t, y + [gm + LamP])) / abs(lhs)
report("Thm 11.16: c_q(t) = e^t[y_0..y_q], y_j = E_uj + sum_{r>j} lam_r",
       rel < 1e-35, f"{float(rel):.0e}")
report("Thm 11.16: forced response appends gamma + Lambda (not gamma)",
       rel2 < 1e-18, f"{float(rel2):.0e}")

# Listing 11.3 for a driven three-level system; Corollary 11.16 (Sambe)
D3 = np.array([0.0, 1.0, 2.3])
V0 = np.array([[0, 0.2, 0], [0.2, 0, 0.1], [0, 0.1, 0]], float)
X3 = np.zeros((3, 3))
X3[0, 1] = X3[1, 0] = 0.5
Om3 = 0.9
Vs, ws = [V0, X3, X3], [0.0, Om3, -Om3]


def H3(s):
    return np.diag(D3) + V0 + 2 * X3 * np.cos(Om3 * s)


def ode_ref(t, x):
    v0 = np.zeros(3, complex)
    v0[x] = 1
    sol = solve_ivp(lambda s, v: -1j * (H3(s) @ v), (0, t), v0, method="DOP853",
                    rtol=1e-13, atol=1e-15)
    return sol.y[:, -1]


def sambe_ref(t, K=40):
    nk = 2 * K + 1
    ks = np.arange(-K, K + 1)
    HF = (np.kron(np.diag(D3), np.eye(nk)) - np.kron(np.eye(3), np.diag(ks * Om3))
          + np.kron(V0, np.eye(nk)) + np.kron(X3, np.eye(nk, k=-1))
          + np.kron(X3, np.eye(nk, k=1)))
    UF = sl.expm(-1j * t * HF)
    out = np.zeros((3, 3), complex)
    for x in range(3):
        for yy in range(3):
            out[yy, x] = sum(np.exp(-1j * t * k * Om3) * UF[yy * nk + k + K, x * nk + K]
                             for k in ks)
    return out


Us = sambe_ref(2.0)
d_ode = max(np.abs(Us[:, x] - ode_ref(2.0, x)).max() for x in range(3))
report("Cor 11.17: Sambe-space expm = time-ordered propagator (ODE)", d_ode < 1e-11,
       f"{d_ode:.0e}")
Gam3 = sum(np.abs(Vm).sum(axis=0).max() for Vm in Vs)
print("    driven 3-level system, <1|U(t)|0>; error and tail bound vs Q:")
TAB3 = []
for t in [0.25, 0.5, 1.0]:
    Uref = sambe_ref(t)
    for Q in [2, 4, 6, 8]:
        t0 = time.time()
        val = dyson_driven(D3, Vs, ws, t, 0, 1, Q)
        el = time.time() - t0
        tail = sum((t * Gam3)**q / math.factorial(q) for q in range(Q + 1, 100))
        TAB3.append((t, Q, abs(val - Uref[1, 0]), tail, el))
    print("    t=%.2f " % t + "  ".join(f"Q={Q}: {e:.0e}/{b:.0e}"
                                      for (tt, Q, e, b, _) in TAB3 if tt == t))
report("Listing 11.3: error <= tail bound sum_{q>Q} (t Gamma)^q/q!",
       all(e <= b + 2e-15 for (_, _, e, b, _) in TAB3),
       "bound/error %.1f-%.1f" % (min(b / e for (_, _, e, b, _) in TAB3),
                                  max(b / e for (_, _, e, b, _) in TAB3)))

# Proposition 11.17 L1 bound for polynomial time dependence
ok, ratios = True, []
for trial in range(30):
    q = rng.randint(1, 3)
    P = rng.randint(1, 2)
    dt = rng.uniform(0.3, 2.0)
    al = [rng.uniform(-3, 3) for _ in range(q)]
    cs = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(P + 1)]
    Lam = 0.0
    for p in itertools.product(range(P + 1), repeat=q):
        val = ordered_poly_integral([-a for a in al], list(p), dt)
        Lam += math.prod(abs(cs[pk]) for pk in p) * abs(val)
    B = sum(abs(cs[pp]) * dt**pp for pp in range(P + 1))
    bound = (B * dt)**q / math.factorial(q)
    ok &= Lam <= bound * (1 + 1e-12)
    ratios.append(Lam / bound)
report("Prop 11.18: Lambda_q <= (B_P dt)^q / q!", ok, f"max ratio {max(ratios):.2f}")
# term-by-term: sum_M coefficient * T^n/n! equals the zero-rate integral
ok = True
for p in [(1, 0, 2), (2, 2), (1, 1, 1, 1)]:
    q = len(p)
    _, cnt = allocation_mp([mp.mpc(0)] * q, list(p), 1)
    n = q + sum(p)
    s = mp.fsum(cnt.values()) / mp.factorial(n)
    z = ordered_exact([mp.mpc(0)] * q, [{pk: 1} for pk in p], 1)
    ok &= abs(s - z) < 1e-40 and s <= 1 / mp.factorial(q)
report("Prop 11.18: sum of termwise bounds = zero-rate integral <= 1/q!", ok)

# Proposition 11.18 piecewise composition, M = 2, q = 2, s_2(t) = t
lp = [mp.mpc(0.5, -0.25), mp.mpc(-0.75, 0.5)]
tau1, T = mp.mpf(0.625), mp.mpf(1.5)
full = ordered_exact(lp, [{0: 1}, {1: 1}], T)


def piece(ks, a, length):
    """ordered integral over Delta_{|ks|}(length) of the events ks, with
    the weights shifted by a: s_k(a + u)"""
    polys = []
    for k in ks:
        polys.append({0: 1} if k == 0 else {0: a, 1: 1})
    return ordered_exact([lp[k] for k in ks], polys, length) if ks else mp.mpf(1)


tot = 0
for m1, m2 in [(1, 1), (1, 2), (2, 2)]:
    K1 = [k for k, mk in zip((0, 1), (m1, m2)) if mk == 1]
    K2 = [k for k, mk in zip((0, 1), (m1, m2)) if mk == 2]
    L2 = mp.fsum(lp[k] for k in K2)
    tot += piece(K1, 0, tau1) * mp.exp(tau1 * L2) * piece(K2, tau1, T - tau1)
report("Prop 11.19: piecewise composition (M = 2, q = 2)",
       abs(tot - full) / abs(full) < 1e-35)

# ============================================================== 11.5 algebra
print("--- 11.5 Algebraic structure")
xs = [mp.mpc(-0.3, 1), mp.mpc(0.2, -0.4), mp.mpc(-1, 0), mp.mpc(0.2, -0.4)]
s0 = mp.mpc(1.5, 0.2)
with mp.workdps(25):
    lap = mp.quad(lambda T: mp.exp(-s0 * T) * mp_ddc(T, xs, 30), [0, 5, 20, 40, 70])
rel = abs(lap - 1 / mp.fprod(s0 - v for v in xs)) * abs(mp.fprod(s0 - v for v in xs))
report("Prop 11.20: Laplace transform = prod 1/(s - x_j) (repeated node)", rel < 1e-18,
       f"{float(rel):.0e}")
# Born series: Laplace transform of the Dyson series = resolvent expansion
Hb = np.diag([0.0, 1.0, 2.5]) + 0.2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
sres = 4.0 + 0.5j
R0 = np.diag(1 / (sres - np.diag(Hb)))
Vb = Hb - np.diag(np.diag(Hb))
born = sum(np.linalg.matrix_power(R0 @ Vb, q) @ R0 for q in range(80))
report("Prop 11.20: Born series sums to (s - H)^{-1}",
       np.abs(born - np.linalg.inv(sres * np.eye(3) - Hb)).max() < 1e-14)

# Theorem 11.20 shuffle
a, b, T = mp.mpc(0.3, -0.7), mp.mpc(-0.5, 0.25), mp.mpf(1.4)
lhs = mp_ddc(T, [a, 0]) * mp_ddc(T, [b, 0])
rhs = mp_ddc(T, [a + b, b, 0]) + mp_ddc(T, [a + b, a, 0])
report("Thm 11.21: e^T[a,0] e^T[b,0] = e^T[a+b,b,0] + e^T[a+b,a,0]",
       abs(lhs - rhs) < 1e-40)


def phi(beta, E):
    """(-1)^q e^{-beta[E]} >= 0"""
    q = len(E) - 1
    return (-1)**q * mp_ddc(-beta, E)


def shuffle_check(lists, beta):
    m = len(lists)
    qs = [len(L) - 1 for L in lists]
    letters = [b for b in range(m) for _ in range(qs[b])]
    total = 0
    for word in set(itertools.permutations(letters)):
        k = [0] * m
        E = [mp.fsum(lists[b][0] for b in range(m))]
        for s in word:
            k[s] += 1
            E.append(mp.fsum(lists[b][k[b]] for b in range(m)))
        total += phi(beta, E)
    prod = mp.fprod(phi(beta, L) for L in lists)
    return abs(total - prod) / abs(prod)


r1 = shuffle_check([[mp.mpf(0.3), mp.mpf(-0.4), mp.mpf(1.1)], [mp.mpf(0.5), mp.mpf(-0.2)]], 1.7)
r2 = shuffle_check([[mp.mpf(0.3), mp.mpf(0.3), mp.mpf(1.1)], [mp.mpf(0.5), mp.mpf(0.5)],
                    [mp.mpf(-1), mp.mpf(0.25)]], 2.3)
report("Thm 11.21: exponential shuffle identity (m = 2, 3; coalescing)",
       r1 < 1e-35 and r2 < 1e-35, f"{float(max(r1, r2)):.0e}")
# composite system: walk weights of H1 x 1 + 1 x H2 factorize
H1 = np.array([[0.0, 0.3], [0.3, 1.0]])
H2 = np.array([[0.2, 0.4], [0.4, -0.5]])
Hc = np.kron(H1, np.eye(2)) + np.kron(np.eye(2), H2)
beta = 1.3
report("Thm 11.21: e^{-beta(H1+H2)} = e^{-beta H1} (x) e^{-beta H2} (sanity)",
       np.abs(sl.expm(-beta * Hc) - np.kron(sl.expm(-beta * H1), sl.expm(-beta * H2))).max() < 1e-14)

# Proposition 11.21 time reversal and permutation symmetry
lamr = [mp.mpc(0.25, -0.5), mp.mpc(-0.75, 0.125), mp.mpc(0.5, 1), mp.mpc(-0.125, -0.25)]
T = mp.mpf(1.25)
fwd = ordered_exact(lamr, [{0: 1}] * 4, T)
bwd = ordered_exact([-v for v in lamr[::-1]], [{0: 1}] * 4, T)
report("Prop 11.22: time reversal I(lam) = e^{T sum lam} I(-lam reversed)",
       abs(fwd - mp.exp(T * mp.fsum(lamr)) * bwd) / abs(fwd) < 1e-35)
Sx = [mp.fsum(lamr[j:]) for j in range(4)] + [mp.mpc(0)]
ok = True
for perm in random.Random(5).sample(list(itertools.permutations(range(5))), 6):
    S2 = [Sx[i] - Sx[perm[-1]] for i in perm]               # last node 0
    l2 = [S2[k - 1] - S2[k] for k in range(1, 5)]
    val = ordered_exact(l2, [{0: 1}] * 4, T)
    ok &= abs(val * mp.exp(T * Sx[perm[-1]]) - fwd) / abs(fwd) < 1e-35
report("Prop 11.22: any permutation of the suffix nodes (shifted)", ok)
# general reversal J_T(a_1..a_m) = J_T(abar_m..abar_1) with polynomial letters
a1 = (mp.mpc(0.5, -0.25), {0: 1, 2: 1})       # (1 + t^2) e^{lam t}
a2 = (mp.mpc(-0.25, 0.5), {1: 1})
T = mp.mpf(1.5)
fw = ordered_exact([a1[0], a2[0]], [a1[1], a2[1]], T)


def reflect(letter):
    lam, poly = letter         # a(T - t) = e^{lam T} e^{-lam t} poly(T - t)
    newp = {}
    for p_, c_ in poly.items():
        for k in range(p_ + 1):
            newp[k] = newp.get(k, 0) + c_ * mp.binomial(p_, k) * T**(p_ - k) * (-1)**k
    return -lam, newp, mp.exp(lam * T)


r2_, r1_ = reflect(a2), reflect(a1)
bw = ordered_exact([r2_[0], r1_[0]], [r2_[1], r1_[1]], T) * r1_[2] * r2_[2]
report("Prop 11.22: J_T(a_1, a_2) = J_T(abar_2, abar_1)", abs(fw - bw) / abs(fw) < 1e-35)
# Corollary 11.22 exponential-weight convolution
xl = [mp.mpf(0.3), mp.mpf(-0.2), mp.mpf(1.0)]
yl = [mp.mpf(0.5), mp.mpf(0.1)]
lam, T = mp.mpc(0.4, 0.7), mp.mpf(1.6)
with mp.workdps(25):
    lhs = mp.quad(lambda tau: mp.exp(lam * tau) * mp_ddc(-tau, yl, 30)
                  * mp_ddc(-(T - tau), xl, 30), [0, T])
rhs = -mp_ddc(-T, xl + [v - lam for v in yl])
report("Cor 11.23: int e^{lam tau} e^{-tau}[y] e^{-(T-tau)}[x] = -e^{-T}[x, y-lam]",
       abs(lhs - rhs) / abs(rhs) < 1e-18, f"{float(abs(lhs - rhs) / abs(rhs)):.0e}")

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
