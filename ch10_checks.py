"""Numerical checks for Chapter 10 (matrix functions and operators).

Run:  python3 code/ch10_checks.py

The functions loewner_exp, frechet_hermitian, frechet2_hermitian,
leja_interval, newton_fAv, walk_element, matrix_dd_exp and
matrix_dd_uniformized are the listings of the chapter, copied verbatim.
Listings 9.1, 9.2 and the update formula of Proposition 9.9 are reused
from code/ch09_checks.py.  Double precision is IEEE binary64
(u = 2^-53); references use mpmath.
"""
import math
import random
import time
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy.linalg as sl

warnings.filterwarnings("ignore")
U = 2.0**-53
RESULTS = []
HERE = Path(__file__).resolve().parent


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:64s} {info}")


# Listings 9.1 and 9.2 (and Listing 8.3) from Chapter 9
_ns = {"__file__": str(HERE / "ch09_checks.py")}
exec((HERE / "ch09_checks.py").read_text().split(
    "# ============================================================== 9.1 Taylor")[0], _ns)
dd_exp_taylor, dd_exp_squaring, ref_exp = (
    _ns["dd_exp_taylor"], _ns["dd_exp_squaring"], _ns["ref_exp"])


# ===================================================================== listings
# ---- Listing 10.1 -----------------------------------------------------------
def loewner_exp(lam):
    """L[i, j] = exp[lam_i, lam_j] for real lam, without cancellation:
    exp[a, b] = e^min(a,b) expm1(|a - b|)/|a - b|, eq. (10.4)."""
    a, b = np.meshgrid(lam, lam, indexing="ij")
    d = np.abs(a - b)
    ratio = np.ones_like(d)
    nz = d > 0
    ratio[nz] = np.expm1(d[nz]) / d[nz]
    return np.exp(np.minimum(a, b)) * ratio


def frechet_hermitian(A, E, loewner=loewner_exp):
    """Frechet derivative L_f(A, E) at Hermitian A (Theorem 10.2):
    Q (L o (Q* E Q)) Q*, L the Loewner matrix of the eigenvalues."""
    lam, Q = np.linalg.eigh(A)
    Et = Q.conj().T @ E @ Q
    return Q @ (loewner(lam) * Et) @ Q.conj().T


# ---- Listing 10.2 -----------------------------------------------------------
def frechet2_hermitian(A, E, F, dd3):
    """Second Frechet derivative L^(2)_f(A; E, F) at Hermitian A
    (Corollary 10.5); dd3(a, b, c) must return f[a, b, c]."""
    lam, Q = np.linalg.eigh(A)
    n = len(lam)
    Et, Ft = Q.conj().T @ E @ Q, Q.conj().T @ F @ Q
    T = np.array([[[dd3(lam[i], lam[k], lam[j]) for j in range(n)]
                   for k in range(n)] for i in range(n)])
    S = (np.einsum("ik,kj,ikj->ij", Et, Ft, T)
         + np.einsum("ik,kj,ikj->ij", Ft, Et, T))
    return Q @ S @ Q.conj().T


# ---- Listing 10.3 -----------------------------------------------------------
def leja_interval(a, b, m, ncand=20000):
    """m Leja points for [a, b], starting at b, chosen from a grid of
    Chebyshev candidates (log products as in Listing 8.1)."""
    z = (a + b) / 2 + (b - a) / 2 * np.cos(np.pi * np.arange(ncand)
                                           / (ncand - 1))
    pts = [b]
    logp = np.log(np.abs(z - b) + 1e-300)
    for _ in range(m - 1):
        k = int(np.argmax(logp))
        pts.append(z[k])
        logp += np.log(np.abs(z - z[k]) + 1e-300)
    return np.array(pts)


def newton_fAv(matvec, v, xi, d, tol=1e-12):
    """p(A) v for the Newton interpolant with nodes xi, coefficients
    d[k] = f[xi_0..xi_k]; stops after two successive small terms."""
    r = v.copy()
    w = d[0] * r
    small = 0
    for k in range(1, len(xi)):
        r = matvec(r) - xi[k - 1] * r       # r = omega_k(A) v
        term = d[k] * r
        w = w + term
        small = small + 1 if (np.linalg.norm(term)
                              <= tol * np.linalg.norm(w)) else 0
        if small == 2:
            return w, k
    raise RuntimeError("not converged: use more Leja points")


# ---- Listing 10.4 -----------------------------------------------------------
def dd_push(w, q, y):
    """Proposition 9.9 (ii): state of order q + 1 from the state w of
    order q (q = -1 for the empty list), as a new array."""
    out = np.empty_like(w)
    prev = 0.0
    for m in range(len(w)):
        if q < 0:
            prev = 1.0 if m == 0 else y * prev / m
        else:
            prev = ((q + 1) * w[m] + y * prev) / (m + q + 1)
        out[m] = prev
    return out


def walk_element(H, x, y, beta, Q):
    """<y| e^{-beta H} |x> from the walks x -> y of length <= Q
    (Theorem 10.7), by depth-first search over the entries of H."""
    E = np.diag(H).real
    sigma = -beta * E.max()              # nodes -beta E >= sigma
    M = math.ceil(math.e * beta * np.ptp(E) + 37)   # Lemma 9.1
    total = 0.0

    def dfs(z, q, amp, w):
        nonlocal total
        if z == y:                  # add C f[E_x0..E_xq], f = e^(-bt)
            total += amp * (-beta)**q * w.sum() / math.factorial(q)
        if q == Q:
            return
        for z2 in np.nonzero(H[:, z])[0]:
            if z2 != z:
                dfs(z2, q + 1, amp * H[z2, z],
                    dd_push(w, q, -beta * E[z2] - sigma))

    w0 = dd_push(np.r_[1.0, np.zeros(M)], -1, -beta * E[x] - sigma)
    dfs(x, 0, 1.0, w0)
    return math.exp(sigma) * total


# ---- Listing 10.5 -----------------------------------------------------------
def matrix_dd_exp(As, beta):
    """e^beta[A_1..A_L]: corner block of exp(beta T), Prop. 10.11."""
    L, d = len(As), As[0].shape[0]
    T = np.zeros((L * d, L * d), dtype=np.result_type(*As))
    for j in range(L):
        T[j*d:(j+1)*d, j*d:(j+1)*d] = As[L - 1 - j]
        if j + 1 < L:
            T[j*d:(j+1)*d, (j+1)*d:(j+2)*d] = np.eye(d)
    return sl.expm(beta * T)[:d, -d:]


def matrix_dd_uniformized(As, beta, lam, eps=2.0**-53):
    """e^beta[A_1..A_L] by the uniformized series, Prop. 10.13; all
    terms are entrywise nonnegative if every P_l = I + A_l/lam is."""
    L, d = len(As), As[0].shape[0]
    P = [np.eye(d) + A / lam for A in As]
    rho = max(np.abs(Pl).sum(axis=1).max() for Pl in P)
    Nmax = math.ceil(math.e * lam * beta * rho + math.log(1 / eps))
    S = [np.eye(d)] * L                 # S[l] = S^(l)_N, here N = 0
    W = np.zeros((d, d))
    for N in range(Nmax + 1):
        logc = (-lam * beta + (L - 1) * math.log(beta)
                + N * math.log(lam * beta) - math.lgamma(N + L))
        W = W + math.exp(logc) * S[L - 1]
        new = []      # S^(l)_{N+1} = P_l S^(l)_N + S^(l-1)_{N+1}
        for l in range(L):
            new.append(P[l] @ S[l] + (new[l - 1] if l else 0))
        S = new
    return W


# ===================================================================== helpers
def mp_block_corner(blocks_diag, blocks_sup, beta=1.0, dps=50):
    """corner block of exp(beta T) in mpmath, T upper block bidiagonal"""
    with mp.workdps(dps):
        L, d = len(blocks_diag), blocks_diag[0].shape[0]
        M = mp.matrix(L * d, L * d)
        for j in range(L):
            for a in range(d):
                for b in range(d):
                    M[j*d + a, j*d + b] = mp.mpf(blocks_diag[j][a, b]) * beta
                    if j + 1 < L:
                        M[j*d + a, (j+1)*d + b] = mp.mpf(blocks_sup[j][a, b]) * beta
        F = mp.expm(M)
        return [[F[a, (L-1)*d + b] for b in range(d)] for a in range(d)]


def sym(rng, n):
    X = rng.standard_normal((n, n))
    return (X + X.T) / 2


def with_spectrum(rng, lam):
    Q, _ = np.linalg.qr(rng.standard_normal((len(lam), len(lam))))
    A = Q @ np.diag(lam) @ Q.T
    return (A + A.T) / 2


# ============================================================== 10.1 Daleckii-Krein
print("--- 10.1 Daleckii-Krein")
rng = np.random.default_rng(1)
# polynomial identity behind the theorem: p of [[D, E], [0, D]]
D = np.diag(rng.standard_normal(5)); E = rng.standard_normal((5, 5))
coef = rng.standard_normal(7)
M = np.block([[D, E], [np.zeros((5, 5)), D]])
PM = sum(c * np.linalg.matrix_power(M, k) for k, c in enumerate(coef))
lam = np.diag(D)
pdd = lambda a, b: (sum(c * a**k for k, c in enumerate(coef)) - sum(c * b**k for k, c in enumerate(coef))) / (a - b)
ok = all(abs(PM[i, 5 + j] - E[i, j] * pdd(lam[i], lam[j])) < 1e-10 for i in range(5) for j in range(5) if i != j)
report("block identity: p([[D,E],[0,D]])_12 = E o [p[l_i,l_j]]", ok)

DK = []
for gap in [1e-1, 1e-4, 1e-8, 1e-12]:
    n = 10
    lam = np.sort(np.concatenate([gap * rng.random(5), 1 + gap * rng.random(5)]))
    A = with_spectrum(rng, lam); E = sym(rng, n)
    Rm = mp_block_corner([A, A], [E], dps=40)
    R = np.array([[float(v) for v in row] for row in Rm])
    naive = lambda l: np.where(np.subtract.outer(l, l) == 0, np.exp(np.add.outer(l, 0 * l)),
                               (np.subtract.outer(np.exp(l), np.exp(l))) / np.where(np.subtract.outer(l, l) == 0, 1, np.subtract.outer(l, l)))
    e1 = np.linalg.norm(frechet_hermitian(A, E) - R) / np.linalg.norm(R)
    e2 = np.linalg.norm(frechet_hermitian(A, E, naive) - R) / np.linalg.norm(R)
    e3 = np.linalg.norm(sl.expm_frechet(A, E, compute_expm=False) - R) / np.linalg.norm(R)
    DK.append((gap, e1, e2, e3))
    print(f"    gap={gap:.0e}: Listing 10.1 {e1:.1e}  naive Loewner {e2:.1e}  expm_frechet {e3:.1e}")
report("Listing 10.1 accurate (< 1e-14) for all gaps", max(r[1] for r in DK) < 1e-14)
report("naive Loewner loses accuracy ~ u/gap", DK[-1][2] > 1e-6 and DK[0][2] < 1e-12)

# eigenbasis entries for a widely spread spectrum
with mp.workdps(50):
    n = 8
    lam = np.sort(np.concatenate([-60 + rng.random(3), -30 * rng.random(3), rng.random(2)]))
    A = with_spectrum(rng, lam); E = sym(rng, n)
    ev, EV = mp.eigsy(mp.matrix(A.tolist()))
    idx = sorted(range(n), key=lambda i: ev[i])
    Etm = EV.T * mp.matrix(E.tolist()) * EV
    Ref = np.zeros((n, n))
    for a_ in range(n):
        for b_ in range(n):
            i, j = idx[a_], idx[b_]
            x_, y_ = ev[i], ev[j]
            dd = mp.e**x_ if x_ == y_ else (mp.e**x_ - mp.e**y_) / (x_ - y_)
            Ref[a_, b_] = float(abs(dd * Etm[i, j]))
lamc, Qc = np.linalg.eigh(A)
B_back = np.abs(Qc.T @ frechet_hermitian(A, E) @ Qc)
B_fre = np.abs(Qc.T @ sl.expm_frechet(A, E, compute_expm=False) @ Qc)
B_eig = np.abs(loewner_exp(lamc) * (Qc.T @ E @ Qc))
r_back = (np.abs(B_back - Ref) / Ref).max()
r_fre = (np.abs(B_fre - Ref) / Ref).max()
r_eig = (np.abs(B_eig - Ref) / Ref).max()
print(f"    spectrum in [-60,1]: eigenbasis entries: back-transformed {r_back:.1e}, expm_frechet {r_fre:.1e}, eigenbasis form {r_eig:.1e}")
report("eigenbasis form accurate entrywise, normwise methods not", r_eig < 1e-10 and r_fre > 1e3 and r_back > 1e3)

# condition number = max |f[l_i, l_j]| (normal A, Frobenius norm)
ok = True
for t in range(4):
    n = 5
    lam = np.sort(rng.standard_normal(n) * 2)
    A = with_spectrum(rng, lam)
    K = np.zeros((n * n, n * n))
    for k in range(n * n):
        Eb = np.zeros(n * n); Eb[k] = 1
        K[:, k] = frechet_hermitian(A, Eb.reshape(n, n)).ravel()
    ok &= abs(np.linalg.norm(K, 2) - loewner_exp(lam).max()) < 1e-12 * loewner_exp(lam).max()
report("Prop 10.3: ||L_f(A)||_F = max |f[l_i, l_j]| for Hermitian A", ok)

# Loewner matrices: operator monotone (log, sqrt) PSD; x^2, exp not
lam = np.sort(rng.random(8) * 5 + 0.1)
def loew(f, fp, l):
    a, b = np.meshgrid(l, l, indexing="ij")
    with np.errstate(all="ignore"):
        L = (f(a) - f(b)) / (a - b)
    L[a == b] = fp(a[a == b])
    return L
eig = lambda L: np.linalg.eigvalsh((L + L.T) / 2).min()
ok = eig(loew(np.log, lambda x: 1 / x, lam)) > -1e-12 and eig(loew(np.sqrt, lambda x: 0.5 / np.sqrt(x), lam)) > -1e-12 \
    and eig(loew(np.square, lambda x: 2 * x, lam)) < -1e-3 and eig(loewner_exp(lam)) < -1e-3
report("Loewner matrix PSD for log, sqrt; not for x^2, exp", ok)

# ============================================================== 10.2 higher derivatives
print("--- 10.2 Higher Frechet derivatives")
dd3 = lambda a, b, c: math.exp(dd_exp_taylor([a, b, c]))
ok, worst = True, 0
for t in range(3):
    n = 6
    lam = np.sort(rng.random(n) * 3 - 1)
    if t == 2:
        lam[:3] = lam[0] + 1e-9 * np.arange(3)
    A = with_spectrum(rng, lam); E = sym(rng, n); F = sym(rng, n)
    R = np.zeros((n, n))
    for (X, Y) in [(E, F), (F, E)]:
        Rm = mp_block_corner([A, A, A], [X, Y], dps=40)
        R += np.array([[float(v) for v in row] for row in Rm])
    L2 = frechet2_hermitian(A, E, F, dd3)
    worst = max(worst, np.linalg.norm(L2 - R) / np.linalg.norm(R))
report("Listing 10.2 = mixed second derivative (block 3x3 corner), incl. cluster", worst < 1e-13, f"{worst:.1e}")

# general k-th derivative via block bidiagonal: corner = sum over paths
ok = True
n, k = 4, 3
lam = rng.standard_normal(n)
D = np.diag(lam); Es = [rng.standard_normal((n, n)) for _ in range(k)]
Rm = mp_block_corner([D] * (k + 1), Es, dps=30)
for i in range(n):
    for j in range(n):
        s = 0.0
        for k1 in range(n):
            for k2 in range(n):
                s += Es[0][i, k1] * Es[1][k1, k2] * Es[2][k2, j] * \
                    math.exp(dd_exp_taylor([lam[i], lam[k1], lam[k2], lam[j]]))
        ok &= abs(s - float(Rm[i][j])) < 1e-12 * max(1, abs(s))
report("Thm 10.4: corner of f(block bidiagonal) = path sum of f[...]", ok)

# ============================================================== 10.3 Newton-Leja
print("--- 10.3 Newton and Leja interpolation for f(A)v")


def coef_table(xi, f):
    d = list(f(xi)); out = [d[0]]; n = len(xi)
    for k in range(1, n):
        d = [(d[i + 1] - d[i]) / (xi[i + k] - xi[i]) for i in range(n - k)]
        out.append(d[0])
    return np.array(out)


def coef_exp_accurate(xi, t):
    """f[xi_0..xi_k] for f(x) = e^{-t x}, from the first row of Listing 9.2"""
    mid, th, G = dd_exp_squaring(list(-t * xi))
    k = np.arange(len(xi))
    return (-1.0)**k * np.exp(k * math.log(t) + mid - k * math.log(th) + np.log(G[0]))


n = 400
A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)         # spectrum in (0, 4)
lamA, QA = np.linalg.eigh(A)
v = np.random.default_rng(0).standard_normal(n); v /= np.linalg.norm(v)
mv = lambda x: A @ x
LEJA = []
ok_acc, ok_tab = True, True
for t in [1, 5, 25, 100, 300, 1000, 3000]:
    m = int(12 * math.sqrt(t)) + 40
    xi = leja_interval(0.0, 4.0, m)
    ref = QA @ (np.exp(-t * lamA) * (QA.T @ v))
    dt = coef_table(xi, lambda z: np.exp(-t * z))
    wt, kt = newton_fAv(mv, v, xi, dt)
    et = np.linalg.norm(wt - ref) / np.linalg.norm(ref)
    if t <= 100:
        da = coef_exp_accurate(xi, t)
        wa, ka = newton_fAv(mv, v, xi, da)
        ea = np.linalg.norm(wa - ref) / np.linalg.norm(ref)
        crel = np.max(np.abs(dt[:ka + 1] - da[:ka + 1]) / np.abs(da[:ka + 1]))
        ok_acc &= ea < 1e-12
    else:
        ea, crel, ka = float("nan"), float("nan"), kt
    est = math.sqrt(2 * (2 * t) * math.log(1e12))
    LEJA.append((t, kt, est, et, ea, crel))
    ok_tab &= et < 1e-10
    print(f"    t={t:5d} matvecs={kt:4d} (estimate {est:5.0f})  error: table coef {et:.1e}, accurate coef {ea:.1e}; coef rel. diff {crel:.0e}")
report("Newton-Leja with accurate coefficients: error < 1e-12 (t <= 100)", ok_acc)
report("Newton-Leja with table coefficients: error < 1e-10 (t <= 3000)", ok_tab)
report("matvecs within 30% of sqrt(2c ln(1/eps)), c = 2t, for t >= 25",
       all(abs(k - e) <= 0.3 * e for (t, k, e, *_ ) in LEJA if t >= 25))

# ============================================================== 10.4 matrix elements
print("--- 10.4 Matrix elements by walks")
nq = 3
Xp = np.array([[0., 1.], [1., 0.]]); Zp = np.diag([1., -1.])


def site(o, i):
    m = np.array([[1.]])
    for k in range(nq):
        m = np.kron(m, o if k == i else np.eye(2))
    return m


Gam = 0.3
H = (sum(-site(Zp, i) @ site(Zp, i + 1) for i in range(nq - 1))
     - 0.2 * sum(site(Zp, i) for i in range(nq)) - Gam * sum(site(Xp, i) for i in range(nq)))
beta = 1.0
R = sl.expm(-beta * H)
Ed = np.diag(H)
LamV = nq * Gam
WALK = []
ok_b, ok_c = True, True
for (x, y) in [(0, 0), (0, 1), (0, 7)]:
    for Q in [2, 4, 6, 8, 10]:
        t0 = time.time()
        val = walk_element(H, x, y, beta, Q)
        tt = time.time() - t0
        bound = math.exp(-beta * Ed.min()) * sum((beta * LamV)**q / math.factorial(q) for q in range(Q + 1, 60))
        err = abs(val - R[y, x])
        WALK.append((x, y, Q, err, bound, tt))
        ok_b &= err <= bound
    ok_c &= WALK[-1][3] < 1e-8
    print("    <%d|e^-bH|%d>: " % (y, x) + "  ".join(f"Q={w[2]}:{w[3]:.0e}" for w in WALK[-5:]) + f"   bound(Q=10) {WALK[-1][4]:.0e}")
report("Thm 10.8 truncation bound holds for all Q", ok_b)
report("walk sums converge (error < 1e-8 at Q = 10)", ok_c)
# larger beta: positivity of every DD weight term and accuracy of dd_push states
H2 = H.copy()
val = walk_element(H2, 0, 0, 3.0, 12)
R2 = sl.expm(-3.0 * H2)[0, 0]
report("walk sum at beta = 3, Q = 12 matches expm to 1e-4", abs(val - R2) / R2 < 1e-4, f"{abs(val - R2) / R2:.1e}")

# ============================================================== 10.5 matrix nodes
print("--- 10.5 Block and matrix-node divided differences")
rng = np.random.default_rng(5)
# finite walk expansion for a layered (nilpotent) off-diagonal part: Identity 4.42
qq, r = 3, 2
Dj = [np.diag(rng.standard_normal(r)) for _ in range(qq + 1)]
Tj = [rng.standard_normal((r, r)) for _ in range(qq)]
nB = (qq + 1) * r
B = np.zeros((nB, nB))
for j in range(qq + 1):
    B[j*r:(j+1)*r, j*r:(j+1)*r] = Dj[j]
    if j:
        B[j*r:(j+1)*r, (j-1)*r:j*r] = Tj[j - 1]          # lower block bidiagonal
bt = 0.7
F = sl.expm(-bt * B)
ok = True
for a0 in range(r):
    for aq in range(r):
        s = 0.0
        for a1 in range(r):
            for a2 in range(r):
                al = [a0, a1, a2, aq]
                amp = np.prod([Tj[j][al[j + 1], al[j]] for j in range(qq)])
                nodes_ = [-bt * Dj[j][al[j], al[j]] for j in range(qq + 1)]
                s += amp * (-bt)**qq * math.exp(dd_exp_taylor(nodes_)) / (-bt)**qq * (-bt)**qq
        ok &= abs(F[qq*r + aq, a0] - s) < 1e-12 * max(1, abs(s))
report("Identity 4.42 (hidden-state block bidiagonal)", ok)

# Prop: commuting (diagonal) arguments reduce to scalar divided differences
L_, d_, beta = 4, 3, 1.3
diags = [rng.standard_normal(d_) for _ in range(L_)]
W = matrix_dd_exp([np.diag(a) for a in diags], beta)
ok = all(abs(W[k, k] - math.exp(dd_exp_taylor([beta * a[k] for a in diags])) * beta**(L_ - 1)) < 1e-13 for k in range(d_))
report("commuting arguments: e^b[A..] = diag of scalar e^{b[a..]}", ok)

# simplex integral, L = 2 and L = 3
from scipy.integrate import quad_vec, dblquad
A1, A2, A3 = (rng.standard_normal((d_, d_)) for _ in range(3))
I2, _ = quad_vec(lambda s: sl.expm((beta - s) * A2) @ sl.expm(s * A1), 0, beta, epsabs=1e-14, epsrel=1e-13)
ok2 = np.abs(matrix_dd_exp([A1, A2], beta) - I2).max() < 1e-12
I3, _ = quad_vec(lambda s1: quad_vec(lambda s2: sl.expm((beta - s1 - s2) * A3) @ sl.expm(s2 * A2) @ sl.expm(s1 * A1),
                                     0, beta - s1, epsabs=1e-13, epsrel=1e-12)[0], 0, beta, epsabs=1e-13, epsrel=1e-12)
ok3 = np.abs(matrix_dd_exp([A1, A2, A3], beta) - I3).max() < 1e-10
report("Identity 4.45: block exponential = time-ordered simplex integral (L=2,3)", ok2 and ok3)

# contour form, Identity 4.46 (trapezoid rule on a circle)
As = [A1, A2, A3]
rad = 1 + max(np.abs(np.linalg.eigvals(A)).max() for A in As)
Nt = 256
th = 2 * np.pi * np.arange(Nt) / Nt
acc = np.zeros((d_, d_), dtype=complex)
for t_ in th:
    s_ = rad * np.exp(1j * t_)
    Pm = np.eye(d_, dtype=complex)
    for A in As:                                    # (s - A_L)^-1 ... (s - A_1)^-1
        Pm = np.linalg.inv(s_ * np.eye(d_) - A) @ Pm
    acc += np.exp(beta * s_) * Pm * s_
acc /= Nt
err446 = np.abs(acc - matrix_dd_exp(As, beta)).max()
report("Identity 4.46: contour integral of e^{bs} prod (s - A_l)^-1", err446 < 1e-11, f"{err446:.1e}")

# uniformized series, Identity 4.47, vs block exponential and nonnegativity
ok = True
for t in range(5):
    As = []
    for l in range(5):
        A = np.abs(rng.standard_normal((d_, d_))); np.fill_diagonal(A, -10 * rng.random(d_)); As.append(A)
    W1 = matrix_dd_exp(As, beta); W2 = matrix_dd_uniformized(As, beta, 12.0)
    ok &= np.abs(W1 - W2).max() < 1e-13 * np.abs(W1).max() and (W2 >= 0).all()
report("Identity 4.47: uniformized series = block exponential, entries >= 0", ok)

# many blocks: the corner is small; expm loses it, the uniformized series does not
MANY = []
for L_ in [10, 20, 30]:
    rng2 = np.random.default_rng(L_)
    As = []
    for l in range(L_):
        A = 0.5 * np.abs(rng2.standard_normal((2, 2))); np.fill_diagonal(A, -2 * rng2.random(2)); As.append(A)
    Rm = mp_block_corner([As[L_ - 1 - j] for j in range(L_)], [np.eye(2)] * (L_ - 1), beta=beta, dps=80)
    rel = lambda W: max(float(abs(W[i, j] - Rm[i][j]) / abs(Rm[i][j])) for i in range(2) for j in range(2))
    e1 = rel(matrix_dd_exp(As, beta)); e2 = rel(matrix_dd_uniformized(As, beta, 4.0))
    MANY.append((L_, float(abs(Rm[0][0])), e1, e2))
    print(f"    L={L_}: corner ~{float(abs(Rm[0][0])):.0e}  expm rel {e1:.1e}  uniformized rel {e2:.1e}")
report("many blocks: expm loses the corner, uniformized accurate (< 1e-14)",
       MANY[-1][2] > 1e-6 and max(m[3] for m in MANY) < 1e-14)


# superdiagonal scaling (as for Z_theta in Chapter 8) rescues expm here too
THETA = []
for L_ in [20, 30]:
    rng2 = np.random.default_rng(L_)
    As = []
    for l in range(L_):
        A = 0.5 * np.abs(rng2.standard_normal((2, 2))); np.fill_diagonal(A, -2 * rng2.random(2)); As.append(A)
    Rm = mp_block_corner([As[L_ - 1 - j] for j in range(L_)], [np.eye(2)] * (L_ - 1), beta=beta, dps=80)
    th_ = float(L_)
    T = np.zeros((2 * L_, 2 * L_))
    for j in range(L_):
        T[2*j:2*j+2, 2*j:2*j+2] = As[L_ - 1 - j]
        if j + 1 < L_:
            T[2*j:2*j+2, 2*j+2:2*j+4] = th_ * np.eye(2)
    Wt = sl.expm(beta * T)[:2, -2:] / th_**(L_ - 1)
    e = max(float(abs(Wt[i, j] - Rm[i][j]) / abs(Rm[i][j])) for i in range(2) for j in range(2))
    THETA.append((L_, e))
    print(f"    L={L_}: expm with couplings theta = L: rel {e:.1e}")
report("theta-scaled couplings make expm accurate (< 1e-12)", max(e for _, e in THETA) < 1e-12)

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
