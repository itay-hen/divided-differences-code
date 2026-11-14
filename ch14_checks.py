"""Numerical checks for Chapter 14 (quantum simulation algorithms).

Run:  python3 code/ch14_checks.py

The functions phasor_dd and walk_operator are the listings of the
chapter, copied verbatim.  References: mpmath divided differences
(Taylor series about the centre at a precision set by the spread),
scipy.linalg.expm, and closed forms.
"""
import itertools
import math
import random
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy.linalg as sl
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")
RESULTS = []
HERE = Path(__file__).resolve().parent


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:64s} {info}")


# ===================================================================== listings
# ---- Listing 14.1 -----------------------------------------------------------
def phasor_dd(x, dt, K):
    """K-fold subdivided phasor approximation of e^{-i dt}[x_0..x_q]:
    the sum over block boundaries j_0 <= ... <= j_K of the products of
    Phi[x_a..x_b] = (-i d)^n / n! e^{-i d mean(x_a..x_b)}, d = dt/K,
    n = b - a, by dynamic programming in O(K q^2) operations."""
    q, d = len(x) - 1, dt / K
    pre = np.concatenate([[0.0], np.cumsum(x)])

    def phi(a, b):
        n = b - a
        mean = (pre[b + 1] - pre[a]) / (n + 1)
        ph = np.exp(-1j * d * mean)
        return (-1j * d)**n / math.factorial(n) * ph

    F = np.array([phi(0, j) for j in range(q + 1)])  # one block
    for _ in range(K - 1):
        F = np.array([sum(F[i] * phi(i, j) for i in range(j + 1))
                      for j in range(q + 1)])
    return F[q]


# ---- Listing 14.2 -----------------------------------------------------------
def walk_operator(E, offs, f_dd, Q):
    """T_{<=Q} = sum over words of length <= Q of P_word D_word for
    H = diag(E) + sum_b diag(d_b) X^b (Theorem 12.18): <z_q|T|z_0>
    collects prod_k d(z_k) * f_dd([E_{z_0}, ..., E_{z_q}])."""
    N = len(E)
    T = np.zeros((N, N), dtype=complex)

    def rec(z0, z, amp, nodes):
        T[z, z0] += amp * f_dd(nodes)
        if len(nodes) > Q:
            return
        for b, d in offs.items():
            z2 = z ^ b
            if d[z2] != 0:
                rec(z0, z2, amp * d[z2], nodes + [E[z2]])

    for z0 in range(N):
        rec(z0, z0, 1.0, [E[z0]])
    return T


# ===================================================================== helpers
def mp_dd_c(c, x, dps=40):
    """e^{c[x_0..x_q]} for complex c, x (repeats allowed), in mpmath."""
    with mp.workdps(dps):
        c = mp.mpc(c)
        z = [c * mp.mpc(v) for v in x]
        q = len(z) - 1
        m = sum(z) / (q + 1)
        r = max(abs(v - m) for v in z)
        with mp.workdps(dps + int(float(r) / 2.3) + 10):
            y = [v - m for v in z]
            K = q + int(math.e * float(r)) + 60
            coef = [mp.mpf(1)]
            for k in range(1, K + 1):
                coef.append(coef[-1] / k)
            v = [mp.mpc(0)] * (q + 1)
            for ck in reversed(coef):
                for a in range(q):
                    v[a] = y[a] * v[a] + v[a + 1]
                v[q] = y[q] * v[q] + ck
            return +(c**q * mp.exp(m) * v[0])


def ddi(x, t):
    """e^{-it[x]} in complex double, from mpmath."""
    return complex(mp_dd_c(-1j * t, x))


I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], float)
SZ = np.diag([1.0, -1.0])


def op(n, i, P):
    out = np.array([[1.0]])
    for k in range(n):
        out = np.kron(out, P if k == i else I2)
    return out


def pmr(H):
    N = H.shape[0]
    z = np.arange(N)
    out = {}
    for b in range(1, N):
        d = H[z, z ^ b]
        if np.abs(d).max() > 1e-14:
            out[b] = d.copy()
    return np.diag(H).real.copy(), out


rng = random.Random(14)
# ============================================================== 14.1 subdivision and phasors
print("--- 14.1 Subdivision and the phasor approximation")
ok = True
for trial in range(5):
    q = rng.randint(1, 5)
    x = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(q + 1)]
    c = complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
    K = rng.randint(2, 4)
    taus = [rng.uniform(0.1, 1) for _ in range(K)]
    lhs = mp_dd_c(c * sum(taus), x)
    rhs = 0
    for js in itertools.combinations_with_replacement(range(q + 1), K - 1):
        jj = (0,) + js + (q,)
        term = 1
        for m in range(K):
            term *= mp_dd_c(c * taus[m], x[jj[m]:jj[m + 1] + 1])
        rhs += term
    ok &= abs(lhs - rhs) < 1e-13 * max(1, abs(lhs))   # c, tau are doubles
report("Prop 14.1: subdivision identity (compendium 7.4)", ok)

ok = True
for q in [1, 4, 9]:
    for K in [1, 3, 8]:
        dt, dE = 1.3, 0.7
        x = [j * dE for j in range(q + 1)]
        ex = ddi(x, dt)
        cf = (-2j * np.exp(-1j * dt * dE / 2) * math.sin(dt * dE / 2) / dE)**q / math.factorial(q)
        u = dt * dE / (2 * K)
        ap = phasor_dd(x, dt, K)
        cf2 = (-1j * dt * np.exp(-1j * dt * dE / 2) * math.sin(dt * dE / 2)
               / (K * math.sin(u)))**q / math.factorial(q)
        ok &= abs(ex - cf) < 1e-13 and abs(ap - cf2) < 1e-13
        ok &= abs(ex / ap - (math.sin(u) / u)**q) < 1e-12
report("Prop 14.2: closed forms for equally spaced nodes, ratio (sin u/u)^q", ok)

# |phasor approx| <= dt^q/q!  and normalization of the phasor LCU
ok = True
for trial in range(50):
    q = rng.randint(0, 8)
    K = rng.randint(1, 6)
    x = [rng.uniform(-3, 3) for _ in range(q + 1)]
    dt = rng.uniform(0.1, 3)
    ok &= abs(phasor_dd(x, dt, K)) <= dt**q / math.factorial(q) * (1 + 1e-12)
    s = sum(math.prod((dt / K)**n / math.factorial(n) for n in np.diff((0,) + js + (q,)))
            for js in itertools.combinations_with_replacement(range(q + 1), K - 1))
    ok &= abs(s - dt**q / math.factorial(q)) < 1e-12 * dt**q / math.factorial(q) + 1e-300
report("Prop 14.2: |phasor sum| <= dt^q/q!; coefficients sum to dt^q/q!", ok)

# Theorem 14.3: error bounds; the bound of [KH25] fails for q >= 7
print("    equally spaced nodes: error / (dt^q/q!), new bound q u^2/6, [KH25] bound u^2")
TAB = []
grid_fail = {"KH": 0, "new": 0, "n7": 0}
for q in [3, 5, 7, 10, 20]:
    for K in [2, 8, 64]:
        for dtdE in [0.3, 1.0, 3.0]:
            dE, dt = 1.0, dtdE
            x = [j * dE for j in range(q + 1)]
            u = dt * dE / (2 * K)
            err = abs(ddi(x, dt) - phasor_dd(x, dt, K)) / (dt**q / math.factorial(q))
            new = min(1, q * u * u / 6)
            kh = u * u
            grid_fail["new"] += err > new * (1 + 1e-9)
            if q >= 7:
                grid_fail["n7"] += 1
                grid_fail["KH"] += err > kh * (1 + 1e-9)
            if dtdE == 0.3 and K in (8, 64):
                TAB.append((q, K, err, kh, new))
for r in TAB:
    print(f"    q={r[0]:2d} K={r[1]:2d}: err {r[2]:.3e}  KH25 {r[3]:.3e}  new {r[4]:.3e}"
          f"  err/KH {r[2] / r[3]:.2f}  err/new {r[2] / r[4]:.3f}")
report("Thm 14.3(a): new bound holds on the grid; [KH25] bound fails for q >= 7",
       grid_fail["new"] == 0 and grid_fail["KH"] > 0,
       f"KH25 fails in {grid_fail['KH']} of {grid_fail['n7']} cases with q >= 7")
worst = 0
for trial in range(300):
    q = rng.randint(1, 10)
    K = rng.randint(1, 8)
    dE = rng.uniform(0.1, 2)
    steps = [rng.uniform(-dE, dE) for _ in range(q)]
    x = list(np.cumsum([0.0] + steps))
    rng.shuffle(steps)
    dt = rng.uniform(0.05, 4)
    u = dt * dE / (2 * K)
    err = abs(ddi(x, dt) - phasor_dd(x, dt, K)) / (dt**q / math.factorial(q))
    worst = max(worst, err / min(2, q * u * u / 6))
report("Thm 14.3(b): arbitrary nodes with gaps <= dE: err <= min(2, q u^2/6)",
       worst <= 1 + 1e-9, f"worst ratio {worst:.3f}")
# sharpness
q, K, dt, dE = 7, 3, 1e-3, 1.0
x = [j * dE for j in range(q + 1)]
u = dt * dE / (2 * K)
with mp.workdps(60):
    exm = mp_dd_c(-1j * dt, x, dps=60)
    apm = 0
    d = mp.mpf(dt) / K
    for js in itertools.combinations_with_replacement(range(q + 1), K - 1):
        jj = (0,) + js + (q,)
        term = 1
        for m in range(K):
            n = jj[m + 1] - jj[m]
            mean = mp.fsum(x[jj[m]:jj[m + 1] + 1]) / (n + 1)
            term *= (-1j * d)**n / mp.factorial(n) * mp.exp(-1j * d * mean)
        apm += term
    ratio = abs(exm - apm) / (mp.mpf(dt)**q / mp.factorial(q)) / (q * mp.mpf(u)**2 / 6)
report("Thm 14.3: constant q/6 asymptotically sharp", abs(ratio - 1) < 1e-5,
       f"ratio {float(ratio):.7f}")

# Theorem 14.3(c): operator form on a two-qubit model
H = (-op(2, 0, SZ) @ op(2, 1, SZ) - 0.6 * (op(2, 0, SX) + op(2, 1, SX))
     + 0.25 * op(2, 0, SZ))
E, offs = pmr(H)
Gam = sum(np.abs(d).max() for d in offs.values())
gapE = max(abs(E[z ^ b] - E[z]) for b in offs for z in range(4))
ok, rat = True, []
for t, K in [(0.5, 2), (1.0, 4), (1.0, 16)]:
    Q = 10
    Tex = walk_operator(E, offs, lambda nodes: ddi(nodes, t), Q)
    Tph = walk_operator(E, offs, lambda nodes: phasor_dd(nodes, t, K), Q)
    z = Gam * t
    bound = t**2 * gapE**2 / (24 * K**2) * z * math.exp(z)
    diff = np.linalg.norm(Tex - Tph, 1)          # max column sum
    ok &= diff <= bound
    rat.append(diff / bound)
    ok &= np.abs(Tex - sl.expm(-1j * t * H)).max() < sum(z**q / math.factorial(q) for q in range(Q + 1, 60))
report("Thm 14.3(c): operator error <= t^2 dE^2/(24K^2) z e^z", ok,
       "ratios " + " ".join(f"{r:.2f}" for r in rat))

# ============================================================== 14.2 LCU normalization
print("--- 14.2 Linear combinations of unitaries")
ok = True
for dtG in [0.25, math.log(2), 1.0]:
    norm = sum(dtG**q / math.factorial(q) for q in range(60))
    ok &= abs(norm - math.exp(dtG)) < 1e-14
report("Prop 14.4: LCU normalization of one time step = e^{dt Gamma}", ok,
       f"dt Gamma = ln 2 gives {math.exp(math.log(2)):.1f}")

# ============================================================== 14.4 block encodings
print("--- 14.4 Divided-difference coefficients in block encodings")
H3 = (-sum(op(3, i, SZ) @ op(3, i + 1, SZ) for i in range(2))
      - 0.5 * sum(op(3, i, SX) for i in range(3)))
E3, offs3 = pmr(H3)
lamV = sum(np.abs(d).max() for d in offs3.values())
LamV = np.abs(H3 - np.diag(E3)).sum(axis=0).max()
Emin = E3.min()
TABB = []
for beta in [0.5, 1.0, 2.0]:
    f = lambda nodes: complex(mp_dd_c(-beta, [v - Emin for v in nodes]))
    G = sl.expm(-beta * (H3 - Emin * np.eye(8)))
    for Q in [2, 4, 6]:
        T = walk_operator(E3, offs3, f, Q).real
        err = np.linalg.norm(G - T, 2)
        tail = sum((beta * LamV)**q / math.factorial(q) for q in range(Q + 1, 80))
        A_Q = sum((beta * lamV)**q / math.factorial(q) for q in range(Q + 1))
        TABB.append((beta, Q, err, tail, A_Q, np.linalg.norm(G, 2)))
ok = all(r[2] <= r[3] * (1 + 1e-9) for r in TABB)
for r in TABB:
    print(f"    beta={r[0]} Q={r[1]}: err {r[2]:.2e} bound {r[3]:.2e}  A_Q {r[4]:.3f}"
          f"  ||G|| {r[5]:.3f}")
report("Prop 14.7: Gibbs block encoding: error <= tail, A_Q <= e^{beta lambda_V}", ok and
       all(r[4] <= math.exp(r[0] * lamV) for r in TABB))

# ============================================================== 14.5 differential equations
print("--- 14.5 Linear and nonlinear differential equations")
# driven linear ODE x' = A x + e^{gamma s} b, A = D + V (non-Hermitian, dissipative)
nA = 4
rs = np.random.default_rng(5)
A = 0.1 * rs.standard_normal((nA, nA))
np.fill_diagonal(A, -np.array([0.5, 1.0, 1.7, 2.2]))
Ed = np.diag(A).copy()
gam = -0.2 + 0.9j
bvec = rs.standard_normal(nA)
x0 = np.zeros(nA)
t = 2.0


def walk_vec(src, nodes_extra, Q, t):
    """sum over walks u -> v of prod A_{..} e^{t[E_path, extra]} times src_u"""
    out = np.zeros(nA, dtype=complex)

    def rec(u0, v, amp, nodes):
        out[v] += amp * src[u0] * complex(mp_dd_c(t, nodes + nodes_extra, 30))
        if len(nodes) > Q:
            return
        for w in range(nA):
            if w != v:
                rec(u0, w, amp * A[w, v], nodes + [Ed[w]])

    for u0 in range(nA):
        rec(u0, u0, 1.0, [Ed[u0]])
    return out


sol = solve_ivp(lambda s, y: A @ y + np.exp(gam * s) * bvec, (0, t), x0.astype(complex),
                method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
wv = walk_vec(bvec, [gam], 8, t)
report("Prop 14.8: driven ODE: walks with the appended input node gamma",
       np.abs(wv - sol).max() < 1e-6, f"Q = 8: {np.abs(wv - sol).max():.1e}")
# Carleman: x' = -x + eps x^2, exact series
eps, xi = 0.4, 0.9
tt = 1.7
series = sum(math.factorial(q) * eps**q * xi**(q + 1)
             * float(mp_dd_c(tt, [-(k + 1) for k in range(q + 1)]).real) for q in range(60))
exact = xi * math.exp(-tt) / (1 - eps * xi * (1 - math.exp(-tt)))
cf = all(abs(float(mp_dd_c(tt, [-(k + 1) for k in range(q + 1)]).real)
             - math.exp(-tt) * (1 - math.exp(-tt))**q / math.factorial(q)) < 1e-15
         for q in range(10))
report("Example 14.9: Carleman walk series = exact logistic solution", abs(series - exact) < 1e-13
       and cf, f"{series:.15f} vs {exact:.15f}")

# ============================================================== 14.6 spectral kernels
print("--- 14.6 Finite-time spectral kernels")
ok = True
for trial in range(10):
    q = rng.randint(0, 4)
    x = [rng.uniform(-1, 1) for _ in range(q + 1)]
    beta = rng.uniform(0.3, 3)
    zc = complex(rng.uniform(-1, 1), rng.uniform(0.05, 1))
    with mp.workdps(30):
        g = lambda v: (1 - mp.exp(-beta * (v - zc))) / (v - zc)
        # divided difference of g by the contour-free recursion (distinct nodes)
        def ddg(xs):
            if len(xs) == 1:
                return g(xs[0])
            return (ddg(xs[1:]) - ddg(xs[:-1])) / (xs[-1] - xs[0])
        lhs = ddg([mp.mpf(v) for v in x])
        rhs = -mp.exp(beta * zc) * mp_dd_c(-beta, x + [zc], 30)
    ok &= abs(lhs - rhs) < 1e-15 * max(1, abs(lhs))
report("Prop 14.10: augmented-node identity g[x] = -e^{bz} e^{-b}[x, z]", ok)
ok = True
for beta, eta in [(5.0, 0.2), (20.0, 0.05)]:
    for Dl in [-0.5, -0.05, 0.0, 0.1, 1.0]:
        zc = 1j * eta
        g = (1 - np.exp(-beta * (Dl - zc))) / (Dl - zc)
        K1 = g.imag / math.pi
        K2 = (eta / (Dl**2 + eta**2) - math.exp(-beta * Dl) / (Dl**2 + eta**2)
              * (eta * math.cos(beta * eta) + Dl * math.sin(beta * eta))) / math.pi
        ok &= abs(K1 - K2) < 1e-10 * max(1, abs(K1))
K0 = (1 - math.cos(20 * 0.05)) / (math.pi * 0.05)
negs = [(Dl, ((0.05 / (Dl**2 + 0.0025)) - math.exp(-20 * Dl) / (Dl**2 + 0.0025)
              * (0.05 * math.cos(1.0) + Dl * math.sin(1.0))) / math.pi)
        for Dl in [-0.5, -0.3, -0.1]]
lor = 0.05 / (0.25 + 0.0025) / math.pi
ok &= abs(K0 - (1 - math.cos(1.0)) / (math.pi * 0.05)) < 1e-15
report("Prop 14.11: finite-time kernel formula; huge for Delta < 0",
       ok and abs(negs[0][1]) > 1e3 * lor, "K(-0.5) = %.2e vs Lorentzian %.2e" % (negs[0][1], lor))
# resolvent as half-Fourier transform (compendium 4.51)
ok = True
for trial in range(4):
    q = rng.randint(0, 3)
    x = [rng.uniform(-1, 1) for _ in range(q + 1)]
    zc = complex(rng.uniform(-1, 1), rng.uniform(0.3, 1))
    with mp.workdps(20):
        f = lambda tt: mp.exp(1j * zc * tt) * mp_dd_c(-1j * tt, x, 25)
        lhs = mp.quad(f, [0, 5, 10, 20, 40, 80])
    rhs = 1j / mp.fprod(zc - v for v in x)
    ok &= abs(lhs - rhs) < 1e-8
report("Prop 14.12: int_0^inf e^{izt} e^{-it}[E] dt = i / prod (z - E_r)", ok)

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
