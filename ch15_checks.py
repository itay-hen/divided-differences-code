"""Numerical checks for Chapter 15 (open systems, path sums, and open
questions).

Run:  python3 code/ch15_checks.py

The functions occupation_expansion and winding_tower are the listings of
the chapter, copied verbatim.  dd_push (Proposition 9.9) is reused from
code/ch10_checks.py and dd_exp_taylor (Listing 9.1) from
code/ch09_checks.py.  References: scipy.linalg.expm, mpmath.
"""
import itertools
import math
import random
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy.linalg as sl

warnings.filterwarnings("ignore")
RESULTS = []
HERE = Path(__file__).resolve().parent


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:64s} {info}")


_ns = {"__file__": str(HERE / "ch10_checks.py")}
exec((HERE / "ch10_checks.py").read_text().split(
    "# ===================================================================== helpers")[0], _ns)
dd_push, dd_exp_taylor = _ns["dd_push"], _ns["dd_exp_taylor"]


# ===================================================================== listings
# ---- Listing 15.1 -----------------------------------------------------------
def occupation_expansion(lev, eps, V, u, T, Q):
    """Column u of e^{-T(A+V)}, A = diag(eps[lev[i]]), summed over
    occupation vectors nu (Thm 15.6): walks are grouped by how often
    they visit each level, which fixes the divided difference."""
    N, r = len(lev), len(eps)
    start = tuple(int(a == lev[u]) for a in range(r))
    cur = {start: np.eye(N)[u]}              # W^{(0)}_{u->.}(nu)
    out = np.zeros(N)
    for q in range(Q + 1):
        for nu, vec in cur.items():          # (-1)^q W e^{T[-eps^nu]}
            nodes = [-T * eps[a] for a in range(r)
                     for _ in range(nu[a])]
            lg = dd_exp_taylor(nodes) + q * math.log(T)
            out += vec * math.exp(lg)        # (-1)^q cancels the sign
        new = {}                             # recursion (15.x)
        for nu, vec in cur.items():
            w = -V @ vec                     # one step of -TV
            for a in range(r):
                part = np.where(lev == a, w, 0.0)
                if part.any():
                    key = nu[:a] + (nu[a] + 1,) + nu[a + 1:]
                    new[key] = new.get(key, 0.0) + part
        cur = new
    return out


# ---- Listing 15.2 -----------------------------------------------------------
def winding_tower(E, A, beta):
    """Omega(W) = -beta sum_n A^n/n e^{-beta}[E^n] for a primitive
    cycle with energies E and amplitude A, from the roots of
    P(z) - A, P(z) = prod (z - E_j) (Thm 15.11)."""
    P = np.poly(E).astype(complex)           # coefficients of P(z)
    P[-1] -= A
    lam = np.roots(P)
    E = np.asarray(E, dtype=float)
    val = np.exp(-beta * lam).sum() - np.exp(-beta * E).sum()
    return val.real


# ===================================================================== helpers
def mp_ddc(c, x, dps=40):
    """e^{c[x_0..x_q]} for complex c, x (repeats allowed), in mpmath."""
    with mp.workdps(dps):
        c = mp.mpc(c)
        z = [c * mp.mpc(v) for v in x]
        q = len(z) - 1
        m = sum(z) / (q + 1)
        rr = max(abs(v - m) for v in z)
        with mp.workdps(dps + int(float(rr) / 2.3) + 10):
            y = [v - m for v in z]
            K = q + int(math.e * float(rr)) + 60
            coef = [mp.mpf(1)]
            for k in range(1, K + 1):
                coef.append(coef[-1] / k)
            v = [mp.mpc(0)] * (q + 1)
            for ck in reversed(coef):
                for a in range(q):
                    v[a] = y[a] * v[a] + v[a + 1]
                v[q] = y[q] * v[q] + ck
            return +(c**q * mp.exp(m) * v[0])


def liouvillian(H, Ls):
    """Superoperator of drho/dt = -i[H, rho] + sum L rho L^+ - {L^+L, rho}/2
    on row-major vec(rho), vec(A rho B) = (A kron B^T) vec(rho)."""
    n = H.shape[0]
    I = np.eye(n)
    S = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for L in Ls:
        LdL = L.conj().T @ L
        S += np.kron(L, L.conj()) - 0.5 * np.kron(LdL, I) - 0.5 * np.kron(I, LdL.T)
    return S


rng = random.Random(15)
nrng = np.random.default_rng(15)
# ============================================================== 15.1 Lindblad
print("--- 15.1 Lindblad dynamics as walks in Liouville space")
ok = True
for trial in range(200):
    n = nrng.integers(2, 5)
    A = nrng.standard_normal((n, n)) + 1j * nrng.standard_normal((n, n))
    H = A + A.conj().T
    Ls = [nrng.standard_normal((n, n)) + 1j * nrng.standard_normal((n, n))
          for _ in range(nrng.integers(1, 4))]
    S = liouvillian(H, Ls)
    ok &= np.diag(S).real.max() <= 1e-12
report("Prop 15.1: diagonal of a Lindbladian has Re <= 0 (any basis)", ok, "200 random")

# walk expansion of e^{tS} for a driven, decaying, dephasing qubit
w0, g, gam, kap = 1.0, 0.35, 0.3, 0.2
Hq = 0.5 * w0 * np.diag([1.0, -1.0]) + g * np.array([[0, 1], [1, 0]])
S = liouvillian(Hq, [math.sqrt(gam) * np.array([[0, 0], [1, 0]], complex),
                     math.sqrt(kap) * np.diag([1.0, -1.0]).astype(complex)])
Ed = np.diag(S).copy()
Vs = S - np.diag(Ed)
Gam = np.abs(Vs).sum(axis=0).max()
t = 1.5


def liouville_walks(src, Q):
    """sum over walks of prod V e^{t[nodes]} applied to src (Thm 11.9 with
    complex nodes), depth-first with the Taylor dd-state"""
    c = Ed.mean()
    rad = t * max(abs(e - c) for e in Ed)
    M = math.ceil(math.e * rad + 40)
    out = np.zeros(4, complex)

    def dfs(z, q, amp, w):
        out[z] += amp * t**q * w.sum() / math.factorial(q)
        if q == Q:
            return
        for z2 in np.nonzero(Vs[:, z])[0]:
            dfs(z2, q + 1, amp * Vs[z2, z], dd_push(w, q, t * (Ed[z2] - c)))

    for z0 in range(4):
        if src[z0] != 0:
            w0_ = np.zeros(M + 1, complex)
            dfs(z0, 0, src[z0], dd_push(w0_, -1, t * (Ed[z0] - c)))
    return np.exp(t * c) * out


rho0 = np.array([1, 0, 0, 0], complex)                 # |0><0|
ref = sl.expm(t * S) @ rho0
ok, rat = True, []
for Q in [4, 8, 12]:
    got = liouville_walks(rho0, Q)
    err = np.abs(got - ref).sum()
    tail = sum((t * Gam)**q / math.factorial(q) for q in range(Q + 1, 80))
    ok &= err <= tail
    rat.append(err / tail)
report("Thm 15.2: Liouville-space walk expansion, error <= tail(t Gamma)", ok,
       "ratios " + " ".join(f"{x:.2f}" for x in rat))

# ============================================================== 15.2 path sums
print("--- 15.2 Path sums on Hamiltonian graphs")
# Riesz-projector expansion (3.29): Jordan example
nu = 0.3 + 0.2j
J = np.array([[nu, 1], [0, nu]])
N_ = np.array([[0, 1], [0, 0]], complex)
V2 = np.array([[0.4, -0.7j], [1.1, 0.25]])
T = 1.3
from scipy.integrate import quad_vec
lhs = quad_vec(lambda s: sl.expm((T - s) * J) @ V2 @ sl.expm(s * J), 0, T, epsabs=1e-14)[0]
rhs = np.exp(nu * T) * (T * V2 + T**2 / 2 * (N_ @ V2 + V2 @ N_) + T**3 / 6 * N_ @ V2 @ N_)
report("Example 15.4: Jordan block first-order term (compendium 3.29)",
       np.abs(lhs - rhs).max() < 1e-12, f"{np.abs(lhs - rhs).max():.0e}")
# full Riesz expansion for a 3x3 non-diagonalizable L0, f = exp
L0 = np.array([[0.2, 1.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, -0.5]], complex)
P1 = np.diag([1, 1, 0]).astype(complex)
P2 = np.diag([0, 0, 1]).astype(complex)
N1 = L0 @ P1 - 0.2 * P1
labels = [(0.2, P1, N1, 2), (-0.5, P2, np.zeros((3, 3)), 1)]
Vr = 0.15 * (nrng.standard_normal((3, 3)) + 1j * nrng.standard_normal((3, 3)))
Tt = 1.0
tot = np.zeros((3, 3), complex)
for q in range(0, 9):
    for als in itertools.product(range(2), repeat=q + 1):
        for ms in itertools.product(*[range(labels[a][3]) for a in als]):
            nodes = [labels[a][0] for a, m in zip(als, ms) for _ in range(m + 1)]
            ddv = complex(mp_ddc(Tt, nodes, 30))
            Mx = np.linalg.matrix_power(labels[als[0]][2], ms[0]) @ labels[als[0]][1]
            for k in range(1, q + 1):
                a = als[k]
                Mx = np.linalg.matrix_power(labels[a][2], ms[k]) @ labels[a][1] @ Vr @ Mx
            tot += ddv * Mx
err = np.abs(tot - sl.expm(Tt * (L0 + Vr))).max()
report("Thm 15.3: Riesz-projector path expansion (non-normal L0)", err < 1e-9, f"{err:.0e}")
# occupation grouping (3.31, 3.32), Listing 15.1
eps = np.array([0.0, 0.8, 1.9])
lev = np.array([0, 1, 1, 2, 2, 2])
Vg = nrng.standard_normal((6, 6))
Vg = 0.25 * (Vg + Vg.T)
Ag = np.diag(eps[lev])
Tg = 1.2
ref = sl.expm(-Tg * (Ag + Vg))
ok, counts = True, []
for u in range(6):
    got = occupation_expansion(lev, eps, Vg, u, Tg, 30)
    ok &= np.abs(got - ref[:, u]).max() < 1e-12
report("Thm 15.6 / Listing 15.1: occupation-grouped series = e^{-T(A+V)}", ok)
nwalk, ncls = [], []
for q in range(1, 13):
    nwalk.append(6 * 6**q)
    ncls.append(math.comb(q + 1 + 2, 2))
print("    q, walks from one state (6^q), occupation classes C(q+3, 2):",
      [(q, 6**q, math.comb(q + 3, 2)) for q in (4, 8, 12)])
# birth-death chain (3.33)
Ew = [0.0, 0.7, 1.5, 2.6]
bw = [0.5, 0.4, 0.3]
cw = [0.6, 0.3, 0.45]
Hb = np.diag(Ew).astype(float)
for w in range(3):
    Hb[w + 1, w] = bw[w]
    Hb[w, w + 1] = cw[w]
Tb = 1.1
ref = sl.expm(-Tb * Hb)[0, 0]
tot = 0.0
for ks in itertools.product(range(7), repeat=3):          # k_0, k_1, k_2
    k = list(ks) + [0]
    coef = 1.0
    for w in range(1, 4):
        if k[w - 1] == 0 and k[w] > 0:
            coef = 0.0
            break
        if k[w - 1] + k[w] > 0:
            coef *= math.comb(k[w - 1] + k[w] - 1, k[w])
    if coef == 0:
        continue
    for w in range(3):
        coef *= (bw[w] * cw[w])**k[w]
    mult = [k[0] + 1] + [k[w - 1] + k[w] for w in range(1, 4)]
    nodes = [Ew[w] for w in range(4) for _ in range(mult[w])]
    tot += coef * float(mp_ddc(-Tb, nodes, 30).real)
report("Prop 15.7: birth-death occupation series (compendium 3.33)", abs(tot - ref) < 1e-10,
       f"{abs(tot - ref):.0e}")
# self-loop resummation (3.34)
x = [0.2, -0.4, 0.9]
a, tt, j = 0.15, 1.3, 1
lhs = mp.fsum(a**k * mp_ddc(tt, x[:j] + [x[j]] * (k + 1) + x[j + 1:]) for k in range(60))
rhs = mp_ddc(tt, x[:j] + [x[j] + a] + x[j + 1:])
report("Prop 15.8: self-loop resummation shifts the node", abs(lhs - rhs) < 1e-14,
       f"{float(abs(lhs - rhs)):.0e} (x_j + a rounded in double)")

# ============================================================== 15.3 cycle resummation
print("--- 15.3 Cycle resummation")
ok, worst = True, 0
for trial in range(20):
    q = rng.randint(1, 4)
    E = [rng.uniform(-1, 1) for _ in range(q)]
    A = rng.uniform(-1.5, 1.5)
    beta = rng.uniform(0.3, 2.5)
    ser = mp.fsum(-beta / n * mp.mpf(A)**n * mp_ddc(-beta, E * n, 30) for n in range(1, 40))
    got = winding_tower(E, A, beta)
    worst = max(worst, abs(float(ser.real) - got))
report("Thm 15.11 / Listing 15.2: winding tower = sum of root exponentials", worst < 1e-10,
       f"{worst:.0e}")
# ring matrix
ok = True
for trial in range(10):
    q = rng.randint(2, 5)
    E = [rng.uniform(-1, 1) for _ in range(q)]
    a = [rng.uniform(0.2, 1.2) * rng.choice([-1, 1]) for _ in range(q)]
    HW = np.diag(E).astype(float)
    for jj in range(q - 1):
        HW[jj + 1, jj] = a[jj]
    HW[0, q - 1] = a[q - 1]
    beta = rng.uniform(0.3, 2)
    lhs = np.trace(sl.expm(-beta * HW)) - np.exp(-beta * np.array(E)).sum()
    ok &= abs(lhs - winding_tower(E, math.prod(a), beta)) < 1e-10
    ok &= abs(np.linalg.det(np.eye(q) * 0.37 - HW)
              - (np.prod([0.37 - e for e in E]) - math.prod(a))) < 1e-12
report("Prop 15.10: ring matrix det(z - H_W) = P_W(z) - A_W, Omega = Tr difference", ok)
# loop (Tr log) expansion
Hl = np.diag([0.0, 1.0, 2.2]).astype(complex)
Vl = 0.2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], complex)
zc = 4.0 + 1.0j
lhs = np.log(np.linalg.det(zc * np.eye(3) - Hl - Vl) / np.linalg.det(zc * np.eye(3) - Hl))
G0 = np.diag(1 / (zc - np.diag(Hl)))
tr = -sum(np.trace(np.linalg.matrix_power(G0 @ Vl, L)) / L for L in range(1, 80))
report("Thm 15.9: log det(z-H)/det(z-D) = -sum_L Tr(G0 V)^L / L", abs(lhs - tr) < 1e-12)
# sign: positive case and activity bound, Dirichlet contraction
ok_pos, ok_act, ok_dir = True, True, True
for trial in range(200):
    q = rng.randint(1, 5)
    E = [rng.uniform(-1, 1) for _ in range(q)]
    beta = rng.uniform(0.2, 3)
    A = rng.uniform(0.05, 3)
    if q % 2:
        A = -A                                       # x_W = (-beta)^q A > 0
    xW = (-beta)**q * A
    Om = winding_tower(E, A, beta)
    ok_pos &= Om > 0
    w1 = -beta * A * float(mp_ddc(-beta, E, 30).real)
    dW = Om / w1 - 1
    Dq = sum(math.exp((n - 1) * math.log(abs(xW)) + math.lgamma(q + 1)
                      - math.lgamma(n * q + 1)) for n in range(2, 40))
    ok_act &= abs(dW) <= Dq * (1 + 1e-9)
    for n in range(1, 5):
        lhsn = math.factorial(n * q - 1) / beta**(n * q - 1) * abs(float(mp_ddc(-beta, E * n, 30).real))
        rhs1 = math.factorial(q - 1) / beta**(q - 1) * abs(float(mp_ddc(-beta, E, 30).real))
        ok_dir &= lhsn <= rhs1 * (1 + 1e-12)
report("Prop 15.12: x_W > 0 implies Omega(W) > 0", ok_pos)
report("Prop 15.13: Dirichlet contraction (nq-1)!/b^(nq-1)|e[E^n]| <= ...", ok_dir)
report("Cor 15.14: activity bound |delta_W| <= Delta_q(|x_W|)", ok_act)
# cyclic block resummation (4.43) with 1x1 blocks reduces to root summation
ok = True
for trial in range(10):
    q = rng.randint(2, 5)
    E = [rng.uniform(-1, 1) for _ in range(q)]
    b = rng.uniform(0.3, 2)
    rooted = sum(float(mp_ddc(-b, E[jj:] + E[:jj] + [E[jj]], 30).real) for jj in range(q))
    ok &= abs(rooted - (-b) * float(mp_ddc(-b, E, 30).real)) < 1e-12
# block form: rooted weights with hidden labels
d = 2
Dbl = [np.diag(nrng.uniform(-1, 1, d)) for _ in range(3)]
Tbl = [0.5 * nrng.standard_normal((d, d)) for _ in range(3)]
b = 1.1
# C: block lower bidiagonal with D_0, D_1, D_2 and subdiagonal T_1, T_2; T_3 closes 2 -> 0
C = np.zeros((3 * d, 3 * d))
for k in range(3):
    C[k*d:(k+1)*d, k*d:(k+1)*d] = Dbl[k]
for k in range(1, 3):
    C[k*d:(k+1)*d, (k-1)*d:k*d] = Tbl[k - 1]
Wcyc = -b * np.trace(Tbl[2] @ sl.expm(-b * C)[2*d:3*d, 0:d])
# brute force: hidden labels h_0, h_1, h_2 (one per block) and three rotations
brute = 0.0
for hs in itertools.product(range(d), repeat=3):
    Es = [Dbl[k][hs[k], hs[k]] for k in range(3)]
    amp = Tbl[0][hs[1], hs[0]] * Tbl[1][hs[2], hs[1]] * Tbl[2][hs[0], hs[2]]
    for jj in range(3):
        brute += amp * float(mp_ddc(-b, Es[jj:] + Es[:jj] + [Es[jj]], 30).real)
report("Prop 15.15: cyclic block resummation W_cyc = -b Tr(T_q [e^{-bC}]_{q-1,0})",
       ok and abs(Wcyc - brute) < 1e-12, f"{abs(Wcyc - brute):.0e}")

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
