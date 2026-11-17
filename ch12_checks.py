"""Numerical checks for Chapter 12 (off-diagonal expansions of quantum
partition functions).

Run:  python3 code/ch12_checks.py

The functions pmr_decompose and cheb_walk_moment are the listings of the
chapter, copied verbatim.  dd_push (Proposition 9.9) and walk_element
(Listing 10.4) are reused from code/ch10_checks.py, and the reference
routine ref_exp from code/ch09_checks.py.  Exact values come from dense
diagonalization (numpy, or mpmath where stated).
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
U = 2.0**-53
RESULTS = []
HERE = Path(__file__).resolve().parent


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:64s} {info}")


_ns = {"__file__": str(HERE / "ch10_checks.py")}
exec((HERE / "ch10_checks.py").read_text().split(
    "# ===================================================================== helpers")[0], _ns)
dd_push, walk_element, ref_exp = _ns["dd_push"], _ns["walk_element"], _ns["ref_exp"]
dd_exp_squaring = _ns["dd_exp_squaring"]


# ===================================================================== listings
# ---- Listing 12.1 -----------------------------------------------------------
def pmr_decompose(H, tol=0.0):
    """PMR of a 2^n x 2^n matrix: H = sum_b D_b X^b with D_b diagonal,
    X^b |z> = |z xor b>.  Returns {b: d_b} with d_b[z] = <z|H|z^b>,
    keeping the b with some |d_b| > tol; key 0 is the diagonal."""
    N = H.shape[0]
    z = np.arange(N)
    out = {}
    for b in range(N):
        d = H[z, z ^ b]
        if b == 0 or np.abs(d).max() > tol:
            out[b] = d.copy()
    return out


# ---- Listing 12.2 -----------------------------------------------------------
def cheb_walk_moment(E, offs, n, absolute=False):
    """mu_n = Tr T_n(H), H = diag(E) + sum_b diag(d_b) X^b with
    spectrum in [-1, 1], as a sum over closed walks of length <= n.
    offs = {b: d_b}; a key b = 0 (the diagonal) is ignored.  The
    column (T_m[y_0..y_k])_{m<=n} is updated by the three-term
    recurrence when a node is appended (Chapter 5)."""
    total = 0.0

    def dfs(z0, z, k, amp, col):
        nonlocal total
        if z == z0:
            w = amp * col[n]
            total += abs(w) if absolute else w
        if k == n:
            return
        for b, d in offs.items():
            z2 = z ^ b
            if b == 0 or d[z2] == 0:
                continue
            y, new = E[z2], np.zeros(n + 1)
            if k == 0:
                new[1] = 1.0
            for m in range(1, n):        # k + 1 >= 1, new[0] = 0
                new[m + 1] = 2 * (y * new[m] + col[m]) - new[m - 1]
            dfs(z0, z2, k + 1, amp * d[z2], new)

    for z0 in range(len(E)):
        th = np.arccos(np.clip(E[z0], -1, 1))
        col = np.cos(np.arange(n + 1) * th)     # T_m(E_z0)
        dfs(z0, z0, 0, 1.0, col)
    return total


# ===================================================================== helpers
I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], float)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1.0, -1.0])


def op(n, i, P):
    """P on qubit i of n (qubit 0 is the most significant bit)."""
    out = np.array([[1.0]])
    for k in range(n):
        out = np.kron(out, P if k == i else I2)
    return out


def tfim(n, J, h, periodic=False):
    H = sum(-J * op(n, i, SZ) @ op(n, (i + 1) % n, SZ)
            for i in range(n if periodic else n - 1))
    return H - h * sum(op(n, i, SX) for i in range(n))


def heisenberg(n, J, bonds):
    H = 0
    for i, j in bonds:
        for P in (SX, SY, SZ):
            H = H + J * op(n, i, P) @ op(n, j, P)
    return np.real_if_close(H)


def logZ(H, beta):
    w = np.linalg.eigvalsh(H)
    return -beta * w.min() + math.log(np.exp(-beta * (w - w.min())).sum())


def stoquastized(H):
    """D - |V|, the Hamiltonian whose walks carry the weights |W|."""
    A = -np.abs(H)
    np.fill_diagonal(A, np.diag(H).real)
    return A


# ============================================================== 12.1 PMR
print("--- 12.1 Permutation matrix representation")
rng = np.random.default_rng(12)
ok = True
for n in [2, 3, 4]:
    N = 2**n
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = A + A.conj().T
    pm = pmr_decompose(H)
    z = np.arange(N)
    R = sum(np.diag(d)[:, z ^ b] for b, d in pm.items())   # D_b X^b
    ok &= np.abs(R - H).max() < 1e-13
    # Hermiticity: conj(d_b(z)) = d_b(z ^ b)
    ok &= all(np.abs(d.conj() - d[z ^ b]).max() < 1e-13 for b, d in pm.items())
report("Prop 12.2: H = sum_b D_b X^b reconstructs H; D_b^* = X^b D_b X^b", ok)
Hh = heisenberg(3, 1.0, [(0, 1), (1, 2), (0, 2)])
pm = pmr_decompose(Hh, 1e-14)
bs = sorted(b for b in pm if b)
z = np.arange(8)
zz = {b: np.array([1.0 if bin(zz_ & b).count("1") % 2 == 0 else -1.0 for zz_ in z])
      for b in bs}
ok = all(np.allclose(pm[b], 1.0 - zz[b]) for b in bs)
report("Example: Heisenberg bond: D_{ij} = J (1 - Z_i Z_j), 3 permutations", ok
       and len(bs) == 3, f"permutations {bs}")

# ============================================================== 12.2 Z as closed walks
print("--- 12.2 Partition function as a sum over closed walks")
H2 = tfim(2, 1.0, 0.6) + 0.3 * op(2, 0, SX) @ op(2, 1, SX)
E2 = np.diag(H2).real
Vabs = np.abs(H2 - np.diag(E2))
Lam = Vabs.sum(axis=0).max()
beta = 1.0
Zex = math.exp(logZ(H2, beta))
ok, rat = True, []
for Q in [4, 7, 10]:
    ZQ = sum(walk_element(H2, x, x, beta, Q) for x in range(4))
    tail = 4 * math.exp(-beta * E2.min()) * sum((beta * Lam)**q / math.factorial(q)
                                               for q in range(Q + 1, 80))
    ok &= abs(ZQ - Zex) <= tail
    rat.append(f"{abs(ZQ - Zex) / tail:.2f}")
report("Thm 12.4/12.6: Z_Q -> Z, |Z - Z_Q| <= N e^{-b Emin} tail(b Lam)", ok,
       "ratios " + " ".join(rat))
# root (rotation) summation: Z = sum e^{-bE} - b sum_q (1/q) sum_walks A e^{-b[cycle]}
def closed_walks(Hm, z0, q):
    """all closed walks z0 -> ... -> z0 of length q with nonzero amplitude"""
    N = Hm.shape[0]
    out = []

    def rec(path):
        if len(path) == q:
            if Hm[z0, path[-1]] != 0 and path[-1] != z0:
                out.append(path)
            return
        for z2 in range(N):
            if z2 != path[-1] and Hm[z2, path[-1]] != 0:
                rec(path + [z2])
    rec([z0])
    return out


def ddbeta(beta, E):
    """e^{-beta[E]} for real E, from the mpmath reference of Chapter 9"""
    q = len(E) - 1
    return float((-beta)**q * ref_exp([-beta * e for e in E], 40))


# absolute weights: |W| of each closed walk is the weight of the same walk
# for D - |V|; compare the sums of |W| up to length 6 with Listing 10.4
Hs = stoquastized(H2)
absum = sum(math.exp(-beta * e) for e in E2)
for q in range(2, 7):
    for x in range(4):
        for path in closed_walks(H2, x, q):
            amp = math.prod(H2[path[(k + 1) % q], path[k]] for k in range(q))
            absum += abs(amp * ddbeta(beta, [E2[p] for p in path] + [E2[x]]))
ws_ = sum(walk_element(Hs, x, x, beta, 6) for x in range(4))
report("Thm 12.4: sum of |W| = walk sum of D - |V| (orders <= 6)",
       abs(absum - ws_) / ws_ < 1e-12)


Qmax = 6
lhs = sum(walk_element(H2, x, x, beta, Qmax) for x in range(4))
rhs = sum(math.exp(-beta * e) for e in E2)
for q in range(1, Qmax + 1):
    s = 0.0
    for x in range(4):
        for path in closed_walks(H2, x, q):
            amp = math.prod(H2[path[(k + 1) % q], path[k]] for k in range(q))
            s += amp * ddbeta(beta, [E2[p] for p in path])
    rhs += -beta * s / q
report("Prop 12.5: root-summed form equals the rooted closed-walk sum (Q=6)",
       abs(lhs - rhs) / abs(lhs) < 1e-12, f"{abs(lhs - rhs) / abs(lhs):.0e}")

# ============================================================== 12.3 response
print("--- 12.3 Response functions")
beta = 0.8
Hr = tfim(2, 1.0, 0.6)
Er = np.diag(Hr).real
Aop = np.diag(op(2, 0, SZ)).copy()          # diagonal observables
Bop = np.diag(op(2, 1, SZ) + 0.5 * op(2, 0, SZ)).copy()
w, Q_ = np.linalg.eigh(Hr)
An = Q_.T @ np.diag(Aop) @ Q_
Bn = Q_.T @ np.diag(Bop) @ Q_


def chi_lehmann(om):
    tot = 0j
    for a in range(4):
        for b in range(4):
            x, y = w[a], w[b] - 1j * om        # -e^{-beta[x, y]}
            if abs(x - y) < 1e-12:
                dd = -beta * np.exp(-beta * x)
            else:
                dd = (np.exp(-beta * x) - np.exp(-beta * y)) / (x - y)
            tot += -An[a, b] * Bn[b, a] * dd
    return tot


def chi_walks(om, Q):
    """-sum over closed walks z -> w -> z with A at w, B at z, nodes
    [x, y - i om] (Prop 12.8), total number of V-steps <= Q."""
    V = Hr - np.diag(Er)
    c = -beta * (Er.max() + Er.min()) / 2
    M = math.ceil(math.e * beta * (np.ptp(Er) / 2 + om) + 40)
    total = 0j

    def dfs(z0, z, steps, phase, amp, wst, nn):
        nonlocal total
        if phase == 1:                       # may switch here: append E_z
            dfs(z0, z, steps, 2, amp * Aop[z],
                dd_push(wst, nn - 1, -beta * Er[z] - c), nn + 1)
        elif z == z0:                        # phase 2 back at the start
            total += (-amp * Bop[z0] * (-beta)**(nn - 1) * wst.sum()
                      / math.factorial(nn - 1))
        if steps == Q:
            return
        for z2 in np.nonzero(V[:, z])[0]:
            node = Er[z2] - (1j * om if phase == 1 else 0)
            dfs(z0, z2, steps + 1, phase, amp * V[z2, z],
                dd_push(wst, nn - 1, -beta * node - c), nn + 1)

    for z0 in range(4):
        w0 = np.zeros(M + 1, dtype=complex)
        dfs(z0, z0, 0, 1, 1.0, dd_push(w0, -1, -beta * (Er[z0] - 1j * om) - c), 1)
    return np.exp(c) * total


TAB_M = []
for nmat in [0, 1, 2]:
    om = 2 * math.pi * nmat / beta
    ex = chi_lehmann(om)
    got = chi_walks(om, 14)
    TAB_M.append((nmat, ex, got))
    print(f"    Matsubara n={nmat}: Lehmann {ex.real:+.12f}{ex.imag:+.12f}i "
          f"walks {abs(got - ex):.1e}")
report("Prop 12.8: Matsubara correlator as shifted-node closed walks",
       all(abs(g - e) < 1e-9 * abs(e) for (_, e, g) in TAB_M),
       f"max rel. {max(abs(g - e) / abs(e) for (_, e, g) in TAB_M):.0e}")
# omega = 0, B = 1: beta <A> Z, i.e. Prop 12.7 (d/dh of Z)
Zr = math.exp(logZ(Hr, beta))
h = 1e-5
dZ = (math.exp(logZ(Hr - h * np.diag(Aop), beta))
      - math.exp(logZ(Hr + h * np.diag(Aop), beta))) / (2 * h)
Bsave, Asave = Bop, Aop
Bop = np.ones(4)
Aop = np.diag(op(2, 0, SZ) @ op(2, 1, SZ)).copy() + 0.25
dZ = (math.exp(logZ(Hr - h * np.diag(Aop), beta))
      - math.exp(logZ(Hr + h * np.diag(Aop), beta))) / (2 * h)
got = chi_walks(0.0, 14)
Bop, Aop = Bsave, Asave
report("Prop 12.7: dZ/dh = sum A sum_j Lam_j e^{-b}[E, E_j] = beta Tr(Lam e^{-bH})",
       abs(got.real - dZ) / abs(dZ) < 1e-7, f"{abs(got.real - dZ) / abs(dZ):.0e}")

# ============================================================== 12.4 signs
print("--- 12.4 Signs")
ok = True
for nq in [2, 3]:
    Ht = tfim(nq, 1.0, 0.7)
    E = np.diag(Ht)
    for x in range(2**nq):
        for q in range(1, 5):
            for path in closed_walks(Ht, x, q):
                amp = math.prod(Ht[path[(k + 1) % q], path[k]] for k in range(q))
                ok &= amp * ddbeta(1.3, [E[p] for p in path] + [E[path[0]]]) >= 0
report("Prop 12.9: stoquastic H: every closed-walk weight >= 0", ok)
# rooted weights of a cycle share their sign, and sum to -b A e^{-b}[cycle]
ok = True
rr = random.Random(3)
for trial in range(40):
    q = rr.randint(2, 6)
    Ec = [rr.choice([0.0, 0.5, 1.2, 2.0]) for _ in range(q)]
    Acyc = rr.choice([-1, 1]) * rr.uniform(0.1, 2)
    b = rr.uniform(0.2, 3)
    ws = [Acyc * ddbeta(b, Ec + [Ec[j]]) for j in range(q)]
    ok &= len({np.sign(v) for v in ws}) == 1
    ok &= abs(sum(ws) - (-b) * Acyc * ddbeta(b, Ec)) < 1e-12 * sum(map(abs, ws))
report("Prop 12.11: rooted weights have one sign; sum = -b A e^{-b}[cycle]", ok)
# average sign = Z / Z_abs, Table 12.1
print("    average sign <sgn> = Tr e^{-bH} / Tr e^{-b(D-|V|)}:")
TAB_S = []
models = {"TFIM chain (stoq.)": lambda n: tfim(n, 1.0, 1.0),
          "Heisenberg AFM chain": lambda n: heisenberg(n, 1.0, [(i, i + 1) for i in range(n - 1)]),
          "Heisenberg AFM ring": lambda n: heisenberg(n, 1.0, [(i, (i + 1) % n) for i in range(n)])}
for name, mk in models.items():
    for n in [3, 4, 5, 6]:
        Hm = mk(n)
        row = []
        for b in [0.5, 1.0, 2.0, 4.0]:
            row.append(math.exp(logZ(Hm, b) - logZ(stoquastized(Hm), b)))
        TAB_S.append((name, n, row))
        print(f"    {name:22s} n={n}  " + "  ".join(f"{v:.3e}" for v in row))
report("average sign: 1 for stoquastic, < 1 and decaying for the AFM ring",
       all(abs(v - 1) < 1e-12 for (nm, n, r) in TAB_S if "TFIM" in nm for v in r)
       and [r for (nm, n, r) in TAB_S if nm.endswith("ring") and n == 5][0][-1] < 0.5)
# Marshall: AFM on an open chain (bipartite) is stoquastic after rotating one sublattice
Hm = heisenberg(5, 1.0, [(i, i + 1) for i in range(4)])
Rm = np.eye(1)
for i in range(5):
    Rm = np.kron(Rm, SZ if i % 2 else I2)
Hrot = Rm @ Hm @ Rm
report("bipartite AFM: rotation prod_{odd i} Z_i gives stoquastic H, sign 1",
       abs(math.exp(logZ(Hrot, 2.0) - logZ(stoquastized(Hrot), 2.0)) - 1) < 1e-12)
# gauge: a diagonal +-1 conjugation leaves Z and Z_abs unchanged; sign-free
# iff every closed walk has (-1)^q A >= 0 (balanced), e.g. even rings
Sg = np.diag(np.where(np.arange(32) % 3 == 0, -1.0, 1.0))
Hm = heisenberg(5, 1.0, [(i, (i + 1) % 5) for i in range(5)])
inv = abs(logZ(Sg @ Hm @ Sg, 1.0) - logZ(Hm, 1.0)) + abs(
    logZ(stoquastized(Sg @ Hm @ Sg), 1.0) - logZ(stoquastized(Hm), 1.0))
report("Prop 12.10: average sign invariant under diagonal sign changes", inv < 1e-12)
# Example 12.12: winding tower of a three-cycle with equal energies
with mp.workdps(30):
    Om = lambda y: mp.exp(-y) + 2 * mp.exp(y / 2) * mp.cos(mp.sqrt(3) * y / 2) - 3
    ser = lambda y: 3 * mp.nsum(lambda k: (-1)**k * y**(3 * k) / mp.factorial(3 * k), [1, mp.inf])
    ystar = mp.findroot(Om, 5.5)
    ok = (abs(Om(1.3) - ser(1.3)) < 1e-25 and Om(1.0) < 0 and Om(4 * mp.pi / mp.sqrt(3)) > 0
          and abs(ystar - mp.mpf("5.549438398")) < 1e-8)
report("Example 12.12: Omega(y) = e^-y + 2e^{y/2}cos(sqrt3 y/2) - 3 changes sign",
       ok, f"y* = {float(ystar):.9f}, x_W = {float(-ystar**3):.6f}")

# ============================================================== 12.5 Chebyshev
print("--- 12.5 Chebyshev trace moments and closed walks")


def rescaled(H):
    w = np.linalg.eigvalsh(H)
    a = (w.max() - w.min()) / 2 * 1.01
    c = (w.max() + w.min()) / 2
    return (H - c * np.eye(len(H))) / a


Hc = rescaled(tfim(3, 1.0, 0.8, periodic=True))
pmc = pmr_decompose(Hc, 1e-14)
Ec = pmc.pop(0).real
offs = {b: d.real for b, d in pmc.items()}
lam = np.linalg.eigvalsh(Hc)
ok, worst = True, 0
for nn in [1, 2, 4, 6, 8]:
    mu = np.cos(nn * np.arccos(lam)).sum()
    got = cheb_walk_moment(Ec, offs, nn)
    worst = max(worst, abs(got - mu))
report("Thm 12.13 / Listing 12.2: closed walks of length <= n give mu_n", worst < 1e-11,
       f"{worst:.0e}")
# Prop 12.14: Z_abs >= 2^{n-1} Tr B^n, |<sgn>| <= N / (2^{n-1} Tr B^n)
Bm = sum(np.diag(np.abs(d))[:, np.arange(8) ^ b] for b, d in offs.items())
ok = True
for nn in [2, 4, 6, 8]:
    mu = np.cos(nn * np.arccos(lam)).sum()
    mabs = cheb_walk_moment(Ec, offs, nn, absolute=True)
    ok &= mabs >= 2**(nn - 1) * np.trace(np.linalg.matrix_power(Bm, nn)) * (1 - 1e-12)
    ok &= abs(mu) / mabs <= 8 / (2**(nn - 1) * np.trace(np.linalg.matrix_power(Bm, nn))) + 1e-12
report("Prop 12.14: longest walks force mu_abs >= 2^{n-1} Tr B^n", ok)
# Prop 12.15: sum_q |c_q| y^q = cosh(n arsinh y) = |T_n(iy)|, n even
ok = True
for nn in [2, 4, 8, 12]:
    c = np.polynomial.chebyshev.cheb2poly([0] * nn + [1])
    for y in [0.3, 1.0, 2.5]:
        ok &= abs(sum(abs(cq) * y**q for q, cq in enumerate(c))
                  - math.cosh(nn * math.asinh(y))) < 1e-9 * math.cosh(nn * math.asinh(y))
report("Prop 12.15: sum_q |T_n[0^(q+1)]| y^q = cosh(n arsinh y)", ok)
# Prop 12.16: constant diagonal: mu_abs = sum_i cosh(n arsinh lambda_i(B))
Hx = sum(op(3, i, SX) for i in range(3)) + 0.5 * op(3, 0, SX) @ op(3, 1, SX)
Hx = Hx / 3.6
pmx = pmr_decompose(Hx, 1e-14)
Ex = pmx.pop(0).real
offx = {b: d.real for b, d in pmx.items()}
Bx = sum(np.diag(np.abs(d))[:, np.arange(8) ^ b] for b, d in offx.items())
lb = np.linalg.eigvals(Bx)
ok, FIG1 = True, []
for nn in [2, 4, 6, 8, 10]:
    mabs = cheb_walk_moment(Ex, offx, nn, absolute=True)
    pred = sum(np.cosh(nn * np.arcsinh(l.real + 0j)).real for l in lb)
    mu = np.cos(nn * np.arccos(np.linalg.eigvalsh(Hx))).sum()
    ok &= abs(mabs - pred) < 1e-9 * pred
    FIG1.append((nn, mu, mabs))
report("Prop 12.16: constant diagonal: mu_abs = sum_i cosh(n arsinh lambda_i(B))",
       ok, " ".join(f"n={a}:{c / 8:.1f}" for a, b, c in FIG1))
# Prop 12.17: conditioning of power-moment reconstruction
ok = True
for nn in [4, 8, 16]:
    c = np.polynomial.chebyshev.cheb2poly([0] * nn + [1])
    mk = [np.sum(lam**k) for k in range(nn + 1)]
    mu = sum(ck * m for ck, m in zip(c, mk))
    krel = sum(abs(ck * m) for ck, m in zip(c, mk)) / abs(mu)
    s = np.abs(lam).max()
    # the bound needs +s and -s in the spectrum; TFIM spectrum is symmetric
    ok &= krel >= 2 * math.cosh(nn * math.asinh(s)) / 8 * (1 - 1e-9)
report("Prop 12.17: kappa_rel >= 2 cosh(n arsinh s) / N", ok)

# ============================================================== 12.6 envelopes
print("--- 12.6 Envelopes and truncation")
# geometric tail: sum_{q>Q} x^q/q! <= (ex/m)^m / (1 - x/(m+1)) needs x < m+1
ok = True
for x in [0.5, 2.0, 7.0]:
    for m in range(1, 30):
        if x < m + 1:
            tail = sum(math.exp(q * math.log(x) - math.lgamma(q + 1)) for q in range(m, 200))
            ok &= tail <= (math.e * x / m)**m / (1 - x / (m + 1)) * (1 + 1e-12)
neg = (math.e * 10 / 1)**1 / (1 - 10 / 2)
report("Cor 12.19: geometric tail bound (x < m + 1); fails at x=10, m=1",
       ok and neg < 0, f"RHS at x=10, m=1: {neg:.1f}")
# analytic envelope: f = exp, nodes in [-1, 1], circle |z| = R:
# B_q <= |G| M / (2 pi rho^{q+1}) = R e^R / (R - 1)^{q+1}; true B_q = e/q!
ok, rat = True, []
for q in range(0, 12):
    Bq = math.e / math.factorial(q)
    best = min(R * math.exp(R) / (R - 1)**(q + 1) for R in np.linspace(1.05, 40, 4000))
    ok &= Bq <= best
    rat.append(best / Bq)
report("Prop 12.20: analytic envelope for exp on [-1, 1]", ok,
       f"bound/true from {min(rat):.1f} to {max(rat):.1f}")


# normalized Gibbs divided differences e_jk in (0, 1], doubling recurrence
def e_norm_ref(dE, tau):
    """e_{0q}(tau) = q! e^{[-tau dE]} (mpmath reference)"""
    q = len(dE) - 1
    return float(ref_exp([-tau * x for x in dE], 50) * mp.factorial(q))


def double(Em):
    q = Em.shape[0] - 1
    out = np.zeros_like(Em)
    for j in range(q + 1):
        for k in range(j, q + 1):
            out[j, k] = sum(math.comb(k - j, m - j) * Em[j, m] * Em[m, k]
                            for m in range(j, k + 1)) / 2**(k - j)
    return out


rr = random.Random(7)
dE = [0.0, 1.0] + [rr.random() for _ in range(10)]
q = len(dE) - 1
tau = 20.0
ref = e_norm_ref(dE, tau)
ok = 0 < ref <= 1
# doubling identity at tau
Ehalf = np.array([[e_norm_ref(dE[j:k + 1], tau / 2) if k >= j else 0 for k in range(q + 1)]
                  for j in range(q + 1)])
ok &= abs(double(Ehalf)[0, q] - ref) < 1e-14
report("Prop 12.21: 0 < e_jk <= 1 and the doubling recurrence", ok)
# small-step bound |e_jk(d) - e^{-d mean}| <= (e/2)(d omega)^2
ok = True
for trial in range(50):
    nd = rr.randint(1, 6)
    xs = [rr.uniform(0, 1) for _ in range(nd + 1)]
    d = rr.uniform(0.01, 1)
    om = max(xs) - min(xs)
    diff = e_norm_ref(xs, d) - math.exp(-d * sum(xs) / len(xs))
    ok &= -1e-15 <= diff <= math.e / 2 * (d * om)**2
report("Prop 12.21: 0 <= e_jk(d) - e^{-d mean} <= (e/2)(d omega)^2", ok)
# accuracy of doubling from small-step start vs from a Taylor start
print("    e_{0q}(tau), q = 11, tau = 20, spread 1: error after k doublings")
TAB_D = []
for k in [4, 8, 12, 16, 20, 24, 28, 32]:
    dl = tau / 2**k
    Em = np.zeros((q + 1, q + 1))
    for j in range(q + 1):
        for kk in range(j, q + 1):
            Em[j, kk] = math.exp(-dl * sum(dE[j:kk + 1]) / (kk - j + 1))
    for _ in range(k):
        Em = double(Em)
    e_small = abs(Em[0, q] - ref)
    TAB_D.append((k, e_small))
lo, th, G = dd_exp_squaring([-tau * x for x in dE])
tay = math.exp(lo) * G[0, q] / th**q * math.factorial(q)
e_tay = abs(tay - ref)
print("    " + "  ".join(f"k={k}: {e:.1e}" for k, e in TAB_D) + f"   Taylor start: {e_tay:.1e}")
best = min(e for _, e in TAB_D)
report("small-step start: best error >> Taylor start (~u)",
       best > 1e3 * e_tay and e_tay < 1e-15, f"best small-step {best:.1e}")


# ============================================================== 12.3 static susceptibility and trace convexity
_r = np.random.default_rng(12)
_H = _r.normal(size=(6, 6)); _H = _H + _H.T
_Aop = _r.normal(size=(6, 6)); _Aop = _Aop + _Aop.T
_b = 0.7
_E, _U = np.linalg.eigh(_H)
_At = _U.T @ _Aop @ _U
with mp.workdps(40):
    Hm, Amx = mp.matrix(_H.tolist()), mp.matrix(_Aop.tolist())
    Zh = lambda h: sum(mp.expm(-_b * (Hm - h * Amx))[i, i] for i in range(6))
    d2 = mp.diff(Zh, 0, 2)
    eb = lambda a, c: (-_b * mp.e**(-_b * a) if a == c else
                       (mp.e**(-_b * a) - mp.e**(-_b * c)) / (a - c))
    chi = mp.fsum(_At[n, k]**2 * (-eb(mp.mpf(_E[n]), mp.mpf(_E[k]))) for n in range(6) for k in range(6))
report("beta chi_AA(0) = d^2 Z / dh^2", abs(d2 / (_b * chi) - 1) < 1e-12, f"{float(abs(d2 / (_b * chi) - 1)):.1e}")

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
