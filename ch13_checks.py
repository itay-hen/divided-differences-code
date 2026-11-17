"""Numerical checks for Chapter 13 (quantum Monte Carlo with
divided-difference weights).

Run:  python3 code/ch13_checks.py

The class PMRQMC and the functions estimators and fidsus_terms are the
listings of the chapter, copied verbatim.  ExpDD (Listing 9.4) and
dd_exp_taylor (Listing 9.1) are reused from code/ch09_checks.py.
References: exact diagonalization (numpy), exact enumeration of all
configurations up to a maximal length, and mpmath quadrature.
"""
import itertools
import math
import random
import time
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad

warnings.filterwarnings("ignore")
U = 2.0**-53
RESULTS = []
HERE = Path(__file__).resolve().parent


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:64s} {info}")


_ns = {"__file__": str(HERE / "ch09_checks.py")}
exec((HERE / "ch09_checks.py").read_text().split(
    "# ============================================================== 9.1 Taylor")[0], _ns)
ExpDD, dd_exp_taylor, ref_exp = _ns["ExpDD"], _ns["dd_exp_taylor"], _ns["ref_exp"]


# ===================================================================== listings
# ---- Listing 13.1 -----------------------------------------------------------
class PMRQMC:
    """PMR-QMC for H = diag(E) + sum_j diag(d_j) X^(b_j), d_j <= 0.
    A configuration is (z0, S), with z_k = z_{k-1} ^ b_{S_k}, and
    W = prod_k |d_{S_k}(z_k)| beta^q exp[-beta E_{z_0..z_q}]."""

    def __init__(self, E, ops, beta, seed=0):
        self.E, self.ops, self.beta = np.asarray(E, float), ops, beta
        self.rng = random.Random(seed)
        self.sig = -beta * self.E.max()            # nodes >= sig
        self.Mt = math.ceil(math.e * beta * np.ptp(self.E) + 37)
        self.nbits = len(self.E).bit_length() - 1
        self.z0, self.S = 0, []
        self.reset()

    def node(self, z):
        return -self.beta * self.E[z]

    def path(self, z0, S):
        zs = [z0]
        for j in S:
            zs.append(zs[-1] ^ self.ops[j][0])
        return zs

    def logamp(self, zs, S):            # -inf if an amplitude is zero
        a = [-self.ops[j][1][z] for j, z in zip(S, zs[1:])]
        return sum(map(math.log, a)) if all(a) else -math.inf

    def reset(self):                    # rebuild the dd-state
        self.dd = ExpDD(self.sig, self.Mt)
        for z in self.path(self.z0, self.S)[:len(self.S) + 1]:
            self.dd.add(self.node(z))

    def logw(self, dd, zs, S):
        return (self.logamp(zs, S) + len(S) * math.log(self.beta)
                + dd.log_value())

    @staticmethod
    def clone(dd):
        c = ExpDD.__new__(ExpDD)
        c.sigma, c.M, c.tol = dd.sigma, dd.M, dd.tol
        c.nodes, c.w, c.e = dd.nodes[:], dd.w[:], dd.e[:]
        return c


# ---- Listing 13.2 -----------------------------------------------------------
def pmr_move(s):
    """One Metropolis move: classical flip, pair insertion or
    deletion, or swap of neighbours (Prop. 13.3)."""
    r, q, M = s.rng, len(s.S), len(s.ops)
    zs = s.path(s.z0, s.S)
    old = s.logw(s.dd, zs, s.S)
    kind = r.randrange(3)
    if kind == 0:                                  # classical move
        z0 = s.z0 ^ (1 << r.randrange(s.nbits))
        S, dd = s.S, ExpDD(s.sig, s.Mt)
        for z in s.path(z0, S)[:q + 1]:
            dd.add(s.node(z))
        bias = 0.0
    elif kind == 1 and r.random() < 0.5:           # insert P_j P_j
        k, j = r.randrange(q + 1), r.randrange(M)
        z0, S = s.z0, s.S[:k] + [j, j] + s.S[k:]
        dd = s.clone(s.dd)
        dd.add(s.node(zs[k] ^ s.ops[j][0]))
        dd.add(s.node(zs[k]))
        bias = math.log(M)
    elif kind == 1:                                # delete a pair
        if q < 2:
            return
        k = r.randrange(q - 1)
        if s.S[k] != s.S[k + 1]:
            return
        z0, S = s.z0, s.S[:k] + s.S[k + 2:]
        dd = s.clone(s.dd)
        dd.remove(s.node(zs[k + 1]))
        dd.remove(s.node(zs[k]))
        bias = -math.log(M)
    else:                                          # swap neighbours
        if q < 2:
            return
        k = r.randrange(q - 1)
        if s.S[k] == s.S[k + 1]:
            return
        z0, S = s.z0, s.S[:]
        S[k], S[k + 1] = S[k + 1], S[k]
        dd = s.clone(s.dd)
        dd.remove(s.node(zs[k + 1]))
        dd.add(s.node(zs[k] ^ s.ops[S[k]][0]))
        bias = 0.0
    new = s.logw(dd, s.path(z0, S), S)
    if math.log(r.random() + 1e-300) < new - old + bias:
        s.z0, s.S, s.dd = z0, S, dd


# ---- Listing 13.3 -----------------------------------------------------------
def estimators(s, diag):
    """Estimators for one configuration: two energies, q, and the
    diagonal observable at z0 and averaged over the gaps (13.3)."""
    q, b = len(s.S), s.beta
    zs = s.path(s.z0, s.S)
    E = s.E
    lv = s.dd.log_value()
    eA = E[s.z0] - q / b                          # Prop. 13.5
    eB = E[s.z0]
    if q:                                         # Prop. 3.16 form
        dd = s.clone(s.dd)
        dd.remove(s.node(s.z0))
        eB -= math.exp(dd.log_value() - lv) / b
    gaps = []
    for z in zs[:q + 1]:                          # E_mu[L_k] / beta
        dd = s.clone(s.dd)
        dd.add(s.node(z))
        gaps.append(math.exp(dd.log_value() - lv))
    lam_t = sum(g * diag[z] for g, z in zip(gaps, zs))
    if q == 0:
        lam_t = diag[s.z0]
    return eA, eB, q, diag[s.z0], lam_t


# ---- Listing 13.4 -----------------------------------------------------------
def fidsus_terms(x, beta):
    """Fidelity-susceptibility estimator f(C) of Prop. 13.12 for a
    walk with energies x_0..x_q (x_q = x_0); J(k) from Prop. 13.9."""
    q = len(x) - 1
    if q < 2:
        return 0.0
    h = beta / 2

    def dd(nodes, c):          # e^{-c}[nodes] as (sign, log|.|)
        n = len(nodes) - 1
        lg = dd_exp_taylor([-c * v for v in nodes])
        return (-1)**n, n * math.log(c) + lg

    sw, lw = dd(list(x), beta)
    tot = 0.0
    for k in range(1, q):
        a = list(x[k + 1:q + 1])              # segment at beta - tau
        bb = list(x[1:k + 1])                 # segment at tau
        for r in range(len(a)):
            s1, l1 = dd(a[:r + 1], h)
            for m in bb:
                s2, l2 = dd(a[r:] + bb + [m], h)
                tot += s1 * s2 * sw * math.exp(l1 + l2 - lw)
    return tot


# ===================================================================== helpers
I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], float)
SZ = np.diag([1.0, -1.0])


def op(n, i, P):
    out = np.array([[1.0]])
    for k in range(n):
        out = np.kron(out, P if k == i else I2)
    return out


def model(n, J, h, a=0.0):
    """H = -J sum Z_i Z_{i+1} - sum_i h (1 + a Z_{i+1}) X_i, open bonds,
    Z_{n+1} = Z_1 in the field term:
    the transverse-field Ising chain for a = 0, a stoquastic model with
    nonconstant D_j for 0 < a < 1.  Returns (H, E, ops)."""
    N = 2**n
    H = sum(-J * op(n, i, SZ) @ op(n, i + 1, SZ) for i in range(n - 1))
    ops = []
    for i in range(n):
        Dz = h * (np.eye(N) + (a * op(n, (i + 1) % n, SZ) if a else 0))
        H = H - Dz @ op(n, i, SX)
        ops.append((1 << (n - 1 - i), -np.diag(Dz).copy()))
    return H, np.diag(H).copy(), ops


def ed(H, beta):
    w, V = np.linalg.eigh(H)
    p = np.exp(-beta * (w - w.min()))
    return w, V, p / p.sum()


def binned(xs, nb=20):
    xs = np.asarray(xs, float)
    m = len(xs) // nb
    b = xs[:m * nb].reshape(nb, m).mean(axis=1)
    return b.mean(), b.std(ddof=1) / math.sqrt(nb)


# ============================================================== 13.1 configurations
print("--- 13.1-13.2 Configurations, moves and incremental weights")
H, E, ops = model(3, 1.0, 0.8, a=0.4)
beta = 1.5
s = PMRQMC(E, ops, beta, seed=1)
worst = 0.0
for it in range(4000):
    pmr_move(s)
    if it % 50 == 0:
        ref = PMRQMC(E, ops, beta)
        ref.z0, ref.S = s.z0, s.S[:]
        ref.reset()
        worst = max(worst, abs(s.dd.log_value() - ref.dd.log_value()))
report("Listing 13.1-13.2: incremental node updates = recomputation",
       worst < 1e-12, f"max |log diff| {worst:.0e}")

# exact enumeration: sum over all configurations (z0, S), |S| <= Q
H2, E2, ops2 = model(2, 1.0, 0.25, a=0.3)
beta2 = 1.0


def configs(nb, M, Q):
    for q in range(0, Q + 1):
        for S in itertools.product(range(M), repeat=q):
            for z0 in range(2**nb):
                yield z0, list(S)


def closed(s, z0, S):
    return s.path(z0, S)[-1] == z0


s2 = PMRQMC(E2, ops2, beta2)
Zsum, acc = 0.0, np.zeros(5)
for z0, S in configs(2, 2, 10):
    if not closed(s2, z0, S):
        continue
    s2.z0, s2.S = z0, S
    s2.reset()
    w = math.exp(s2.logw(s2.dd, s2.path(z0, S), S))
    est = estimators(s2, np.diag(op(2, 0, SZ) @ op(2, 1, SZ)))
    Zsum += w
    acc += w * np.array(est)
acc /= Zsum
w_, V_, p_ = ed(H2, beta2)
Zex = np.exp(-beta2 * w_).sum()
Hd = np.diag(E2)
Vd = H2 - Hd
Lam = np.diag(op(2, 0, SZ) @ op(2, 1, SZ))
e_ex = (p_ * w_).sum()
lam_ex = sum(p_[k] * V_[:, k] @ (Lam * V_[:, k]) for k in range(4))
v_ex = sum(p_[k] * V_[:, k] @ Vd @ V_[:, k] for k in range(4))
report("Thm 13.2: enumerated weights sum to Z (|S| <= 10)", abs(Zsum - Zex) / Zex < 1e-9,
       f"{abs(Zsum - Zex) / Zex:.0e}")
ok = (abs(acc[0] - e_ex) < 1e-8 and abs(acc[1] - e_ex) < 1e-8
      and abs(-acc[2] / beta2 - v_ex) < 1e-8 and abs(acc[3] - lam_ex) < 1e-8
      and abs(acc[4] - lam_ex) < 1e-8)
report("Props 13.5-13.6: estimators unbiased (exact enumeration)", ok,
       f"E {acc[0]:.10f}/{acc[1]:.10f} vs {e_ex:.10f}")

# ============================================================== 13.3 identities
print("--- 13.3 Estimator identities")
rr = random.Random(5)
ok = True
for trial in range(30):
    q = rr.randint(1, 7)
    x = [rr.choice([0.0, 0.5, 1.0, 1.75]) for _ in range(q)]
    x = x + [x[0]]                                     # closed walk
    b = rr.uniform(0.3, 3)
    W = (-b)**q * ref_exp([-b * v for v in x], 40)
    lhs = -x[0] / W * mp.fsum(x[j] * (-b)**(q + 1) * ref_exp([-b * v for v in x + [x[j]]], 40)
                              for j in range(q + 1))
    Wm = (-b)**(q - 1) * ref_exp([-b * v for v in x[:-1]], 40)
    rhs = b * x[0]**2 + q * x[0] + b * x[0] * Wm / W
    ok &= abs(lhs - rhs) < 1e-12 * max(1, abs(rhs))   # b is a float
report("Prop 13.7: closed-walk ES identity (compendium 2.24)", ok)
# suffix identities 2.25-2.26
ok = True
for trial in range(30):
    q = rr.randint(1, 5)
    x = [rr.uniform(-1, 1) for _ in range(q + 1)]
    t = rr.uniform(0.2, 2)
    r = rr.randint(0, q)

    def D(nodes, c=t):
        n = len(nodes) - 1
        return (-c)**n * ref_exp([-c * v for v in nodes], 40)

    Fr = D(x[r:])
    Sr = mp.fsum((l - r) * D(x[r:] + [x[l]]) for l in range(r, q + 1))
    Gr = mp.fsum(l * D(x[r:] + [x[l]]) for l in range(r, q + 1))
    lhs = mp.fsum(x[i] * D(x[r:] + [x[i], x[j]]) for i in range(r, q + 1) for j in range(i, q + 1))
    mid = D(x[r + 1:]) if r < q else 0
    rhs = t**2 / 2 * x[r] * Fr + t**2 / 2 * mid - Sr
    ok &= abs(lhs - rhs) < 1e-12 and abs(Sr - (Gr + t * r * Fr)) < 1e-12
report("Prop 13.8: suffix identities (compendium 2.25, 2.26)", ok)

# ============================================================== 13.5 imaginary time
print("--- 13.5 Imaginary-time integrals and fidelity susceptibility")


def Dmp(nodes, c):
    n = len(nodes) - 1
    return (-mp.mpf(c))**n * ref_exp([-c * v for v in nodes], 30)


ok, worst = True, 0
for trial in range(6):
    q = rr.randint(1, 5)
    x = [rr.choice([0.0, 0.3, 1.1]) + rr.uniform(-0.2, 0.2) * (trial % 2) for _ in range(q + 1)]
    b = rr.uniform(0.5, 3)
    j = rr.randint(0, q - 1)
    with mp.workdps(20):
        f = lambda tau: tau * Dmp(x[j + 1:], tau) * Dmp(x[:j + 1], b - tau)
        lhs = mp.quad(f, [0, b / 2])
        g = lambda tau: tau * Dmp(x[j:], tau) * Dmp(x[:j + 1], b - tau)
        lhs2 = mp.quad(g, [0, b / 2])
    rhs = mp.fsum(Dmp(x[:r + 1], b / 2) * mp.fsum(Dmp(x[r:] + [x[m]], b / 2)
                                                for m in range(j + 1, q + 1)) for r in range(j + 1))
    rhs2 = mp.fsum(Dmp(x[:r + 1], b / 2) * mp.fsum(Dmp(x[r:] + [x[j], x[m]], b / 2)
                                                 for m in range(j, q + 1)) for r in range(j + 1))
    sym = mp.fsum(mp.quad(lambda tt: tt * Dmp(x[:jj + 1], tt) * Dmp(x[jj:], b - tt), [0, b / 2])
                  + mp.quad(lambda tt: (b - tt) * Dmp(x[jj:], tt) * Dmp(x[:jj + 1], b - tt), [0, b / 2])
                  for jj in range(q + 1))
    worst = max(worst, float(abs(lhs - rhs)), float(abs(lhs2 - rhs2)),
                float(abs(sym - b**2 / 2 * Dmp(x, b))))
report("Props 13.9, 13.11: half-interval tau-weighted integrals (2.32, 2.34, 2.35)",
       worst < 1e-15, f"{worst:.0e}")
# general tau^k (2.37) for k = 0, 1, 2
worst = 0
for trial in range(4):
    q = rr.randint(2, 4)
    x = [rr.uniform(-1, 1) for _ in range(q + 1)]
    b = rr.uniform(0.5, 2.5)
    j = rr.randint(0, q - 1)
    for k in range(3):
        with mp.workdps(20):
            lhs = mp.quad(lambda tau: tau**k * Dmp(x[j + 1:], tau) * Dmp(x[:j + 1], b - tau),
                          [0, b / 2])
        tail = x[j + 1:]
        rhs = 0
        for r in range(j + 1):
            # g_r[x_{j+1}..x_q] = sum over multisets of size k of tail labels
            inner = mp.fsum(Dmp(x[r:j + 1] + tail + [tail[i] for i in mset], b / 2)
                            for mset in itertools.combinations_with_replacement(range(len(tail)), k))
            rhs += Dmp(x[:r + 1], b / 2) * inner
        rhs *= (-1)**(k + 1) * math.factorial(k)
        worst = max(worst, float(abs(lhs - rhs)))
report("Prop 13.10: tau^k half-interval integral (compendium 2.37)", worst < 1e-15,
       f"{worst:.0e}")

# fidelity susceptibility estimator: exact enumeration vs ED
Vd2 = H2 - np.diag(E2)
Vn = V_.T @ Vd2 @ V_


def chiF_ed(beta, w, Vn):
    p = np.exp(-beta * (w - w.min()))
    Z = p.sum()
    tot = 0.0
    for m in range(len(w)):
        for n_ in range(len(w)):
            om = w[m] - w[n_]
            f = quad(lambda t: t * math.exp(-beta * (w[m] - w.min()) + t * om), 0, beta / 2,
                     epsabs=1e-14, epsrel=1e-13)[0]
            tot += f * Vn[m, n_]**2
    vexp = sum(p[m] * Vn[m, m] for m in range(len(w))) / Z
    return tot / Z - beta**2 / 8 * vexp**2


num = 0.0
qs = 0.0
for z0, S in configs(2, 2, 10):
    if not closed(s2, z0, S):
        continue
    s2.z0, s2.S = z0, S
    s2.reset()
    w = math.exp(s2.logw(s2.dd, s2.path(z0, S), S))
    xs = [E2[z] for z in s2.path(z0, S)]
    num += w * fidsus_terms(xs, beta2)
    qs += w * len(S)
chi_enum = num / Zsum - (qs / Zsum)**2 / 8
chi_ex = chiF_ed(beta2, w_, Vn)
report("Prop 13.12 / Listing 13.4: fidelity-susceptibility estimator (enumeration)",
       abs(chi_enum - chi_ex) < 1e-8, f"{chi_enum:.10f} vs {chi_ex:.10f}")

# ============================================================== 13.4 validation by simulation
print("--- 13.4 Simulation against exact diagonalization")
TAB = []
for (n, J, h, a, beta) in [(4, 1.0, 1.0, 0.0, 1.0), (4, 1.0, 1.0, 0.0, 2.0),
                           (5, 1.0, 0.7, 0.4, 2.0)]:
    H, E, ops = model(n, J, h, a)
    w, V, p = ed(H, beta)
    Vdn = V.T @ (H - np.diag(E)) @ V
    Mz = np.diag(sum(op(n, i, SZ) for i in range(n)))**2 / n**2
    e_ex = (p * w).sum()
    m_ex = sum(p[k] * V[:, k] @ (Mz * V[:, k]) for k in range(len(w)))
    chi_ex = chiF_ed(beta, w, Vdn)
    s = PMRQMC(E, ops, beta, seed=7)
    t0 = time.time()
    for _ in range(5000):
        pmr_move(s)
    data = []
    nsweep = 10000
    for sw in range(nsweep):
        for _ in range(20):
            pmr_move(s)
        est = estimators(s, Mz)
        xs = [E[z] for z in s.path(s.z0, s.S)]
        data.append(list(est) + [fidsus_terms(xs, beta) if sw % 4 == 0 else np.nan])
    el = time.time() - t0
    data = np.array(data)
    eA, dA = binned(data[:, 0])
    eB, dB = binned(data[:, 1])
    mq, dq = binned(data[:, 2])
    m0, d0 = binned(data[:, 3])
    mt, dt = binned(data[:, 4])
    fm, dfm = binned(data[::4, 5])
    chi_mc = fm - mq**2 / 8
    TAB.append(dict(n=n, h=h, a=a, beta=beta, e_ex=e_ex, eA=eA, dA=dA, eB=eB, dB=dB,
                    m_ex=m_ex, m0=m0, d0=d0, mt=mt, dt=dt, chi_ex=chi_ex, chi=chi_mc, dchi=dfm,
                    q=mq, sdA=data[:, 0].std(), sdB=data[:, 1].std(),
                    sd0=data[:, 3].std(), sdt=data[:, 4].std(), sec=el))
    print(f"    n={n} h={h} a={a} beta={beta}: <q>={mq:.2f}  E {eA:.4f}({dA:.4f}) "
          f"{eB:.4f}({dB:.4f}) ex {e_ex:.4f} | m2 {m0:.4f}({d0:.4f}) {mt:.4f}({dt:.4f}) "
          f"ex {m_ex:.4f} | chiF {chi_mc:.4f}({dfm:.4f}) ex {chi_ex:.4f}  [{el:.0f}s]")
ok = all(abs(r["eA"] - r["e_ex"]) < 4 * r["dA"] and abs(r["eB"] - r["e_ex"]) < 4 * r["dB"]
         and abs(r["m0"] - r["m_ex"]) < 4 * r["d0"] and abs(r["mt"] - r["m_ex"]) < 4 * r["dt"]
         and abs(r["chi"] - r["chi_ex"]) < 4 * r["dchi"] + 0.02 * abs(r["chi_ex"])
         for r in TAB)
report("Simulation: E, m^2, chi_F within 4 error bars of ED (3 models)", ok)
report("time-averaged diagonal estimator has smaller variance than z0 alone",
       all(r["sdt"] < r["sd0"] for r in TAB),
       " ".join(f"{r['sdt'] / r['sd0']:.2f}" for r in TAB))

# ============================================================== 13.6 gap moments
print("--- 13.6 Gap estimation from half-interval moments")
H, E, ops = model(6, 1.0, 1.5)
Vd = H - np.diag(E)
w, V = np.linalg.eigh(H)
Vn = V.T @ Vd @ V
gap_true = None
An = Vn[0, 1:]**2
dn = w[1:] - w[0]
gap_true = dn[np.argmax(An > 1e-12)]
ROWS = []
for beta in [5.0, 10.0, 20.0, 40.0]:
    p = np.exp(-beta * (w - w.min()))
    p /= p.sum()
    vexp = (p * np.diag(Vn)).sum()

    def C(t):
        return sum(p[m] * Vn[m, k]**2 * math.exp(-t * (w[k] - w[m]))
                   for m in range(len(w)) for k in range(len(w)) if p[m] > 1e-18) - vexp**2

    Mk = [quad(lambda t: t**k * C(t), 0, beta / 2, limit=200, epsabs=1e-14)[0] for k in range(6)]
    R = [(k + 1) * Mk[k] / Mk[k + 1] for k in range(5)]
    ROWS.append((beta, R))
    print(f"    beta={beta:4.0f}: R_k = " + " ".join(f"{v:.5f}" for v in R) + f"   gap {gap_true:.5f}")
ok = all(all(R[k] >= R[k + 1] - 1e-9 for k in range(4)) and R[-1] >= gap_true - 1e-6
         for _, R in ROWS[2:])
report("Prop 13.14: R_0 >= R_1 >= ... >= gap at low temperature (beta >= 20)", ok)

# ============================================================== 13.7 insertion and shifts
print("--- 13.7 Insertion ratios and shift bounds")


def phi(E, beta):
    q = len(E) - 1
    return beta**q * ref_exp([-beta * v for v in E], 30)


ok = True
for trial in range(5):
    E3 = [rr.uniform(0, 1.5) for _ in range(3)]          # q = 2
    beta = rr.uniform(0.5, 3)
    r = rr.randint(0, 2)
    Dl = rr.uniform(-1, 2)
    with mp.workdps(20):
        def dens(L0, L1):
            L2 = beta - L0 - L1
            return mp.exp(-(E3[0] * L0 + E3[1] * L1 + E3[2] * L2))

        def Ldens(L0, L1, fn):
            L = [L0, L1, beta - L0 - L1][r]
            return fn(L) * dens(L0, L1)

        Z0 = mp.quad(lambda a: mp.quad(lambda b_: dens(a, b_), [0, beta - a]), [0, beta])
        EL = mp.quad(lambda a: mp.quad(lambda b_: Ldens(a, b_, lambda L: L), [0, beta - a]),
                     [0, beta]) / Z0
        F = lambda L: (Dl * L - 1 + mp.exp(-Dl * L)) / Dl**2
        EF = mp.quad(lambda a: mp.quad(lambda b_: Ldens(a, b_, F), [0, beta - a]),
                     [0, beta]) / Z0
    lst = E3[:r + 1] + [E3[r]] + E3[r + 1:]
    ok &= abs(EL - phi(lst, beta) / phi(E3, beta)) < 1e-15
    lst2 = E3[:r + 1] + [E3[r] + Dl, E3[r]] + E3[r + 1:]
    ok &= abs(EF - phi(lst2, beta) / phi(E3, beta)) < 1e-15
report("Props 13.15-13.16: E[L_r] and pair-insertion ratio E[F_Delta(L_r)]", ok)
ok = True
for trial in range(200):
    q = rr.randint(0, 8)
    E0 = [rr.uniform(-1, 1) for _ in range(q + 1)]
    Vx = [rr.uniform(-1, 1) for _ in range(q + 1)]
    J = max(map(abs, Vx))
    beta, th, th2 = rr.uniform(0.2, 4), rr.random(), rr.random()
    th, th2 = min(th, th2), max(th, th2)
    ratio = (phi([e + th2 * v for e, v in zip(E0, Vx)], beta)
             / phi([e + th * v for e, v in zip(E0, Vx)], beta))
    ok &= math.exp(-(th2 - th) * beta * J) * (1 - 1e-12) <= ratio <= math.exp((th2 - th) * beta * J) * (1 + 1e-12)
report("Prop 13.17: shift bound e^{-d beta J} <= ratio <= e^{d beta J}", ok)

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
