"""Regression cases from the external review of November 2026.

Each case was a counterexample to a listing or a statement of the first
complete draft. The listings have been repaired and the statements
corrected; the checks below test the corrected behaviour, or, where the
text now documents a limitation (the contour stopping rule), the
documented behaviour.

Run:  python3 code/regressions.py
"""
import math
import random
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RESULTS = []


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:64s} {info}")


def listings(script):
    """Namespace with the listings of a check script (the part before
    its helpers), without running its checks."""
    src = (HERE / script).read_text()
    cut = src.index("\n# ===================================================================== helpers")
    ns = {"__file__": str(HERE / script), "__name__": "listings"}
    exec(src[:cut], ns)
    return ns


L08, L09, L10 = listings("ch08_checks.py"), listings("ch09_checks.py"), listings("ch10_checks.py")
L11, L13 = listings("ch11_checks.py"), listings("ch13_checks.py")
mp.mp.dps = 40

# ---- 1. scaling and squaring at high order and small spread (Listings 9.2, 11.1)
for name, xs in [("spread 0", [0.0] * 201),
                 ("spread 1e-3", list(np.linspace(0, 1e-3, 201))),
                 ("spread 0.7", list(np.linspace(0, 0.7, 201)))]:
    mid, th, G = L09["dd_exp_squaring"](xs)
    q = len(xs) - 1
    scaled = mp.e ** mid * G[0, -1]            # = th^q exp[x]
    # reference: exp[x] = sum_k h_k(x)/(k+q)!, nodes small and nonnegative
    hs = [mp.mpf(1)] + [mp.mpf(0)] * 40
    for x in xs:                               # h_k of all nodes, k <= 40
        for k in range(1, 41):
            hs[k] = hs[k] + mp.mpf(x) * hs[k - 1]
    ref = sum(hs[k] / mp.factorial(k + q) for k in range(41)) * mp.mpf(th) ** q
    err = abs(scaled / ref - 1)
    report(f"9.2 squaring q=200, {name}: scaled corner accurate", err < 1e-12, f"{float(err):.1e}")
mid, th, G = L11["dd_expi_squaring"]([0.0] * 201, 1.0)
ref = mp.mpf(th) ** 200 / mp.factorial(200)
report("11.1 real-time squaring q=200, spread 0", abs(G[0, -1] / ref - 1) < 1e-12,
       f"{complex(G[0, -1]):.4e}")

# ---- 2. contour stopping rule (Listing 8.2): documented aliasing
f = lambda z: 1 + z ** 32
T = L08["dd_contour"](f, [0.0], R=1.0)
report("8.2 contour: f=1+z^32 aliases at N=16,32 (documented)", abs(T - 2) < 1e-12, f"{T:.3g}")
T = L08["dd_contour"](f, [0.0], R=1.0, N=64)
report("8.2 contour: with N=64 > 32 the value is exact", abs(T - 1) < 1e-14, f"{T:.3g}")

# ---- 3. Loewner matrix with a wide spectrum (Listing 10.1)
Lw = L10["loewner_exp"](np.array([-1000.0, 0.0]))
ref = (1 - math.exp(-1000)) / 1000
report("10.1 Loewner exp[-1000, 0] finite and accurate",
       np.isfinite(Lw).all() and abs(Lw[0, 1] / ref - 1) < 1e-14, f"{Lw[0, 1]:.6g}")
lam = np.array([-3.0, -3.0 + 1e-9, 0.5, 700.0])
Lw = L10["loewner_exp"](lam)
ok = True
for i in range(4):
    for j in range(4):
        a, b = mp.mpf(lam[i]), mp.mpf(lam[j])
        r = mp.e ** a if a == b else (mp.e ** a - mp.e ** b) / (a - b)
        ok &= abs(Lw[i, j] / r - 1) < 1e-13
report("10.1 Loewner: confluent, clustered and large eigenvalues", ok)

# ---- 4. walk sums: grouped convergence versus absolute convergence (Thm 10.8)
V = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, 1, 0, 1], [1, -1, 1, 0]], float)
ok = abs(np.linalg.norm(V, 2) - math.sqrt(5)) < 1e-12
ok &= abs(max(np.linalg.eigvalsh(np.abs(V))) - 3) < 1e-12
c = 2.8                                       # f(z) = 1/(c - z), dd = c^-(q+1)
grouped = [abs(np.linalg.matrix_power(V, q)[0, 0]) / c ** (q + 1) for q in (40, 80, 120)]
absolute = [np.linalg.matrix_power(np.abs(V), q)[0, 0] / c ** (q + 1) for q in (40, 80, 120)]
ok &= grouped[-1] < 1e-5 and absolute[2] > absolute[1] > absolute[0] > 1
report("10.8 Neumann series converges, absolute walk sum diverges", ok,
       f"{grouped[-1]:.1e} vs {absolute[-1]:.1e}")

# ---- 5. Sambe cutoff needs K >= q max|m_r| (Cor. 11.17)
Om, t = 1.0, 0.3
X = np.array([[0, 1], [1, 0]], complex)


def sambe_first_order(K):
    """First-order term of <1|U(t)|0> for H(t) = 2 cos(2 Om t) X, from
    the Sambe space truncated to |k| <= K (harmonics m = +-2)."""
    ks = list(range(-K, K + 1))
    tot = 0
    for k in ks:                               # one event 0 -> 1, k: 0 -> +-2
        if k in (2, -2):
            # walk weight -i t * <1|X|0> * dd_{-it}[E_0 - 0, E_1 - k Om] with D = 0
            a, b = 0.0, -k * Om
            w = (np.exp(-1j * t * a) - np.exp(-1j * t * b)) / (a - b)
            tot += np.exp(-1j * k * t * Om) * w
    return tot


exact = -1j * 2 * math.sin(2 * Om * t) / (2 * Om)   # -i int_0^t 2cos(2 Om s) ds
report("11.17 Sambe: K=1 loses first order, K=2 reproduces it",
       sambe_first_order(1) == 0 and abs(sambe_first_order(2) - exact) < 1e-14,
       f"{exact:.6g}")

# ---- 6. PMR Hermiticity for a three-cycle, and the Pauli phase (Prop. 12.2, Ex. 12.3)
rng = np.random.default_rng(3)
A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
H = A + A.conj().T
P = np.roll(np.eye(3), 1, axis=0)             # cyclic shift
Pi = P.T


def d_of(Pm):                                  # d_P(z') = <z'|H|P^{-1} z'>
    Pinv = Pm.T
    return np.array([H[z, int(np.argmax(Pinv @ np.eye(3)[z]))] for z in range(3)])


DP, DPi = np.diag(d_of(P)), np.diag(d_of(Pi))
ok = np.allclose(DP.conj(), P @ DPi @ Pi) and not np.allclose(DP.conj(), Pi @ DPi @ P)
Hrec = sum(np.diag(d_of(M)) @ M for M in (np.eye(3), P, Pi))
ok &= np.allclose(Hrec, H)
report("12.2 PMR: D_P* = P D_{P^-1} P^-1 for a three-cycle", ok)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0])
report("12.3 Pauli phase: Y = (-i) Z X", np.allclose(Y, -1j * Z @ X))

# ---- 7. QMC with zero amplitudes (Listing 13.1)
E = np.zeros(4)
ops = [(3, np.array([-1.0, 0.0, 0.0, -1.0]))]    # flips both qubits; zero on {1, 2}
s = L13["PMRQMC"](E, ops, beta=1.0, seed=5)
zs = s.path(1, [0, 0])
ok = s.logamp(zs, [0, 0]) == -math.inf
try:
    for _ in range(3000):
        L13["pmr_move"](s)
        assert math.isfinite(s.logw(s.dd, s.path(s.z0, s.S), s.S))
except Exception as e:                         # pragma: no cover
    ok = False
    print("   ", repr(e))
report("13.1 QMC: zero-amplitude proposals rejected, chain stays in support", ok)

# ---- 8. thermal domination needs an absolute value (Section 4.5)
tau, nodes = 1.0, [0.0, 0.0]                   # q = 1: dd_{-tau}[0,0] = -tau
w = -tau * math.exp(0)
report("4.5 signed thermal weight is negative for odd q", w < 0 and abs(w) == 1)

# ---- 9. no exact complex mean value theorem (Theorem 3.2)
d = (mp.e ** (1j * 2 * mp.pi) - 1) / (2 * mp.pi)
report("3.2 exp(iz)[0, 2pi] = 0 while |f'| = 1 on the hull", abs(d) < 1e-30)

# ---- 10. gap ratios at finite temperature (Section 13.6)
mp.mp.dps = 60
En = [0, 10, 11, 12]
Oel = {(0, 1): mp.e ** -260, (1, 2): 1, (1, 3): 1}
Oel.update({(j, i): v for (i, j), v in list(Oel.items())})


def C(tau_, beta):
    Zp = sum(mp.e ** (-beta * e) for e in En)
    return sum(mp.e ** (-beta * En[m]) * mp.e ** (-tau_ * (En[n] - En[m])) * v ** 2
               for (m, n), v in Oel.items()) / Zp


ok = True
for beta in (40, 50):
    M = [mp.quad(lambda x: x ** k * C(x, beta), [0, 1, 5, beta / 2]) for k in range(5)]
    R = [(k + 1) * M[k] / M[k + 1] for k in range(4)]
    ok &= all(R[k] > R[k + 1] for k in range(3)) and R[0] < 1.21 and R[-1] > 1
report("13.6 monotone, stable gap ratios far below the gap 10", ok)

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
