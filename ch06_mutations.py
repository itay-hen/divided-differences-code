"""Mutation tests for the Chapter 6 checks.

Run:  python3 code/ch06_mutations.py

Each test evaluates an identity of Chapter 6 twice: once as stated, and
once with a deliberate error of the kind that is easy to make when
transcribing it. The stated version must pass and the mutated version
must fail, which shows that the checks in ch06_checks.py can detect
such errors. The divided differences are computed as in ch06_checks.py,
from the Cauchy integral by the trapezoid rule.
"""
import itertools
import random

import mpmath as mp

mp.mp.dps = 40
random.seed(66)
NQ = 240
R = mp.mpf(2)
_W = [mp.expjpi(2 * mp.mpf(j) / NQ) for j in range(NQ)]
TOL = mp.mpf(10)**-15
RESULTS = []


def dd(f, xs):
    tot = 0
    for w in _W:
        s = R * w
        p = f(s) * s
        for x in xs:
            p /= (s - x)
        tot += p
    return tot / NQ


def multisets(idx, k):
    return list(itertools.combinations_with_replacement(idx, k))


def dF(f, xs, labels):
    """partial derivative of f[xs] w.r.t. the labelled nodes (Cor. 3.12)."""
    m = [0] * len(xs)
    for i in labels:
        m[i] += 1
    return mp.fprod(mp.factorial(a) for a in m) * dd(f, list(xs) + [xs[i] for i in labels])


def pair(name, lhs, good, bad):
    e_good = abs(lhs - good) / max(1, abs(good))
    e_bad = abs(lhs - bad) / max(1, abs(good))
    ok = e_good < TOL and e_bad > 10**10 * TOL   # mutation detected by a wide margin
    RESULTS.append(ok)
    print(f"{'OK ' if ok else 'FAIL'} {name:46s} stated {mp.nstr(e_good, 2):>8s}"
          f"   mutated {mp.nstr(e_bad, 2):>8s}")


# 1. Exponential hierarchy (6.x, k = 3) with and without the factor 1/2.
tau = mp.mpf("1.3")
for q in (1, 3):
    xs = [mp.mpf(random.uniform(-1, 1)) for _ in range(q + 1)]
    ef = lambda z: mp.e**(-tau * z)
    E = dd(ef, xs)
    dtauE = mp.diff(lambda t: dd(lambda z: mp.e**(-t * z), xs), tau)
    L3 = sum(xs[ms[0]] * dd(ef, xs + [xs[i] for i in ms])
             for ms in multisets(range(q + 1), 3))
    S = sum(min(i, j) * dF(ef, xs, (i, j))
            for i in range(q + 1) for j in range(q + 1))
    pair(f"hierarchy k=3, factor 1/2 dropped, q={q}", L3,
         tau**3 / 6 * dtauE - S / 2, tau**3 / 6 * dtauE - S)

# 2. Lowering identity with integer weights i_1 replaced by nodes x_{i_1}.
f = lambda z: mp.e**((mp.mpf("0.7") - mp.mpf("0.3") * 1j) * z)
c = mp.mpf("0.7") - mp.mpf("0.3") * 1j
q = 3
xs = [mp.mpc(random.uniform(-1, 1), random.uniform(-.5, .5)) for _ in range(q + 1)]
for k in (2, 3):
    Lk = sum(xs[ms[0]] * dd(f, xs + [xs[i] for i in ms])
             for ms in multisets(range(q + 1), k))
    zfk = lambda z: z * c**k * f(z)
    B_int = sum(ms[0] * dd(f, xs + [xs[i] for i in ms])
                for ms in multisets(range(q + 1), k - 1))
    B_node = sum(xs[ms[0]] * dd(f, xs + [xs[i] for i in ms])
                 for ms in multisets(range(q + 1), k - 1))
    head = dd(zfk, xs) / mp.factorial(k)
    pair(f"lowering k={k}, node weights in B_(k-1)", Lk,
         head - B_int, head - B_node)

# 3. Cyclic symmetrization with positions 0 and n merged.
for conf in (False, True):
    n = 4
    xc = [mp.mpf(random.uniform(-1, 1)) for _ in range(n)]
    if conf:
        xc[2] = xc[0]
    g = lambda z: 1 / (4 - z)
    for k in (1, 2):
        rhs = sum(xc[a] * dF(g, xc, (a,) + s) for a in range(n)
                  for s in itertools.product(range(n), repeat=k)) / mp.factorial(k)
        good = bad = 0
        for s in range(n):
            base = [xc[(s + i) % n] for i in range(n + 1)]
            for ms in multisets(range(n + 1), k):
                good += xc[(s + ms[0]) % n] * dd(g, base + [xc[(s + i) % n] for i in ms])
            for ms in multisets(range(n), k):   # label n never used
                bad += xc[(s + ms[0]) % n] * dd(g, base + [xc[(s + i) % n] for i in ms])
        pair(f"cyclic k={k} conf={conf}, labels 0 and n merged", rhs, good, bad)

print(f"{sum(RESULTS)}/{len(RESULTS)} mutation tests behaved as expected")
