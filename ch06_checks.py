"""Numerical checks for Chapter 6 (weighted sums of confluent divided differences).

Run:  python3 code/ch06_checks.py

Divided differences are computed from the Cauchy integral (Theorem 4.x) by the
trapezoid rule on a circle, which is exact to working precision for the test
functions used here and handles repeated nodes with no special treatment.
Node derivatives are taken independently with mpmath's numerical
differentiation where it matters (labelled-node formula); elsewhere the
proved formula of Corollary 3.12 is used.
"""
import itertools
import math
import random

import mpmath as mp

mp.mp.dps = 40
random.seed(6)
NQ = 240          # quadrature points
R = mp.mpf(2)     # contour radius; all nodes lie in |z| <= 1.1

FUNCS = {
    "exp(cz)": (lambda z: mp.e**((mp.mpf("0.7") - mp.mpf("0.3") * 1j) * z)),
    "1/(4-z)": (lambda z: 1 / (4 - z)),
    "sin(2z)": (lambda z: mp.sin(2 * z)),
}
FD = {  # derivatives of order k, for f^(k)
    "exp(cz)": (lambda z, k: (mp.mpf("0.7") - mp.mpf("0.3") * 1j)**k * FUNCS["exp(cz)"](z)),
    "1/(4-z)": (lambda z, k: mp.factorial(k) / (4 - z)**(k + 1)),
    "sin(2z)": (lambda z, k: 2**k * mp.sin(2 * z + k * mp.pi / 2)),
}

_W = [mp.expjpi(2 * mp.mpf(j) / NQ) for j in range(NQ)]


_CACHE = {}


def dd(f, xs, rad=None):
    """f[x_0,...,x_q] by the Cauchy integral (any multiplicities)."""
    rad = R if rad is None else rad
    key = (f, tuple(xs), rad)   # the object itself, so a recycled id cannot hit a stale entry
    if key in _CACHE:
        return _CACHE[key]
    tot = 0
    for w in _W:
        s = rad * w
        p = f(s) * s
        for x in xs:
            p /= (s - x)
        tot += p
    _CACHE[key] = tot / NQ
    return _CACHE[key]


def rnodes(n, conf=False):
    xs = [mp.mpc(random.uniform(-1, 1), random.uniform(-.5, .5)) for _ in range(n)]
    if conf and n >= 3:
        xs[1] = xs[0]
        xs[-1] = xs[0]
    return xs


RESULTS = []


def check(name, a, b, tol=mp.mpf(10)**-25):
    err = abs(a - b) / max(1, abs(b))
    ok = err < tol
    RESULTS.append(ok)
    print(f"{'OK ' if ok else 'FAIL'} {name:60s} relerr={mp.nstr(err, 3)}")


def multisets(idx, k):
    return list(itertools.combinations_with_replacement(idx, k))


def mult(ms, q):
    m = [0] * (q + 1)
    for i in ms:
        m[i] += 1
    return m


def mfact(m):
    return math.prod(math.factorial(a) for a in m)


def dF(f, xs, labels):
    """partial derivative of F=f[xs] w.r.t. the labelled nodes in `labels`
    (a sequence, repetitions allowed), via Corollary 3.12."""
    m = mult(labels, len(xs) - 1)
    return mfact(m) * dd(f, list(xs) + [xs[i] for i in labels])


def tail(xs, a):
    return list(range(a, len(xs)))


# ---------------------------------------------------------------------------
print("== 6.1 master identity")
for fname, f in FUNCS.items():
    for conf in (False, True):
        q = 3
        xs = rnodes(q + 1, conf)
        t = [mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) * mp.mpf("0.01") for _ in xs]
        lhs = dd(f, [x + s for x, s in zip(xs, t)])
        rhs = 0
        K = 14
        for k in range(K + 1):
            for ms in multisets(range(q + 1), k):
                rhs += math.prod(t[i] for i in ms) * dd(f, xs + [xs[i] for i in ms])
        check(f"master identity {fname} conf={conf}", lhs, rhs, mp.mpf(10)**-24)

print("== 6.2 labelled nodes and blocks")
f = FUNCS["exp(cz)"]
y = rnodes(4)
y[1] = y[0]
# labelled-node formula at coincident values, independent derivative


def Fvec(*v):
    return dd(f, list(v))


d01 = mp.diff(Fvec, tuple(y), (1, 1, 0, 0))
d00 = mp.diff(Fvec, tuple(y), (2, 0, 0, 0))
check("labelled: d0 d1 F = f[y,y0,y1] (y0=y1)", d01, dd(f, y + [y[0], y[1]]), mp.mpf(10)**-15)
check("labelled: d0^2 F = 2 f[y,y0,y0] (y0=y1)", d00, 2 * dd(f, y + [y[0], y[0]]), mp.mpf(10)**-15)
# block powers
q = 4
y = rnodes(q + 1, True)
B = [0, 2, 3]
for m in (1, 2, 3):
    # D_B^m F / m! = sum over ordered label sequences / m!  (Cor 3.12)
    lhs = sum(dF(f, y, seq) for seq in itertools.product(B, repeat=m)) / mp.factorial(m)
    rhs = sum(dd(f, y + [y[i] for i in ms]) for ms in multisets(B, m))
    check(f"block powers m={m}", lhs, rhs)
eps = mp.mpf("0.07")
lhs = dd(f, [y[i] + (eps if i in B else 0) for i in range(q + 1)])
rhs = sum(eps**m * sum(dd(f, y + [y[i] for i in ms]) for ms in multisets(B, m)) for m in range(26))
check("block shift generating function", lhs, rhs)

# nested blocks B_k = {0..k-1}: mixed derivative
q = 3
y = rnodes(q + 1)
Bk = {k: list(range(k)) for k in range(1, q + 1)}


def nested_lhs(p):
    seqs = [itertools.product(Bk[k], repeat=p[k - 1]) for k in range(1, q + 1)]
    tot = 0
    for combo in itertools.product(*seqs):
        labels = [i for s in combo for i in s]
        tot += dF(f, y, labels)
    return tot


_NP = {}


def Np(p, m):
    """coefficient of z^m in prod_k (z_0+...+z_{k-1})^{p_k}"""
    if p not in _NP:
        poly = {(0,) * (q + 1): 1}
        for k in range(1, q + 1):
            for _ in range(p[k - 1]):
                new = {}
                for mono, c in poly.items():
                    for a in range(k):
                        mm = list(mono); mm[a] += 1; mm = tuple(mm)
                        new[mm] = new.get(mm, 0) + c
                poly = new
        _NP[p] = poly
    return _NP[p].get(tuple(m), 0)


for p in ((1, 1, 0), (0, 1, 1), (2, 1, 1), (1, 0, 2)):
    k = sum(p)
    rhs = 0
    for ms in multisets(range(q), k):
        m = mult(ms, q)
        rhs += mfact(m) * Np(p, m) * dd(f, y + [y[i] for i in ms])
    check(f"mixed nested blocks p={p}", nested_lhs(p), rhs)
# example q=2, p=(1,1)
y2 = rnodes(3)
lhs = dF(f, y2, [0, 0]) + dF(f, y2, [0, 1])
check("example D_B1 D_B2 F = 2f[y,y0,y0]+f[y,y0,y1]", lhs,
      2 * dd(f, y2 + [y2[0], y2[0]]) + dd(f, y2 + [y2[0], y2[1]]))
# 3.9: nodes S_j(alpha) = sum_{l>j} alpha_l; d/d alpha_k = D_{B_k}
alpha = [None] + [mp.mpc(random.uniform(-.5, .5), random.uniform(-.3, .3)) for _ in range(q)]


def Fal(*a):
    al = [None] + list(a)
    S = [sum(al[l] for l in range(j + 1, q + 1)) for j in range(q + 1)]
    return dd(f, S)


S = [sum(alpha[l] for l in range(j + 1, q + 1)) for j in range(q + 1)]
for k in range(1, q + 1):
    order = tuple(1 if i == k - 1 else 0 for i in range(q))
    lhs = mp.diff(Fal, tuple(alpha[1:]), order)
    rhs = sum(dd(f, S + [S[j]]) for j in range(k))
    check(f"nested block structure d/d alpha_{k}", lhs, rhs, mp.mpf(10)**-15)
# negative control (compendium 3.10), q=2: D_{B_2}F = f[S,S0]+f[S,S1] is not a single f[S,S_a]
alpha2 = [None, mp.mpf("0.3"), mp.mpf("-0.8")]
S2 = [alpha2[1] + alpha2[2], alpha2[2], 0]
g = FUNCS["exp(cz)"]
DB2 = dd(g, S2 + [S2[0]]) + dd(g, S2 + [S2[1]])
singles = [dd(g, S2 + [S2[a]]) for a in range(3)]
bad = all(abs(s_ - DB2) / abs(DB2) > 0.05 for s_ in singles)
RESULTS.append(bad)
print(f"{'OK ' if bad else 'FAIL'} D_B2 F is not a single divided difference (negative control)")
# Exercise: q=2, N_p(m0,m1,0) = binom(p2, m1)
okN = all(Np((p1, p2, 0)[:q] if q == 3 else (p1, p2), [p1 + p2 - m1, m1, 0, 0][:q + 1]) == math.comb(p2, m1)
          for p1 in range(4) for p2 in range(4) for m1 in range(p2 + 1))
RESULTS.append(okN)
print(f"{'OK ' if okN else 'FAIL'} N_p(m0,m1,0) = binom(p2,m1) (B_3 unused)")
# 3.14 multishift
u = [None] + [mp.mpf("0.06") * mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(q)]
U = [sum(u[k] for k in range(j + 1, q + 1)) for j in range(q + 1)]
lhs = dd(f, [y[j] + U[j] for j in range(q + 1)])
rhs = 0
for p in itertools.product(range(11), repeat=q):
    if sum(p) > 10:
        continue
    k = sum(p)
    coeff = 0
    for ms in multisets(range(q), k):
        m = mult(ms, q)
        n = Np(p, m)
        if n:
            coeff += mfact(m) * n * dd(f, y + [y[i] for i in ms])
    rhs += math.prod(u[i + 1]**p[i] for i in range(q)) / math.prod(math.factorial(a) for a in p) * coeff
check("multishift generating identity", lhs, rhs, mp.mpf(10)**-12)

print("== 6.3 weighted sums")
for fname, f in FUNCS.items():
    q = 3
    xs = rnodes(q + 1, True)
    F = dd(f, xs)
    for k in (1, 2, 3):
        A = sum(dd(f, xs + [xs[i] for i in ms]) for ms in multisets(range(q + 1), k))
        fk = lambda z, k=k: FD[fname](z, k)
        check(f"uniform append {fname} k={k}", A, dd(fk, xs) / mp.factorial(k))
    # block restriction (6) and (7)
    B = [1, 3]
    g = [mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(q + 1)]
    for k in (1, 2, 3):
        lhs = sum(dd(f, xs + [xs[i] for i in ms]) for ms in multisets(B, k))
        rhs = sum(dF(f, xs, s) for s in itertools.product(B, repeat=k)) / mp.factorial(k)
        check(f"block restriction {fname} k={k}", lhs, rhs)
        lhs = sum(sum(g[i] for i in ms) * dd(f, xs + [xs[i] for i in ms]) for ms in multisets(B, k))
        rhs = sum(g[a] * dF(f, xs, (a,) + s) for a in B for s in itertools.product(B, repeat=k - 1)) / mp.factorial(k - 1)
        check(f"block restriction weighted {fname} k={k}", lhs, rhs)
    # additive weight
    for k in (1, 2, 3):
        lhs = sum(sum(g[i] for i in ms) * dd(f, xs + [xs[i] for i in ms]) for ms in multisets(range(q + 1), k))
        rhs = sum(g[a] * dF(f, xs, (a,) + s) for a in range(q + 1)
                  for s in itertools.product(range(q + 1), repeat=k - 1)) / mp.factorial(k - 1)
        check(f"additive weight {fname} k={k}", lhs, rhs)
    # generating function in (u, lambda)
    uu, lam = mp.mpf("0.05"), mp.mpf("0.4")
    lhs = dd(f, [x + uu * mp.e**(lam * gi) for x, gi in zip(xs, g)])
    rhs = sum(uu**k * sum(mp.e**(lam * sum(g[i] for i in ms)) * dd(f, xs + [xs[i] for i in ms])
                          for ms in multisets(range(q + 1), k)) for k in range(20))
    check(f"weighted-shift generating function {fname}", lhs, rhs, mp.mpf(10)**-18)
    # order statistic, random symmetric Phi
    k = 3
    tab = {ms: mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) for ms in multisets(range(q + 1), k)}
    lhs = sum(tab[ms] * dd(f, xs + [xs[i] for i in ms]) for ms in tab)
    rhs = sum(tab[tuple(sorted(r))] * dF(f, xs, r) for r in itertools.product(range(q + 1), repeat=k)) / mp.factorial(k)
    check(f"order-statistic identity {fname}", lhs, rhs)
    for ell in (1, 2, 3):
        lhs = sum(xs[ms[ell - 1]] * dd(f, xs + [xs[i] for i in ms]) for ms in multisets(range(q + 1), k))
        rhs = sum(xs[sorted(r)[ell - 1]] * dF(f, xs, r) for r in itertools.product(range(q + 1), repeat=k)) / mp.factorial(k)
        check(f"fixed-position weight {fname} l={ell}", lhs, rhs)
    # universal coefficient, tail shift, multiset form, lowering
    for k in (1, 2, 3):
        Lk = sum(xs[ms[0]] * dd(f, xs + [xs[i] for i in ms]) for ms in multisets(range(q + 1), k))
        rhs = sum(xs[min(r)] * dF(f, xs, r) for r in itertools.product(range(q + 1), repeat=k)) / mp.factorial(k)
        check(f"universal coefficient {fname} k={k}", Lk, rhs)
        # tail shift: (1/k!) sum_a x_a (D_a^k - D_{a+1}^k) F
        def Dpow(a, kk):
            if a > q:
                return 0 if kk > 0 else F
            return sum(dF(f, xs, s) for s in itertools.product(tail(xs, a), repeat=kk))
        rhs = sum(xs[a] * (Dpow(a, k) - Dpow(a + 1, k)) for a in range(q + 1)) / mp.factorial(k)
        check(f"tail-shift form {fname} k={k}", Lk, rhs)
        rhs = sum(xs[ms[0]] / mfact(mult(ms, q)) * dF(f, xs, ms) for ms in multisets(range(q + 1), k))
        check(f"multiset form {fname} k={k}", Lk, rhs)
        zfk = lambda z, k=k: z * FD[fname](z, k)
        if k == 1:
            check(f"lowering identity {fname} k=1", Lk, dd(zfk, xs) - q * F)
        else:
            Bk1 = sum(ms[0] * dd(f, xs + [xs[i] for i in ms]) for ms in multisets(range(q + 1), k - 1))
            check(f"lowering identity {fname} k={k}", Lk, dd(zfk, xs) / mp.factorial(k) - Bk1)

print("== 6.3 exponential hierarchy")
tau = mp.mpf("1.3")
q = 4
xs = [mp.mpf(random.uniform(-1, 1)) for _ in range(q + 1)]
xs[2] = xs[0]
ef = lambda z: mp.e**(-tau * z)
E = dd(ef, xs)
dtauE = mp.diff(lambda t: dd(lambda z: mp.e**(-t * z), xs), tau)
for k in (1, 2, 3, 4):
    Lk = sum(xs[ms[0]] * dd(ef, xs + [xs[i] for i in ms]) for ms in multisets(range(q + 1), k))
    if k == 1:
        rhs = tau * dtauE - q * E
    else:
        Bk1 = sum(min(r) * dF(ef, xs, r) for r in itertools.product(range(q + 1), repeat=k - 1)) / mp.factorial(k - 1)
        rhs = -(-tau)**k / mp.factorial(k) * dtauE - Bk1
    check(f"exponential rooted hierarchy k={k}", Lk, rhs, mp.mpf(10)**-15)
# closed forms k=1,2
L1 = sum(xs[j] * dd(ef, xs + [xs[j]]) for j in range(q + 1))
check("closed form k=1", L1, (-xs[0] * tau - q) * E - tau * dd(ef, xs[1:]))
L2 = sum(xs[i] * dd(ef, xs + [xs[i], xs[j]]) for i in range(q + 1) for j in range(i, q + 1))
check("closed form k=2", L2, tau**2 * xs[0] / 2 * E + tau**2 / 2 * dd(ef, xs[1:])
      - sum(i * dd(ef, xs + [xs[i]]) for i in range(1, q + 1)))
# symmetrized
for k in (1, 2, 3):
    lhs = sum(sum(xs[i] for i in ms) * dd(ef, xs + [xs[i] for i in ms]) for ms in multisets(range(q + 1), k))
    check(f"position-symmetrized collapse k={k}", lhs,
          (-tau)**(k - 1) / mp.factorial(k - 1) * (tau * dtauE - q * E), mp.mpf(10)**-15)

print("== 6.5 cyclic symmetrization")
for fname, f in (("exp(-tau z)", ef), ("1/(4-z)", FUNCS["1/(4-z)"])):
    for conf in (False, True):
        n = 4
        xc = [mp.mpf(random.uniform(-1, 1)) for _ in range(n)]
        if conf:
            xc[2] = xc[0]
        FC = dd(f, xc)
        for k in (1, 2, 3):
            tot = 0
            for s in range(n):
                base = [xc[(s + i) % n] for i in range(n + 1)]
                for ms in multisets(range(n + 1), k):
                    tot += xc[(s + ms[0]) % n] * dd(f, base + [xc[(s + i) % n] for i in ms])
            mid = sum(sum(xc[a] for a in ms) * dd(f, xc + [xc[a] for a in ms]) for ms in multisets(range(n), k + 1))
            check(f"cyclic symmetrization {fname} conf={conf} k={k} (middle)", tot, mid)
            rhs = sum(xc[a] * dF(f, xc, (a,) + s) for a in range(n)
                      for s in itertools.product(range(n), repeat=k)) / mp.factorial(k)
            check(f"cyclic symmetrization {fname} conf={conf} k={k} (right)", tot, rhs)
            if f is ef:
                dtEC = mp.diff(lambda t: dd(lambda z: mp.e**(-t * z), xc), tau)
                check(f"closed-walk collapse conf={conf} k={k}", tot,
                      (-tau)**k / mp.factorial(k) * (tau * dtEC - (n - 1) * FC), mp.mpf(10)**-15)

print("== 6.5 full symmetric averaging")
for fname, f in (("exp(-tau z)", ef), ("1/(4-z)", FUNCS["1/(4-z)"])):
    for conf in (False, True):
        n = 4
        xc = [mp.mpf(random.uniform(-1, 1)) for _ in range(n)]
        if conf:
            xc[2] = xc[0]
        for k in (1, 2):
            def rooted(order):
                base = [xc[order[i % n]] for i in range(n + 1)]
                return sum(xc[order[ms[0] % n]] * dd(f, base + [xc[order[i % n]] for i in ms])
                           for ms in multisets(range(n + 1), k))
            cyc = sum(rooted([(s + i) % n for i in range(n)]) for s in range(n)) / n
            full = sum(rooted(list(pi)) for pi in itertools.permutations(range(n))) / mp.factorial(n)
            check(f"full symmetric average = cyclic average {fname} conf={conf} k={k}", full, cyc)

print("== 6.4 block shift")
f = FUNCS["1/(4-z)"]
a = rnodes(2)
ys = rnodes(3, True)
al = mp.mpf("0.05")
lhs = dd(f, a + [v + al for v in ys])
rhs = sum(al**k * sum(dd(f, a + ys + [ys[i] for i in ms]) for ms in multisets(range(3), k)) for k in range(30))
check("block shift expansion", lhs, rhs, mp.mpf(10)**-20)
for k in (1, 2, 3):
    gk = lambda v, k=k: dd(f, a + [v] * (k + 1), mp.mpf("2.5"))
    mid = sum(dd(f, a + ys + [ys[i] for i in ms]) for ms in multisets(range(3), k))
    # g is analytic in |v| < 2.5; take its divided difference on |z| = 1.5
    gdd = dd(gk, ys, mp.mpf("1.5"))
    check(f"block shift: middle sum = g[y] k={k}", mid, gdd, mp.mpf(10)**-12)

print("== 6.6 jets")
# first-order jet through the Frechet block matrix of the Opitz matrix (exp)
q = 4
xs = [mp.mpf(random.uniform(-1, 1)) for _ in range(q + 1)]
xs[3] = xs[1]
w = [mp.mpf(random.uniform(-1, 1)) for _ in range(q + 1)]
n = q + 1
M = mp.zeros(2 * n, 2 * n)
for i in range(n):
    M[i, i] = M[n + i, n + i] = xs[i]
    M[i, n + i] = w[i]
    if i < q:
        M[i, i + 1] = M[n + i, n + i + 1] = 1
EM = mp.expm(M)
check("first-order jet = Frechet block (exp)", EM[0, n + q],
      sum(w[l] * dd(mp.exp, xs + [xs[l]]) for l in range(n)), mp.mpf(10)**-25)
# second-order jet in one variable, epsilon nilpotent of index 3
P = 3
N = mp.zeros(P, P)
for i in range(P - 1):
    N[i, i + 1] = 1
M = mp.zeros(n * P, n * P)
for i in range(n):
    for r in range(P):
        M[i * P + r, i * P + r] = xs[i]
        if r < P - 1:
            M[i * P + r, i * P + r + 1] = w[i]
    if i < q:
        for r in range(P):
            M[i * P + r, (i + 1) * P + r] = 1
EM = mp.expm(M)
for k in (0, 1, 2):
    jet = EM[0 * P + 0, q * P + k]
    rhs = sum(math.prod(w[i] for i in ms) * dd(mp.exp, xs + [xs[i] for i in ms]) for ms in multisets(range(n), k))
    check(f"jet coefficient eps^{k} (Kronecker-Opitz)", jet, rhs, mp.mpf(10)**-25)
# mixed eps1 eps2 coefficient via two commuting nilpotents (2x2 each)
u = [mp.mpf(random.uniform(-1, 1)) for _ in range(n)]
v = [mp.mpf(random.uniform(-1, 1)) for _ in range(n)]
N2 = mp.matrix([[0, 1], [0, 0]])
I2 = mp.eye(2)
def kron(A, B):
    C = mp.zeros(A.rows * B.rows, A.cols * B.cols)
    for i in range(A.rows):
        for j in range(A.cols):
            for k in range(B.rows):
                for l in range(B.cols):
                    C[i * B.rows + k, j * B.cols + l] = A[i, j] * B[k, l]
    return C
e1, e2 = kron(N2, I2), kron(I2, N2)
blk = 4
M = mp.zeros(n * blk, n * blk)
for i in range(n):
    D = xs[i] * mp.eye(4) + u[i] * e1 + v[i] * e2
    for r in range(blk):
        for c in range(blk):
            M[i * blk + r, i * blk + c] = D[r, c]
    if i < q:
        for r in range(blk):
            M[i * blk + r, (i + 1) * blk + r] = 1
EM = mp.expm(M)
mixed = EM[0, q * blk + 3]          # coefficient of e1 e2 sits in corner (0,3) of the 4x4 block
rhs = sum((u[i] * v[j] + v[i] * u[j]) * dd(mp.exp, xs + [xs[i], xs[j]]) for i in range(n) for j in range(i, n))
check("mixed jet coefficient eps1 eps2", mixed, rhs, mp.mpf(10)**-25)
# kernel form with telescoping
gw = [mp.mpf(random.uniform(-1, 1)) for _ in range(n)]
lhs = sum(gw[i] * dd(mp.exp, xs + [xs[i], xs[j]]) for i in range(n) for j in range(i, n))
rhs = sum(gw[min(a, b)] * dF(mp.exp, xs, (a, b)) for a in range(n) for b in range(n)) / 2
check("double-extra-node kernel form", lhs, rhs)


def D2tail(s):
    return sum(dF(mp.exp, xs, (a, b)) for a in range(s, n) for b in range(s, n))


rhs2 = (gw[0] * D2tail(0) + sum((gw[s] - gw[s - 1]) * D2tail(s) for s in range(1, n))) / 2
check("kernel form via telescoping tail shifts", lhs, rhs2)

print("== extra: B tail form, general Euler, general position-symmetrized, listing")
for fname, f in FUNCS.items():
    q = 3
    xs = rnodes(q + 1, True)
    F = dd(f, xs)
    def Dp(a, kk):
        return sum(dF(f, xs, s_) for s_ in itertools.product(range(a, q + 1), repeat=kk)) if kk else F
    for k in (2, 3):
        Bk1 = sum(ms[0] * dd(f, xs + [xs[i] for i in ms]) for ms in multisets(range(q + 1), k - 1))
        check(f"B tail form {fname} k={k}", Bk1, sum(Dp(a, k - 1) for a in range(1, q + 1)) / mp.factorial(k - 1))
    zf1 = lambda z: z * FD[fname](z, 1)
    check(f"Euler relation {fname}", sum(xs[a] * dF(f, xs, (a,)) for a in range(q + 1)), dd(zf1, xs) - q * F)
    for k in (1, 2, 3):
        lhs = sum(sum(xs[i] for i in ms) * dd(f, xs + [xs[i] for i in ms]) for ms in multisets(range(q + 1), k))
        zfk = lambda z, k=k: z * FD[fname](z, k)
        fk1 = lambda z, k=k: FD[fname](z, k - 1)
        check(f"position-symmetrized general {fname} k={k}", lhs,
              (dd(zfk, xs) - q * dd(fk1, xs)) / mp.factorial(k - 1))
try:
    import numpy as np
    from scipy.linalg import expm_frechet

    def exp_dd_with_append_sum(x, w):
        q = len(x) - 1
        Z = np.diag(x) + np.diag(np.ones(q), 1)
        Fm, L = expm_frechet(Z, np.diag(w))
        return Fm[0, q], L[0, q]
    xr = [0.3, -0.5, 0.3, 0.9, 0.1]
    wr = [1.0, -2.0, 0.5, 0.25, 3.0]
    a, b = exp_dd_with_append_sum(np.array(xr), np.array(wr))
    xm = [mp.mpf(v) for v in xr]
    check("listing: e^[x]", mp.mpf(a), dd(mp.exp, xm), mp.mpf(10)**-13)
    check("listing: weighted append sum", mp.mpf(b),
          sum(mp.mpf(wr[l]) * dd(mp.exp, xm + [xm[l]]) for l in range(5)), mp.mpf(10)**-13)
except ImportError:
    print("scipy not available; listing not checked")

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
