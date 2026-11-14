"""Numerical checks for Chapter 7 (conditioning and floating-point error).

Run:  python3 code/ch07_checks.py

Double-precision computations use Python floats (IEEE binary64, u = 2^-53).
References use mpmath at 80 digits.
"""
import itertools
import math
import random

import mpmath as mp

mp.mp.dps = 80
U = 2.0**-53
random.seed(7)
RESULTS = []


def report(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'OK ' if ok else 'FAIL'} {name:62s} {info}")


def table(fvals, xs):
    """Top entry of the divided-difference table (Eq. 2.x), in the
    arithmetic of the inputs."""
    d = list(fvals)
    n = len(xs)
    for k in range(1, n):
        d = [(d[i + 1] - d[i]) / (xs[i + k] - xs[i]) for i in range(n - k)]
    return d[0]


def ref_exp(xs, c=1):
    """e^{c[x]} at 80 digits via the Opitz matrix exponential."""
    q = len(xs) - 1
    Z = mp.matrix(q + 1, q + 1)
    for i, x in enumerate(xs):
        Z[i, i] = c * mp.mpf(x)
        if i < q:
            Z[i, i + 1] = 1
    return mp.mpf(c)**q * mp.expm(Z)[0, q]


def M_abs(fvals, xs):
    """sum_j |f_j| / prod_{k != j} |x_j - x_k|  (computed in mpmath)."""
    tot = 0
    for j, (fj, xj) in enumerate(zip(fvals, xs)):
        p = mp.mpf(1)
        for k, xk in enumerate(xs):
            if k != j:
                p *= abs(mp.mpf(xj) - mp.mpf(xk))
        tot += abs(mp.mpf(fj)) / p
    return tot


def path_weight_sum(xs):
    """For each leaf j, sum over paths of |product of path factors|
    (computed by running the table recursion on |.|)."""
    n = len(xs)
    out = []
    for j in range(n):
        d = [mp.mpf(1) if i == j else mp.mpf(0) for i in range(n)]
        for k in range(1, n):
            d = [(abs(d[i + 1]) + abs(d[i])) / abs(mp.mpf(xs[i + k]) - mp.mpf(xs[i])) for i in range(n - k)]
        out.append(d[0])
    return out


# ---------------------------------------------------------------------------
print("== 7.1 path signs and the forward error bound for the table")
ok_sign, ok_bound, worst_ratio = True, True, 0
for trial in range(300):
    q = random.randint(2, 14)
    xs = sorted(random.uniform(-3, 3) for _ in range(q + 1))
    if random.random() < 0.5:
        xs = xs[::-1]
    pw = path_weight_sum(xs)
    for j in range(q + 1):
        w = mp.mpf(1)
        for k in range(q + 1):
            if k != j:
                w *= abs(mp.mpf(xs[j]) - mp.mpf(xs[k]))
        if abs(pw[j] * w - 1) > mp.mpf(10)**-60:
            ok_sign = False
    fv = [math.exp(x) for x in xs]
    comp = table(fv, xs)
    exact = ref_exp(xs)
    M = M_abs([mp.e**mp.mpf(x) for x in xs], xs)
    bound = ((1 + mp.mpf(U))**(3 * q + 1) - 1) * M
    err = abs(mp.mpf(comp) - exact)
    if err > bound:
        ok_bound = False
    worst_ratio = max(worst_ratio, float(err / bound))
report("monotone order: all paths to a leaf have one sign", ok_sign)
report("table error <= ((1+u)^(3q+1)-1) * M  (300 sorted sets)", ok_bound, f"max err/bound={worst_ratio:.2e}")

# unsorted orders can have larger path sums, and larger errors
ratios = []
for trial in range(200):
    q = 10
    xs = sorted(random.uniform(0, 1) for _ in range(q + 1))
    ys = xs[:]
    random.shuffle(ys)
    M = M_abs([1] * (q + 1), xs)
    P = sum(path_weight_sum(ys))
    ratios.append(float(P / M))
report("shuffled order: path sum >= sorted path sum", min(ratios) >= 1 - 1e-12,
       f"median ratio={sorted(ratios)[100]:.1e}, max={max(ratios):.1e}")

errs_sorted, errs_shuf = [], []
for trial in range(200):
    q = 12
    xs = sorted(random.uniform(0, 2) for _ in range(q + 1))
    ys = xs[:]
    random.shuffle(ys)
    ex = ref_exp(xs)
    errs_sorted.append(float(abs(table([math.exp(x) for x in xs], xs) - ex) / ex))
    errs_shuf.append(float(abs(table([math.exp(x) for x in ys], ys) - ex) / ex))
med_s = sorted(errs_sorted)[100]
med_u = sorted(errs_shuf)[100]
report("median error: shuffled order worse than sorted", med_u > med_s, f"sorted {med_s:.1e}, shuffled {med_u:.1e}")

print("== Chapter 2 cancellation example")
xs = [0.5 + 1e-3 * j for j in range(8)]
comp = table([math.exp(x) for x in xs], xs)
ex = ref_exp([mp.mpf("0.5") + mp.mpf("0.001") * j for j in range(8)])
report("table value -5.73e2", abs(comp - (-573)) < 1, f"{comp:.4g}")
report("reference 3.28e-4", abs(ex - mp.mpf("3.28e-4")) < mp.mpf("5e-7"), f"{mp.nstr(ex, 6)}")
kv = M_abs([mp.e**mp.mpf(x) for x in xs], xs) / ex
report("kappa_val for that example = 1.28e23 = ((e^h+1)/(e^h-1))^7", 1.27e23 < kv < 1.29e23, f"{mp.nstr(kv, 3)}")

print("== 7.2 condition numbers")
# kappa_val is attained: perturb f_j by eps*|f_j|*sign(c_j)
xs = sorted(random.uniform(-1, 1) for _ in range(9))
xm = [mp.mpf(x) for x in xs]
fv = [mp.e**x for x in xm]
eps = mp.mpf(10)**-30
c = []
for j in range(len(xm)):
    w = mp.mpf(1)
    for k in range(len(xm)):
        if k != j:
            w *= xm[j] - xm[k]
    c.append(1 / w)
pert = [f + eps * abs(f) * mp.sign(cj) for f, cj in zip(fv, c)]
F = sum(f * cj for f, cj in zip(fv, c))
Fp = sum(f * cj for f, cj in zip(pert, c))
report("kappa_val attained by sign-matched perturbation",
       abs((Fp - F) / F / eps - M_abs(fv, xm) / F) < mp.mpf(10)**-20)

# node perturbations for exp: e^{-d} F <= F(x + D) <= e^{d} F, |D_j| <= d
ok = True
for trial in range(300):
    q = random.randint(1, 12)
    xs = [mp.mpf(random.uniform(-5, 5)) for _ in range(q + 1)]
    d = mp.mpf(random.uniform(0, 0.5))
    D = [d * mp.mpf(random.uniform(-1, 1)) for _ in xs]
    F = ref_exp(xs)
    Fp = ref_exp([x + e for x, e in zip(xs, D)])
    if not (mp.e**-d * F * (1 - mp.mpf(10)**-60) <= Fp <= mp.e**d * F * (1 + mp.mpf(10)**-60)):
        ok = False
report("exp: absolute node perturbation d moves F by factor in [e^-d, e^d]", ok)

# kappa_nodes (absolute) = sum |f[x,x_j]| / |f[x]| = 1 for exp, real nodes
xs = [mp.mpf(random.uniform(-2, 2)) for _ in range(8)]
F = ref_exp(xs)
s = sum(ref_exp(xs + [x]) for x in xs)
report("exp: sum_j e^[x,x_j] / e^[x] = 1", abs(s / F - 1) < mp.mpf(10)**-60)
# relative node condition number <= max|x_j|
kr = sum(abs(x) * ref_exp(xs + [x]) for x in xs) / F
report("exp: relative node condition <= max|x_j|", kr <= max(abs(x) for x in xs), f"{mp.nstr(kr, 4)} <= {mp.nstr(max(abs(x) for x in xs), 4)}")

# thermal kernel: |E| relative change <= e^{beta d} - 1
ok = True
for trial in range(100):
    q = random.randint(1, 10)
    beta = mp.mpf(random.uniform(0.1, 10))
    E = [mp.mpf(random.uniform(-2, 2)) for _ in range(q + 1)]
    d = mp.mpf(random.uniform(0, 0.2))
    W = ref_exp(E, -beta)
    Wp = ref_exp([e + d * mp.mpf(random.uniform(-1, 1)) for e in E], -beta)
    if abs(Wp / W - 1) > mp.e**(beta * d) - 1 + mp.mpf(10)**-50:
        ok = False
report("thermal: |dW/W| <= e^{beta d} - 1", ok)

# oscillatory: e^{-it[x]} vanishes at arithmetic progressions with t*Delta = 2 pi
t = mp.mpf(3)
Delta = 2 * mp.pi / t
xs = [j * Delta for j in range(6)]
Fz = ref_exp(xs, -1j * t) if False else None
q = 5
Z = mp.matrix(q + 1, q + 1)
for i, x in enumerate(xs):
    Z[i, i] = -1j * t * x
    if i < q:
        Z[i, i + 1] = 1
Fz = (-1j * t)**q * mp.expm(Z)[0, q]
report("oscillatory kernel vanishes on the resonant progression", abs(Fz) < mp.mpf(10)**-60, f"|F|={mp.nstr(abs(Fz), 3)}")

# Peano-kernel bound for smooth perturbations: |g[x]| <= max|g^(q)|/q!
ok = True
for trial in range(100):
    q = random.randint(1, 8)
    xs = sorted(mp.mpf(random.uniform(0, 1)) for _ in range(q + 1))
    om = mp.mpf(random.uniform(1, 20))
    g = lambda z: mp.sin(om * z)
    cj = []
    for j in range(q + 1):
        w = mp.mpf(1)
        for k in range(q + 1):
            if k != j:
                w *= xs[j] - xs[k]
        cj.append(1 / w)
    gx = sum(g(x) * c for x, c in zip(xs, cj))
    if abs(gx) > om**q / mp.factorial(q) * (1 + mp.mpf(10)**-30):
        ok = False
report("Peano: |g[x]| <= max|g^(q)|/q! for g = sin(omega z)", ok)

print("== 7.3 clustered nodes")
ok = True
for trial in range(200):
    q = random.randint(2, 15)
    a = mp.mpf(random.uniform(-3, 3))
    h = mp.mpf(10)**random.uniform(-4, 0.5)
    xs = sorted(a + h * mp.mpf(random.random()) for _ in range(q + 1))
    xs[0], xs[-1] = a, a + h
    F = ref_exp(xs)
    kv = M_abs([mp.e**x for x in xs], xs) / F
    lb = mp.factorial(q + 1) * mp.e**(-h) / h**q
    if kv < lb * (1 - mp.mpf(10)**-40):
        ok = False
report("cluster lower bound kappa_val >= (q+1)! e^{-h} / h^q", ok)

# equally spaced: kappa_val = e^{-?}... check the closed form for exp
q, h = 10, mp.mpf("0.01")
xs = [j * h for j in range(q + 1)]
F = ref_exp(xs)
kv = M_abs([mp.e**x for x in xs], xs) / F
closed = (1 + mp.e**h)**q / (mp.e**h - 1)**q
report("equispaced exp: kappa_val = ((e^h+1)/(e^h-1))^q", abs(kv / closed - 1) < mp.mpf(10)**-50, f"{mp.nstr(kv, 5)}")

# observed table error tracks q*u*kappa_val (within a factor 100 below, never above bound)
pairs = []
for trial in range(150):
    q = random.choice([5, 10, 15])
    h = 10**random.uniform(-3, 1)
    xs = sorted(random.uniform(0, h) for _ in range(q + 1))
    ex = ref_exp(xs)
    err = abs(mp.mpf(table([math.exp(x) for x in xs], xs)) - ex) / ex
    kv = M_abs([mp.e**mp.mpf(x) for x in xs], xs) / ex
    pairs.append((float(err), float(kv)))
ok_upper = all(e <= 3.2 * 15 * U * k for e, k in pairs)
big = [(e, k) for e, k in pairs if k > 1e3]
frac = sum(1 for e, k in big if e >= 1e-3 * U * k) / max(1, len(big))
report("observed error <= 3(q+1) u kappa_val", ok_upper)
report("observed error >= 1e-3 u kappa_val in most cases (kappa>1e3)", frac > 0.9, f"fraction {frac:.2f}")

print("== 7.4 diagnostics and extended precision")
# precision rule: dps >= 16 + log10(kappa_val) + 3 makes the table accurate to ~1e-16
ok = True
for trial in range(60):
    q = random.randint(3, 25)
    h = mp.mpf(10)**random.uniform(-6, 0)
    xs = sorted(mp.mpf(random.uniform(0, 1)) * h for _ in range(q + 1))
    F = ref_exp(xs)
    kv = M_abs([mp.e**x for x in xs], xs) / F           # estimate in double is fine
    dps = 16 + int(mp.ceil(mp.log10(kv))) + 3
    with mp.workdps(dps):
        comp = table([mp.e**x for x in xs], [+x for x in xs])
    if abs(comp - F) / F > mp.mpf(10)**-16:
        ok = False
report("table at 19 + log10(kappa_val) digits has relative error < 1e-16", ok)

# kappa_val is computed accurately in double precision (positive terms only)
q = 12
xs = sorted(random.uniform(0, 0.01) for _ in range(q + 1))
Md = 0.0
for j in range(q + 1):
    p = 1.0
    for k in range(q + 1):
        if k != j:
            p *= abs(xs[j] - xs[k])
    Md += math.exp(xs[j]) / p
Mm = M_abs([mp.e**mp.mpf(x) for x in xs], xs)
report("M computed in double has relative error < 10 q u", abs(Md - Mm) / Mm < 10 * q * U, f"{float(abs(Md - Mm) / Mm):.1e}")

# translation/permutation spread as an error estimator
good = 0
trials = 100
for trial in range(trials):
    q = 10
    h = 10**random.uniform(-2.5, 0)
    xs = sorted(random.uniform(0, h) for _ in range(q + 1))
    ex = float(ref_exp(xs))
    vals = []
    for s in (0.0, 0.37, -0.81, 1.3):
        ys = [x + s for x in xs]
        vals.append(table([math.exp(y) for y in ys], ys) * math.exp(-s))
    est = (max(vals) - min(vals)) / abs(ex)
    true = abs(vals[0] - ex) / abs(ex)
    if est >= 0.1 * true and est <= 1e3 * true + 1e-15:
        good += 1
report("translation spread estimates the error within [0.1, 1000]x", good >= 0.9 * trials, f"{good}/{trials}")

# Jensen sandwich catches gross errors
xs = [0.5 + 1e-3 * j for j in range(8)]
comp = table([math.exp(x) for x in xs], xs)
lo = math.exp(sum(xs) / 8) / math.factorial(7)
hi = sum(math.exp(x) for x in xs) / 8 / math.factorial(7)
report("Jensen sandwich flags the Chapter 2 value", not (lo <= comp <= hi))

print("== listing and partial clusters")


def exp_dd_reference(xs, digits=16):
    """Listing 7.1 (same code)."""
    xs = sorted(xs)
    q = len(xs) - 1
    log_M = math.log(q + 1) + max(
        xj - sum(math.log(abs(xj - xk)) for xk in xs if xk != xj)
        for xj in xs)
    log_F = sum(xs) / (q + 1) - math.lgamma(q + 1)
    extra = max(0.0, (log_M - log_F) / math.log(10))
    with mp.workdps(int(digits + extra + math.log10(3 * q + 2)) + 3):
        d = [mp.exp(mp.mpf(x)) for x in xs]
        for k in range(1, q + 1):
            d = [(d[i + 1] - d[i]) / (mp.mpf(xs[i + k]) - mp.mpf(xs[i]))
                 for i in range(q + 1 - k)]
        return +d[0]


ok, worst = True, 0
for trial in range(60):
    q = random.randint(1, 30)
    h = 10**random.uniform(-8, 1)
    c0 = random.uniform(-20, 20)
    xs = list({c0 + h * random.random() for _ in range(q + 1)})
    if len(xs) < 2:
        continue
    r = exp_dd_reference(xs)
    ex = ref_exp(sorted(xs))
    e = abs(r - ex) / ex
    worst = max(worst, float(e))
    if e > mp.mpf(10)**-16:
        ok = False
report("Listing 7.1 accurate to 1e-16 (q<=30, h down to 1e-8, |x|<=20)", ok, f"worst {worst:.1e}")

ok = True
for trial in range(200):
    q = random.randint(3, 12)
    m = random.randint(2, q + 1)
    h = mp.mpf(10)**random.uniform(-4, -1)
    cl = [mp.mpf(random.random()) * h for _ in range(m)]
    rest = [mp.mpf(random.uniform(-2, 2)) for _ in range(q + 1 - m)]
    xs = sorted(cl + rest)
    D = xs[-1] - xs[0]
    F = ref_exp(xs)
    kv = M_abs([mp.e**x for x in xs], xs) / F
    lb = m * min(mp.e**x for x in cl) / (h**(m - 1) * D**(q + 1 - m) * F)
    if kv < lb * (1 - mp.mpf(10)**-40):
        ok = False
report("partial-cluster bound (7.15)", ok)

# snapping the Chapter 2 nodes to their mean
xs = [mp.mpf("0.5") + mp.mpf("0.001") * j for j in range(8)]
xbar = sum(xs) / 8
snap = mp.e**xbar / mp.factorial(7)
ex = ref_exp(xs)
dmax = max(abs(x - xbar) for x in xs)
report("snapping to the mean: error <= e^delta - 1", abs(snap / ex - 1) <= mp.e**dmax - 1, f"{mp.nstr(abs(snap / ex - 1), 3)} <= {mp.nstr(mp.e**dmax - 1, 3)}")

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} checks passed")
