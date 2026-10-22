from sympy import primerange, isprime
from sympy.polys.galoistools import gf_from_int_poly, gf_pow_mod, gf_sub, gf_gcd, gf_edf_zassenhaus, gf_monic
from sympy.polys.domains import ZZ
from .core import greedy3

P240 = 1081034284409
F0 = [1, 91354, 228370, -20, -228385, -91348, 1]
F1 = [23667000, 6549182, -338632045, -473340000, -16372955, 135452818, 23667000]


def roots(f, p):
    F = gf_from_int_poly(f, p)
    if len(F) <= 1:
        return []
    F = gf_monic(F, p, ZZ)[1]
    Xp = gf_pow_mod([1, 0], p, F, p, ZZ)
    G = gf_gcd(gf_sub(Xp, [1, 0], p, ZZ), F, p, ZZ)
    if len(G) <= 1:
        return []
    return sorted(int((-fac[1]) % p) for fac in gf_edf_zassenhaus(G, 1, p, ZZ))


def factor_base(f, lo, hi):
    out = []
    for r in primerange(lo, hi):
        if f[0] % r == 0:
            continue
        for t in roots(f, r):
            out.append((r, t))
    return out


def special_q_basis(q, rho):
    b = greedy3([(q, 0, 0), ((-rho) % q, 1, 0), ((-rho * rho) % q, 0, 1)], (1, 1, 1))
    return [tuple(int(x) for x in v) for v in b]


def ideal_lattices(M, fb, q):
    out = []
    skipped = 0
    for r, t in fb:
        if r == q:
            continue
        t2 = (t * t) % r
        u0 = (M[0][0] + M[0][1] * t + M[0][2] * t2) % r
        u1 = (M[1][0] + M[1][1] * t + M[1][2] * t2) % r
        u2 = (M[2][0] + M[2][1] * t + M[2][2] * t2) % r
        if u0 == 0:
            skipped += 1
            continue
        inv = pow(u0, -1, r)
        B = (u1 * inv) % r
        C = (u2 * inv) % r
        if B == 0 or C == 0:
            skipped += 1
            continue
        out.append((r, B, C, t))
    return out, skipped


def special_qs(f, lo, count):
    out = []
    q = lo
    while len(out) < count:
        if isprime(q):
            for rho in roots(f, q):
                out.append((q, rho))
                if len(out) == count:
                    break
        q += 1
    return out
