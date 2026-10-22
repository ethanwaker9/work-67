import random
import time

import numpy as np
from sympy import isprime

from common import Box, save
from latsieve import space_sieve, hakt, ntv
from latsieve.transitions import phi_grid, slab_vectors, coverage_transitions


def tmap(r, B, C, I, J):
    x, y, z = phi_grid(r, B, C, I, J)
    if len(np.unique(z)) < I * J:
        return None
    o = np.argsort(z)
    xs, ys, zs = x[o], y[o], z[o]
    nx = np.roll(xs, -1)
    ny = np.roll(ys, -1)
    dz = (np.roll(zs, -1) - zs) % r
    T = {}
    for a, b, c, d, e in zip(xs.tolist(), ys.tolist(), (nx - xs).tolist(), (ny - ys).tolist(), dz.tolist()):
        T[(a, b)] = (c, d, e)
    return T


def first_landing(L, q, I, J):
    for v in L:
        if 0 <= q[0] + v[0] < I and 0 <= q[1] + v[1] < J:
            return v
    return None


def coverage(T, L, I, J):
    Ls = sorted(set(L), key=lambda v: v[2])
    good = 0
    for q, t in T.items():
        if first_landing(Ls, q, I, J) == t:
            good += 1
    return good / len(T), set(T.values()) <= set(Ls)


def run():
    rng = random.Random(51)
    out = []
    for I in (32, 64, 128, 256):
        J = I
        K = 4 * I
        box = Box(I, J, K)
        rows = []
        while len(rows) < (120 if I <= 64 else 40):
            r = rng.randrange(4 * I * J, I * J * K // 8)
            if not isprime(r):
                continue
            B, C = rng.randrange(1, r), rng.randrange(1, r)
            T = tmap(r, B, C, I, J)
            if T is None:
                continue
            tv = set(T.values())
            zmax = max(v[2] for v in tv)
            t0 = time.perf_counter()
            Z = max(1, (4 * r) // (I * J))
            passes = 0
            while True:
                passes += 1
                V = slab_vectors(r, B, C, I, J, Z)
                ch, ok = coverage_transitions(V, I, J)
                if ok:
                    break
                Z *= 2
            t_alg = time.perf_counter() - t0
            exact = set(ch) == tv
            _, lp = space_sieve.candidates(r, B, C, box)
            ss_cov, ss_all = coverage(T, lp, I, J)
            u, v, w = hakt.basis(r, B, C, I, J)
            hc = [x for x in (u, v, w) if x[2] > 0]
            for i in range(1, 4):
                for j in range(1, 4):
                    d = (i * u[0] + j * v[0], i * u[1] + j * v[1], i * u[2] + j * v[2])
                    if d[2] > 0:
                        hc.append(d)
            hk_cov, hk_all = coverage(T, hc, I, J)
            lev, Tn, Sn, cva = ntv.initialize(r, B, C, box, "global")
            nt_cov, nt_all = coverage(T, Tn[2], I, J)
            rows.append({"r": r, "ntv": len(tv), "zmax_over_gap": zmax * I * J / r, "slab": len(V), "passes": passes, "t_alg": t_alg, "exact": exact, "ss_cov": ss_cov, "ss_all": ss_all, "hakt_cov": hk_cov, "hakt_all": hk_all, "ntv_cov": nt_cov, "ntv_all": nt_all})
        out.append({"I": I, "rows": rows})
        print(I, np.mean([x["ntv"] for x in rows]), np.mean([x["ss_all"] for x in rows]), np.mean([x["ss_cov"] for x in rows]), np.mean([x["hakt_cov"] for x in rows]), np.mean([x["ntv_cov"] for x in rows]), all(x["exact"] for x in rows), np.mean([x["t_alg"] for x in rows]) * 1e3, np.mean([x["slab"] for x in rows]), flush=True)
        save("tvlists.json", out)


if __name__ == "__main__":
    run()
