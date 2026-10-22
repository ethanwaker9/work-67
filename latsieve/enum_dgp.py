import math
from .core import greedy3


def sieve(r, B, C, box, arr, val=1, stats=None):
    I, J, K = box.I, box.J, box.K
    IJ = box.IJ
    off = box.off
    lo, hi = box.lo, box.hi
    w = ((J * K) ** 2, (I * K) ** 2, (I * J) ** 2)
    b = greedy3([(r, 0, 0), ((-B) % r, 1, 0), ((-C) % r, 0, 1)], w)
    h = (I / 2.0, J / 2.0, K / 2.0)
    cen = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0)
    s = [[b[i][j] / h[j] for j in range(3)] for i in range(3)]
    tau = [cen[j] / h[j] for j in range(3)]
    bs = []
    mu = [[0.0] * 3 for _ in range(3)]
    nb = []
    for i in range(3):
        v = list(s[i])
        for j in range(i):
            mu[i][j] = sum(s[i][k] * bs[j][k] for k in range(3)) / nb[j]
            v = [v[k] - mu[i][j] * bs[j][k] for k in range(3)]
        bs.append(v)
        nb.append(sum(x * x for x in v))
    tc = [sum(tau[k] * bs[j][k] for k in range(3)) / nb[j] for j in range(3)]
    R2 = 3.0 * (1.0 + 1e-9) + 1e-9
    nodes = leaves = hits = 0
    b0, b1, b2 = b
    c3 = tc[2]
    r3 = math.sqrt(R2 / nb[2])
    for x3 in range(math.ceil(c3 - r3), math.floor(c3 + r3) + 1):
        nodes += 1
        d3 = (x3 - c3) ** 2 * nb[2]
        rem3 = R2 - d3
        if rem3 < 0:
            continue
        c2 = tc[1] - mu[2][1] * x3
        r2 = math.sqrt(rem3 / nb[1])
        for x2 in range(math.ceil(c2 - r2), math.floor(c2 + r2) + 1):
            nodes += 1
            d2 = (x2 - c2) ** 2 * nb[1]
            rem2 = rem3 - d2
            if rem2 < 0:
                continue
            c1 = tc[0] - mu[1][0] * x2 - mu[2][0] * x3
            r1 = math.sqrt(rem2 / nb[0])
            a = math.ceil(c1 - r1)
            e = math.floor(c1 + r1)
            px = a * b0[0] + x2 * b1[0] + x3 * b2[0]
            py = a * b0[1] + x2 * b1[1] + x3 * b2[1]
            pz = a * b0[2] + x2 * b1[2] + x3 * b2[2]
            for x1 in range(a, e + 1):
                leaves += 1
                if lo[0] <= px <= hi[0] and lo[1] <= py <= hi[1] and lo[2] <= pz <= hi[2]:
                    arr[px + I * py + IJ * pz + off] += val
                    hits += 1
                px += b0[0]
                py += b0[1]
                pz += b0[2]
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + nodes
        stats["leaves"] = stats.get("leaves", 0) + leaves
        stats["hits"] = stats.get("hits", 0) + hits
    return hits
