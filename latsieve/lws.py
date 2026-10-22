from .core import greedy3, dual_basis


def setup(r, B, C, box):
    return setup_dual([(1, B, C), (0, r, 0), (0, 0, r)], r, box)


def setup_dual(D0, det, box):
    L0, L1, L2 = box.L
    D = greedy3(D0, (L0 * L0, L1 * L1, L2 * L2))
    D.sort(key=lambda d: abs(d[0]) * L0 + abs(d[1]) * L1 + abs(d[2]) * L2)
    return D, dual_basis(D, det)


def cofactor_rows(M):
    (a, b, c), (d, e, f), (g, h, i) = M
    return [(e * i - f * h, f * g - d * i, d * h - e * g), (c * h - b * i, a * i - c * g, b * g - a * h), (b * f - c * e, c * d - a * f, a * e - b * d)]


def chains(d1, d2, box):
    lo = box.lo
    L = box.L
    s0 = d1[0] * lo[0] + d1[1] * lo[1] + d1[2] * lo[2]
    m0 = d2[0] * lo[0] + d2[1] * lo[1] + d2[2] * lo[2]
    gens = []
    up = 0
    for i in range(3):
        gs = d1[i] * L[i]
        gm = d2[i] * L[i]
        if gs < 0 or (gs == 0 and gm < 0):
            s0 += gs
            m0 += gm
            gs, gm = -gs, -gm
        if gs == 0:
            up += gm
        elif gm or gs:
            gens.append((gs, gm))
    n = len(gens)
    for a in range(1, n):
        g = gens[a]
        j = a
        while j > 0 and gens[j - 1][1] * g[0] > g[1] * gens[j - 1][0]:
            gens[j] = gens[j - 1]
            j -= 1
        gens[j] = g
    low = []
    s, m = s0, m0
    for gs, gm in gens:
        low.append((s, m, gs, gm))
        s += gs
        m += gm
    high = []
    s, m = s0, m0 + up
    for gs, gm in reversed(gens):
        high.append((s, m, gs, gm))
        s += gs
        m += gm
    return s0, s, low, high


def sieve(r, B, C, box, arr, val=1, stats=None):
    D, b = setup(r, B, C, box)
    return run(D, b, r, box, arr, val, stats)


def sieve_basis(M, box, arr, val=1, stats=None):
    det = abs(M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D, b = setup_dual(cofactor_rows(M), det, box)
    return run(D, b, det, box, arr, val, stats)


def run(D, b, r, box, arr, val=1, stats=None):
    smin, smax, low, high = chains(D[0], D[1], box)
    k1lo = -((-smin) // r)
    k1hi = smax // r
    b1, b2, b3 = b
    lo0, lo1, lo2 = box.lo
    hi0, hi1, hi2 = box.hi
    I = box.I
    IJ = box.IJ
    off = box.off
    e0, e1, e2 = b3
    st = e0 + I * e1 + IJ * e2
    nlow = len(low)
    nhigh = len(high)
    jl = jh = 0
    planes = lines = hits = dead = 0
    for k1 in range(k1lo, k1hi + 1):
        s = k1 * r
        planes += 1
        while jl + 1 < nlow and low[jl + 1][0] <= s:
            jl += 1
        sa, ma, gs, gm = low[jl]
        k2lo = -((-(ma * gs + gm * (s - sa))) // (gs * r))
        while jh + 1 < nhigh and high[jh + 1][0] <= s:
            jh += 1
        sa, ma, gs, gm = high[jh]
        k2hi = (ma * gs + gm * (s - sa)) // (gs * r)
        c0 = k1 * b1[0] + k2lo * b2[0]
        c1 = k1 * b1[1] + k2lo * b2[1]
        c2 = k1 * b1[2] + k2lo * b2[2]
        for _ in range(k2hi - k2lo + 1):
            lines += 1
            tlo = -(1 << 62)
            thi = 1 << 62
            if e0 > 0:
                tlo = -((c0 - lo0) // e0)
                thi = (hi0 - c0) // e0
            elif e0 < 0:
                tlo = -((c0 - hi0) // e0)
                thi = (lo0 - c0) // e0
            elif c0 < lo0 or c0 > hi0:
                thi = tlo - 1
            if e1 > 0:
                v = -((c1 - lo1) // e1)
                if v > tlo:
                    tlo = v
                v = (hi1 - c1) // e1
                if v < thi:
                    thi = v
            elif e1 < 0:
                v = -((c1 - hi1) // e1)
                if v > tlo:
                    tlo = v
                v = (lo1 - c1) // e1
                if v < thi:
                    thi = v
            elif c1 < lo1 or c1 > hi1:
                thi = tlo - 1
            if e2 > 0:
                v = -((c2 - lo2) // e2)
                if v > tlo:
                    tlo = v
                v = (hi2 - c2) // e2
                if v < thi:
                    thi = v
            elif e2 < 0:
                v = -((c2 - hi2) // e2)
                if v > tlo:
                    tlo = v
                v = (lo2 - c2) // e2
                if v < thi:
                    thi = v
            elif c2 < lo2 or c2 > hi2:
                thi = tlo - 1
            n = thi - tlo + 1
            if n <= 0:
                dead += 1
            else:
                i = c0 + tlo * e0 + I * (c1 + tlo * e1) + IJ * (c2 + tlo * e2) + off
                hits += n
                for _ in range(n):
                    arr[i] += val
                    i += st
            c0 += b2[0]
            c1 += b2[1]
            c2 += b2[2]
    if stats is not None:
        stats["planes"] = stats.get("planes", 0) + planes
        stats["lines"] = stats.get("lines", 0) + lines
        stats["hits"] = stats.get("hits", 0) + hits
        stats["dead"] = stats.get("dead", 0) + dead
        stats["u"] = b3
        stats["d"] = (D[0], D[1])
    return hits

