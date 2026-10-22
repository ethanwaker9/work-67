from .core import greedy3


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _line_range(c, e, lo, hi):
    tlo = -(1 << 62)
    thi = 1 << 62
    for k in range(3):
        ek = e[k]
        ck = c[k]
        if ek > 0:
            v = -((ck - lo[k]) // ek)
            if v > tlo:
                tlo = v
            v = (hi[k] - ck) // ek
            if v < thi:
                thi = v
        elif ek < 0:
            v = -((ck - hi[k]) // ek)
            if v > tlo:
                tlo = v
            v = (lo[k] - ck) // ek
            if v < thi:
                thi = v
        elif ck < lo[k] or ck > hi[k]:
            return 1, 0
    return tlo, thi


def _ilp_point(base, v1, v2, n, box):
    lo, hi = box.lo, box.hi
    m1 = _cross(n, v1)
    den = _dot(m1, v2)
    if den < 0:
        m1 = (-m1[0], -m1[1], -m1[2])
        den = -den
    tgt = _dot(n, base)
    cs = box.corners()
    smin = smax = None
    for a in range(8):
        for b in range(a + 1, 8):
            P, Q = cs[a], cs[b]
            if (P[0] != Q[0]) + (P[1] != Q[1]) + (P[2] != Q[2]) != 1:
                continue
            fp = _dot(n, P) - tgt
            fq = _dot(n, Q) - tgt
            if fp == fq:
                if fp != 0:
                    continue
                gs = [_dot(m1, (P[0] - base[0], P[1] - base[1], P[2] - base[2])), _dot(m1, (Q[0] - base[0], Q[1] - base[1], Q[2] - base[2]))]
                for g in gs:
                    lo_v = -((-g) // den)
                    hi_v = g // den
                    smin = lo_v if smin is None or lo_v < smin else smin
                    smax = hi_v if smax is None or hi_v > smax else smax
                continue
            if (fp > 0 and fq > 0) or (fp < 0 and fq < 0):
                continue
            gp = _dot(m1, (P[0] - base[0], P[1] - base[1], P[2] - base[2]))
            gq = _dot(m1, (Q[0] - base[0], Q[1] - base[1], Q[2] - base[2]))
            num = gp * (fq - fp) - (gq - gp) * fp
            d = (fq - fp) * den
            if d < 0:
                num, d = -num, -d
            lo_v = -((-num) // d)
            hi_v = num // d
            smin = lo_v if smin is None or lo_v < smin else smin
            smax = hi_v if smax is None or hi_v > smax else smax
    tries = 0
    if smin is None:
        return None, tries
    for s in range(smin, smax + 1):
        tries += 1
        c = (base[0] + s * v2[0], base[1] + s * v2[1], base[2] + s * v2[2])
        tlo, thi = _line_range(c, v1, lo, hi)
        if tlo <= thi:
            return (c[0] + tlo * v1[0], c[1] + tlo * v1[1], c[2] + tlo * v1[2]), tries
    return None, tries


def sieve(r, B, C, box, arr, val=1, stats=None):
    I = box.I
    J = box.J
    K = box.K
    IJ = box.IJ
    off = box.off
    lo, hi = box.lo, box.hi
    w = ((J * K) ** 2, (I * K) ** 2, (I * J) ** 2)
    v1, v2, v3 = greedy3([(r, 0, 0), ((-B) % r, 1, 0), ((-C) % r, 0, 1)], w)
    n = _cross(v1, v2)
    h = _dot(n, v3)
    if h < 0:
        v3 = (-v3[0], -v3[1], -v3[2])
        h = -h
    vals = [_dot(n, c) for c in box.corners()]
    cmin = -((-min(vals)) // h)
    cmax = max(vals) // h
    st1 = v1[0] + I * v1[1] + IJ * v1[2]
    planes = lines = hits = ilp = 0
    for c in range(cmin, cmax + 1):
        planes += 1
        base = (c * v3[0], c * v3[1], c * v3[2])
        p0, tries = _ilp_point(base, v1, v2, n, box)
        ilp += tries
        if p0 is None:
            continue
        for sgn in (1, -1):
            j = 0 if sgn == 1 else -1
            while True:
                q = (p0[0] + j * v2[0], p0[1] + j * v2[1], p0[2] + j * v2[2])
                lines += 1
                tlo, thi = _line_range(q, v1, lo, hi)
                if tlo > thi:
                    break
                i = q[0] + tlo * v1[0] + I * (q[1] + tlo * v1[1]) + IJ * (q[2] + tlo * v1[2]) + off
                for _ in range(thi - tlo + 1):
                    arr[i] += val
                    i += st1
                hits += thi - tlo + 1
                j += sgn
    if stats is not None:
        stats["planes"] = stats.get("planes", 0) + planes
        stats["lines"] = stats.get("lines", 0) + lines
        stats["hits"] = stats.get("hits", 0) + hits
        stats["ilp"] = stats.get("ilp", 0) + ilp
    return hits
