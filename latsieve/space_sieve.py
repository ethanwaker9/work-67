from .core import greedy3
from .plane_sieve import PlaneMachine


def candidates(r, B, C, box, keep=6):
    I, J = box.I, box.J
    Z = max(1, r // (I * J))
    w = ((J * Z) ** 2, (I * Z) ** 2, (I * J) ** 2)
    b = greedy3([(r, 0, 0), ((-B) % r, 1, 0), ((-C) % r, 0, 1)], w)
    seen = set()
    zero = []
    pos = []
    for a in (-1, 0, 1):
        for c in (-1, 0, 1):
            for d in (-1, 0, 1):
                if a == 0 and c == 0 and d == 0:
                    continue
                v = (a * b[0][0] + c * b[1][0] + d * b[2][0], a * b[0][1] + c * b[1][1] + d * b[2][1], a * b[0][2] + c * b[1][2] + d * b[2][2])
                if v[2] < 0 or (v[2] == 0 and (v[1] < 0 or (v[1] == 0 and v[0] < 0))):
                    v = (-v[0], -v[1], -v[2])
                if v in seen or abs(v[0]) >= I or abs(v[1]) >= J:
                    continue
                seen.add(v)
                if v[2] == 0:
                    zero.append(v)
                else:
                    pos.append(v)
    zero.sort(key=lambda v: (v[0] / I) ** 2 + (v[1] / J) ** 2)
    kept = []
    for v in zero:
        if all(v[0] * u[1] - v[1] * u[0] != 0 for u in kept):
            kept.append(v)
    pos.sort(key=lambda v: v[2])
    return kept, pos[:keep]


def sieve(r, B, C, box, arr, val=1, stats=None):
    I = box.I
    J = box.J
    IJ = box.IJ
    off = box.off
    xs, ys, zs = box.lo
    xh, yh, zh = box.hi
    lz0, lz = candidates(r, B, C, box)
    pm = None
    sx, sy, sz = 0, 0, 0
    hits = fallbacks = 0
    while sz <= zh:
        lc = [(sx, sy, sz)]
        for v in lz0:
            for sgn in (1, -1):
                qx, qy = sx + sgn * v[0], sy + sgn * v[1]
                while xs <= qx <= xh and ys <= qy <= yh:
                    lc.append((qx, qy, sz))
                    qx += sgn * v[0]
                    qy += sgn * v[1]
        for p in lc:
            arr[p[0] + I * p[1] + IJ * p[2] + off] += val
        hits += len(lc)
        best = None
        bz = zh + 1
        for p in lc:
            for v in lz:
                nz = p[2] + v[2]
                if nz >= bz:
                    break
                nx = p[0] + v[0]
                ny = p[1] + v[1]
                if xs <= nx <= xh and ys <= ny <= yh:
                    best = (nx, ny, nz)
                    bz = nz
                    break
        if best is None:
            fallbacks += 1
            if pm is None:
                pm = PlaneMachine(r, B, C, box)
            px, py = pm.first_above(*pm.to_strip(lc[0][0], lc[0][1]))
            z = lc[0][2]
            found = False
            while z < zh:
                z += 1
                px, py = pm.next_plane(px, py)
                if py <= yh:
                    found = True
                    break
            if not found:
                break
            best = (px, py, z)
            d = (px - sx, py - sy, z - sz)
            if abs(d[0]) < I and abs(d[1]) < J:
                k = 0
                while k < len(lz) and lz[k][2] <= d[2]:
                    k += 1
                if d not in lz:
                    lz.insert(k, d)
        sx, sy, sz = best
    if stats is not None:
        stats["hits"] = stats.get("hits", 0) + hits
        stats["fallbacks"] = stats.get("fallbacks", 0) + fallbacks
    return hits
