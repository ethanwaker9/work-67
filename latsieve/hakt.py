def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mul(k, a):
    return (k * a[0], k * a[1], k * a[2])


def _neg(a):
    return (-a[0], -a[1], -a[2])


def _sign(x):
    return (x > 0) - (x < 0)


def _fk2(Ix, u, v):
    u = _neg(u)
    it = 0
    while abs(v[0]) >= Ix and it < 10000:
        it += 1
        a = (-u[0]) // v[0]
        u = _add(u, _mul(a, v))
        u, v = v, u
    if v[0] == 0:
        return u, v
    if v[0] > 0:
        lo = (-Ix - u[0]) // v[0] + 1
    else:
        lo = (Ix - u[0]) // v[0] + 1
    a = max(1, lo)
    while abs(u[0] + a * v[0]) >= Ix and a < lo + 4:
        a += 1
    u = _add(u, _mul(a, v))
    if u[0] < 0:
        u, v = v, u
    return u, v


def _reduce_by(x, w):
    if w[1] == 0:
        return x
    q = int(x[1] / w[1])
    return _sub(x, _mul(q, w))


def _reduce1(w, u, v, Ix, Iy):
    x, y = u, v
    if x[2] > y[2]:
        x, y = y, x
    elif x[2] == y[2] and x[1] > y[1]:
        x, y = y, x
    it = 0
    while it < 20000:
        it += 1
        while abs(w[0] + x[0]) < Ix and it < 20000:
            it += 1
            if abs(w[1]) < abs(u[1]) or abs(w[1]) < Iy:
                return u, v, w
            w = _add(w, x)
        if abs(w[1]) < abs(v[1]) or abs(w[1]) < Iy:
            return u, v, w
        w = _add(w, y)
    return u, v, w


def _reduce2(w, u, v, Ix, Iy):
    if _sign(u[1]) != _sign(w[1]):
        x, y = u, v
    else:
        x, y = v, u
    it = 0
    while abs(w[1]) >= Iy and it < 20000:
        it += 1
        w = _add(w, x)
        while abs(w[0]) >= Ix and it < 20000:
            it += 1
            w = _add(w, y)
        if abs(w[1]) < abs(u[1]) or abs(w[1]) < abs(v[1]):
            break
    return u, v, w


def _is_opposite(u, v, w):
    if u[0] == 0:
        return _sign(v[0]) != _sign(w[0])
    y = u[1] * v[0] - u[0] * v[1]
    z = u[1] * w[0] - u[0] * w[1]
    if u[0] < 0:
        y, z = -y, -z
    return _sign(y) != _sign(z)


def _radiate(u, v, w):
    if _is_opposite(u, v, w):
        if not _is_opposite(v, w, u):
            u = _neg(u)
    else:
        if _is_opposite(v, w, u):
            v = _neg(v)
        else:
            w = _neg(w)
    return u, v, w


def _adjust(u, v, w, Ix, Iy):
    it = 0
    changed = True
    while changed and it < 200:
        changed = False
        it += 1
        vecs = [u, v, w]
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                x, y = vecs[a], vecs[b]
                z = _sub(x, y)
                if abs(z[0]) < Ix and abs(z[1]) < Iy:
                    if abs(x[2]) >= abs(y[2]):
                        vecs[a] = z
                    else:
                        vecs[b] = z
                    u, v, w = _radiate(*vecs)
                    changed = True
                    break
            if changed:
                break
    return u, v, w


def basis(r, B, C, Ix, Iy):
    z1 = (-B) % r
    z2 = (-C) % r
    u = (r, 0, 0)
    v = (z1 - r, 1, 0)
    m = (z2 * pow(z1, -1, r)) % r
    w = (0, -m, 1)
    u, v = _fk2(Ix, u, v)
    if w[1] != 0 and abs(u[1]) >= abs(w[1]):
        u = _reduce_by(u, w)
    if w[1] != 0 and abs(v[1]) >= abs(w[1]):
        v = _reduce_by(v, w)
    it = 0
    while abs(w[1]) >= Iy and it < 200:
        it += 1
        if _sign(u[1]) == _sign(v[1]):
            u, v, w = _reduce1(w, u, v, Ix, Iy)
        else:
            u, v, w = _reduce2(w, u, v, Ix, Iy)
        u, v, w = _radiate(u, v, w)
        if abs(u[1]) > abs(v[1]):
            w, u = u, w
        else:
            w, v = v, w
    u, v, w = _adjust(u, v, w, Ix, Iy)
    out = []
    for x in (u, v, w):
        if x[2] < 0 or (x[2] == 0 and x[1] < 0):
            x = _neg(x)
        out.append(x)
    return out


def sieve(r, B, C, box, arr, val=1, stats=None, depth=3):
    I = box.I
    IJ = box.IJ
    off = box.off
    xs, ys, zs = box.lo
    xh, yh, zh = box.hi
    u, v, w = basis(r, B, C, box.I, box.J)
    if w[2] <= 0:
        cand = [x for x in (u, v, w) if x[2] > 0]
        if cand:
            m = min(cand, key=lambda x: x[2])
            if m is u:
                u, w = w, u
            elif m is v:
                v, w = w, v
    first = [x for x in (u, v, w) if x[2] >= 0 and x != (0, 0, 0)]
    second = []
    for i in range(0, depth + 1):
        for j in range(0, depth + 1):
            if i == 0 and j == 0:
                continue
            d = (i * u[0] + j * v[0], i * u[1] + j * v[1], i * u[2] + j * v[2])
            if d[2] < 0 or d == (0, 0, 0):
                continue
            second.append(d)
            if i > 0 and j > 0:
                first.append(d)
    first = sorted(set(first), key=lambda d: d[2])
    second = sorted(set(second), key=lambda d: d[2])
    px, py, pz = 0, 0, 0
    hits = wsteps = 0
    if xs <= 0 <= xh and ys <= 0 <= yh and zs <= 0 <= zh:
        arr[off] += val
        hits += 1
    wz = w[2]
    while True:
        nxt = None
        for d in first:
            qz = pz + d[2]
            if qz > zh:
                break
            qx = px + d[0]
            qy = py + d[1]
            if xs <= qx <= xh and ys <= qy <= yh:
                nxt = (qx, qy, qz)
                break
        if nxt is None:
            if wz <= 0:
                break
            qx, qy, qz = px, py, pz
            while nxt is None:
                qx += w[0]
                qy += w[1]
                qz += wz
                wsteps += 1
                if qz > zh:
                    break
                if xs <= qx <= xh and ys <= qy <= yh:
                    nxt = (qx, qy, qz)
                    break
                for d in second:
                    tz = qz + d[2]
                    if tz > zh:
                        break
                    tx = qx + d[0]
                    ty = qy + d[1]
                    if xs <= tx <= xh and ys <= ty <= yh:
                        nxt = (tx, ty, tz)
                        break
        if nxt is None:
            break
        px, py, pz = nxt
        arr[px + I * py + IJ * pz + off] += val
        hits += 1
    if stats is not None:
        stats["hits"] = stats.get("hits", 0) + hits
        stats["wsteps"] = stats.get("wsteps", 0) + wsteps
    return hits
