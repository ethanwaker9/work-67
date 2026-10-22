from .core import greedy3, gauss2


def _level(r, box):
    if r < box.I:
        return 0
    if r < box.I * box.J:
        return 1
    return 2


def _shape(r, box, lev):
    I, J = box.I, box.J
    if lev == 2:
        return (I, J, max(1, r // (I * J)))
    if lev == 1:
        return (I, max(1, r // I), 1)
    return (max(1, r), 1, 1)


def _weights(shape):
    a, b, c = shape
    return ((b * c) ** 2, (a * c) ** 2, (a * b) ** 2)


def _kidx(v):
    if v[2] != 0:
        return 2
    if v[1] != 0:
        return 1
    return 0


def _norm(v):
    k = _kidx(v)
    if v[k] < 0:
        return (-v[0], -v[1], -v[2])
    return v


class _CVA:
    __slots__ = ("r", "e0", "e1", "w")

    def __init__(self, r, lam0, box):
        self.r = r
        self.w = (box.J * box.J, box.I * box.I, 1)
        e0, e1, _, _ = gauss2((r, 0, 0), (lam0, 1, 0), self.w)
        self.e0, self.e1 = e0, e1

    def close(self, t, i):
        if i <= 0:
            q = t[0] // self.r
            return [(q * self.r, 0, 0), ((q + 1) * self.r, 0, 0)]
        e0, e1, w = self.e0, self.e1, self.w
        g11 = w[0] * e0[0] * e0[0] + w[1] * e0[1] * e0[1]
        g22 = w[0] * e1[0] * e1[0] + w[1] * e1[1] * e1[1]
        g12 = w[0] * e0[0] * e1[0] + w[1] * e0[1] * e1[1]
        h1 = w[0] * t[0] * e0[0] + w[1] * t[1] * e0[1]
        h2 = w[0] * t[0] * e1[0] + w[1] * t[1] * e1[1]
        det = g11 * g22 - g12 * g12
        a = (h1 * g22 - h2 * g12) // det
        b = (h2 * g11 - h1 * g12) // det
        out = []
        for da in (0, 1):
            for db in (0, 1):
                x, y = a + da, b + db
                out.append((x * e0[0] + y * e1[0], x * e0[1] + y * e1[1], 0))
        return out


def initialize(r, B, C, box, variant):
    lam0 = (-B) % r
    lam1 = (-C) % r
    b0, b1, b2 = (r, 0, 0), (lam0, 1, 0), (lam1, 0, 1)
    lev = _level(r, box)
    w = _weights(_shape(r, box, lev))
    cva = _CVA(r, lam0, box)
    vecs = set()
    if variant == "global" or lev == 2:
        red = greedy3([b0, b1, b2], w)
        for a in (-1, 0, 1):
            for c in (-1, 0, 1):
                for d in (-1, 0, 1):
                    if a or c or d:
                        v = tuple(a * red[0][i] + c * red[1][i] + d * red[2][i] for i in range(3))
                        vecs.add(_norm(v))
    else:
        u, v2, _, _ = gauss2(b0, b1, w)
        for a in (-1, 0, 1):
            for c in (-1, 0, 1):
                if a or c:
                    v = tuple(a * u[i] + c * v2[i] for i in range(3))
                    vecs.add(_norm(v))
        for e in cva.close(b2, 1):
            vecs.add(_norm((b2[0] - e[0], b2[1] - e[1], b2[2] - e[2])))
    lim = (box.I, box.J, box.K)
    S = [[], [], []]
    T = [[], [], []]
    for v in vecs:
        k = _kidx(v)
        S[k].append(v)
        if all(abs(v[i]) < lim[i] for i in range(k)):
            T[k].append(v)
    for k in range(3):
        S[k].sort(key=lambda v: v[k])
        T[k].sort(key=lambda v: v[k])
    return lev, T, S, cva


def sieve(r, B, C, box, arr, val=1, stats=None, variant="local"):
    lev, T, S, cva = initialize(r, B, C, box, variant)
    lo = box.lo
    hi = box.hi
    I = box.I
    IJ = box.IJ
    off = box.off
    cnt = [0, 0]

    def inside(p, upto):
        for i in range(upto + 1):
            if p[i] < lo[i] or p[i] > hi[i]:
                return False
        return True

    def fbadd(c, k, sgn):
        it = 0
        while lo[k] <= c[k] <= hi[k] and it < 4 * (hi[k] - lo[k] + 2):
            it += 1
            cnt[1] += 1
            L = []
            for v in S[k]:
                t = (c[0] + sgn * v[0], c[1] + sgn * v[1], c[2] + sgn * v[2])
                for e in cva.close(t, k - 1):
                    d = (t[0] - e[0], t[1] - e[1], t[2] - e[2])
                    if inside(d, 2):
                        return d
                    L.append(d)
            if not L:
                return None
            if sgn > 0:
                c = min(L, key=lambda p: p[k])
            else:
                c = max(L, key=lambda p: p[k])
        return None

    def step(c, k, sgn, start):
        for v in T[k]:
            d = (c[0] + sgn * v[0], c[1] + sgn * v[1], c[2] + sgn * v[2])
            if inside(d, k - 1):
                return d
        if k > lev or k == 2:
            e = fbadd(c, k, sgn)
            if e is not None:
                nv = (sgn * (e[0] - c[0]), sgn * (e[1] - c[1]), sgn * (e[2] - c[2]))
                if nv[k] > 0 and nv not in T[k] and all(abs(nv[i]) < (box.I, box.J, box.K)[i] for i in range(k)):
                    T[k].append(nv)
                    T[k].sort(key=lambda v: v[k])
                    S[k].append(nv)
                    S[k].sort(key=lambda v: v[k])
            return e
        return None

    def enum(c, k):
        cur = c
        while cur is not None and cur[k] <= hi[k]:
            if k == 0:
                arr[cur[0] + I * cur[1] + IJ * cur[2] + off] += val
                cnt[0] += 1
            else:
                enum(cur, k - 1)
            cur = step(cur, k, 1, c)
        cur = step(c, k, -1, c)
        while cur is not None and cur[k] >= lo[k]:
            if k == 0:
                arr[cur[0] + I * cur[1] + IJ * cur[2] + off] += val
                cnt[0] += 1
            else:
                enum(cur, k - 1)
            cur = step(cur, k, -1, c)

    if inside((0, 0, 0), 2):
        enum((0, 0, 0), 2)
    if stats is not None:
        stats["hits"] = stats.get("hits", 0) + cnt[0]
        stats["fallbacks"] = stats.get("fallbacks", 0) + cnt[1]
    return cnt[0]
