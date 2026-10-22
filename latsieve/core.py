import numpy as np


class Box:
    __slots__ = ("I", "J", "K", "x0", "y0", "z0", "lo", "hi", "L", "size", "off", "IJ")

    def __init__(self, I, J, K, x0=None, y0=None, z0=0):
        self.I, self.J, self.K = I, J, K
        self.x0 = -(I // 2) if x0 is None else x0
        self.y0 = -(J // 2) if y0 is None else y0
        self.z0 = z0
        self.lo = (self.x0, self.y0, self.z0)
        self.hi = (self.x0 + I - 1, self.y0 + J - 1, self.z0 + K - 1)
        self.L = (I - 1, J - 1, K - 1)
        self.size = I * J * K
        self.IJ = I * J
        self.off = -(self.x0 + I * self.y0 + self.IJ * self.z0)

    def index(self, x, y, z):
        return x + self.I * y + self.IJ * z + self.off

    def stride(self, v):
        return v[0] + self.I * v[1] + self.IJ * v[2]

    def contains(self, x, y, z):
        return (self.x0 <= x <= self.hi[0]) and (self.y0 <= y <= self.hi[1]) and (self.z0 <= z <= self.hi[2])

    def corners(self):
        return [(a, b, c) for a in (self.lo[0], self.hi[0]) for b in (self.lo[1], self.hi[1]) for c in (self.lo[2], self.hi[2])]


def truth_counts(r, B, C, box):
    y = np.arange(box.y0, box.y0 + box.J, dtype=np.int64)[:, None]
    z = np.arange(box.z0, box.z0 + box.K, dtype=np.int64)[None, :]
    res = (-(B * y + C * z)) % r
    first = box.x0 + ((res - box.x0) % r)
    cnt = np.zeros((box.K, box.J, box.I), dtype=np.uint8)
    m = 0
    while True:
        xs = first + m * r
        mask = xs <= box.hi[0]
        if not mask.any():
            break
        yy, zz = np.nonzero(mask)
        cnt[zz, yy, xs[yy, zz] - box.x0] += 1
        m += 1
    return cnt.reshape(-1)


def gauss2(u, v, w):
    w0, w1, w2 = w
    u0, u1, u2 = u
    v0, v1, v2 = v
    nu = w0 * u0 * u0 + w1 * u1 * u1 + w2 * u2 * u2
    nv = w0 * v0 * v0 + w1 * v1 * v1 + w2 * v2 * v2
    if nu < nv:
        u0, u1, u2, v0, v1, v2, nu, nv = v0, v1, v2, u0, u1, u2, nv, nu
    while True:
        uv = w0 * u0 * v0 + w1 * u1 * v1 + w2 * u2 * v2
        q = (2 * uv + nv) // (2 * nv)
        if q:
            u0 -= q * v0
            u1 -= q * v1
            u2 -= q * v2
            nu = w0 * u0 * u0 + w1 * u1 * u1 + w2 * u2 * u2
        if nu >= nv:
            return (v0, v1, v2), (u0, u1, u2), nv, nu
        u0, u1, u2, v0, v1, v2, nu, nv = v0, v1, v2, u0, u1, u2, nv, nu


def greedy3(basis, w):
    w0, w1, w2 = w
    (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = basis
    na = w0 * a0 * a0 + w1 * a1 * a1 + w2 * a2 * a2
    nb = w0 * b0 * b0 + w1 * b1 * b1 + w2 * b2 * b2
    nc = w0 * c0 * c0 + w1 * c1 * c1 + w2 * c2 * c2
    if na > nb:
        a0, a1, a2, na, b0, b1, b2, nb = b0, b1, b2, nb, a0, a1, a2, na
    if nb > nc:
        b0, b1, b2, nb, c0, c1, c2, nc = c0, c1, c2, nc, b0, b1, b2, nb
        if na > nb:
            a0, a1, a2, na, b0, b1, b2, nb = b0, b1, b2, nb, a0, a1, a2, na
    while True:
        while True:
            ab = w0 * a0 * b0 + w1 * a1 * b1 + w2 * a2 * b2
            q = (2 * ab + na) // (2 * na)
            if q:
                b0 -= q * a0
                b1 -= q * a1
                b2 -= q * a2
                nb = w0 * b0 * b0 + w1 * b1 * b1 + w2 * b2 * b2
            if nb >= na:
                break
            a0, a1, a2, na, b0, b1, b2, nb = b0, b1, b2, nb, a0, a1, a2, na
        ab = w0 * a0 * b0 + w1 * a1 * b1 + w2 * a2 * b2
        h1 = w0 * c0 * a0 + w1 * c1 * a1 + w2 * c2 * a2
        h2 = w0 * c0 * b0 + w1 * c1 * b1 + w2 * c2 * b2
        det = na * nb - ab * ab
        n1 = h1 * nb - h2 * ab
        n2 = h2 * na - h1 * ab
        f1 = (2 * n1 + det) // (2 * det)
        f2 = (2 * n2 + det) // (2 * det)
        bn = -1
        for s in (f1 - 1, f1, f1 + 1):
            x0 = c0 - s * a0
            x1 = c1 - s * a1
            x2 = c2 - s * a2
            for t in (f2 - 1, f2, f2 + 1):
                y0 = x0 - t * b0
                y1 = x1 - t * b1
                y2 = x2 - t * b2
                n = w0 * y0 * y0 + w1 * y1 * y1 + w2 * y2 * y2
                if bn < 0 or n < bn:
                    bn = n
                    d0, d1, d2 = y0, y1, y2
        c0, c1, c2, nc = d0, d1, d2, bn
        if nc >= nb:
            return [(a0, a1, a2), (b0, b1, b2), (c0, c1, c2)]
        if nc < na:
            a0, a1, a2, na, b0, b1, b2, nb, c0, c1, c2, nc = c0, c1, c2, nc, a0, a1, a2, na, b0, b1, b2, nb
        else:
            b0, b1, b2, nb, c0, c1, c2, nc = c0, c1, c2, nc, b0, b1, b2, nb


def dual_basis(D, r):
    (a, b, c), (d, e, f), (g, h, i) = D
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    s = det // r
    cols = [
        ((e * i - f * h) // s, (f * g - d * i) // s, (d * h - e * g) // s),
        ((c * h - b * i) // s, (a * i - c * g) // s, (b * g - a * h) // s),
        ((b * f - c * e) // s, (c * d - a * f) // s, (a * e - b * d) // s),
    ]
    return cols
