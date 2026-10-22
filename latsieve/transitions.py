import numpy as np
from .lws import sieve as lws_sieve
from .core import Box


def phi_grid(r, B, C, I, J):
    Ci = pow(C, -1, r)
    x = np.repeat(np.arange(I, dtype=np.int64), J)
    y = np.tile(np.arange(J, dtype=np.int64), I)
    z = (-(x + B * y) * Ci) % r
    return x, y, z


def transition_vectors(r, B, C, I, J):
    x, y, z = phi_grid(r, B, C, I, J)
    if len(np.unique(z)) < I * J:
        return None
    o = np.argsort(z, kind="stable")
    xs, ys, zs = x[o], y[o], z[o]
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    dz = (np.roll(zs, -1) - zs) % r
    if len(zs) == 1:
        dz = np.array([r])
    keys = list(zip(dx.tolist(), dy.tolist(), dz.tolist()))
    cnt = {}
    for k in keys:
        cnt[k] = cnt.get(k, 0) + 1
    return cnt


class Recorder:
    __slots__ = ("idx",)

    def __init__(self):
        self.idx = []

    def __getitem__(self, i):
        return 0

    def __setitem__(self, i, v):
        self.idx.append(i)


def box_points(r, B, C, box):
    rec = Recorder()
    lws_sieve(r, B, C, box, rec, 0)
    out = []
    for i in rec.idx:
        zz, rem = divmod(i, box.IJ)
        yy, xx = divmod(rem, box.I)
        out.append((xx + box.x0, yy + box.y0, zz + box.z0))
    return out


def slab_vectors(r, B, C, I, J, Z):
    box = Box(2 * I - 1, 2 * J - 1, Z, x0=-(I - 1), y0=-(J - 1), z0=1)
    return box_points(r, B, C, box)


def coverage_transitions(vectors, I, J):
    vecs = sorted(vectors, key=lambda v: v[2])
    covered = np.zeros((I, J), dtype=bool)
    chosen = []
    k = 0
    while k < len(vecs):
        z = vecs[k][2]
        group = []
        while k < len(vecs) and vecs[k][2] == z:
            group.append(vecs[k])
            k += 1
        new = covered.copy()
        for v in group:
            x0, x1 = max(0, -v[0]), min(I, I - v[0])
            y0, y1 = max(0, -v[1]), min(J, J - v[1])
            if x0 >= x1 or y0 >= y1:
                continue
            if not covered[x0:x1, y0:y1].all():
                chosen.append(v)
            new[x0:x1, y0:y1] = True
        covered = new
        if covered.all():
            return chosen, True
    return chosen, False


def extremal_family(I, J, r):
    inv = pow(I + 1, -1, r)
    B = (-I * inv) % r
    C = (-inv) % r
    return B, C
