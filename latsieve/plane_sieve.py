from .fk import fk_basis


def _gauss2d(u, v):
    nu = u[0] * u[0] + u[1] * u[1]
    nv = v[0] * v[0] + v[1] * v[1]
    if nu < nv:
        u, v, nu, nv = v, u, nv, nu
    while True:
        uv = u[0] * v[0] + u[1] * v[1]
        q = (2 * uv + nv) // (2 * nv)
        if q:
            u = (u[0] - q * v[0], u[1] - q * v[1])
            nu = u[0] * u[0] + u[1] * u[1]
        if nu >= nv:
            return v, u
        u, v, nu, nv = v, u, nv, nu


class PlaneMachine:
    __slots__ = ("r", "B", "C", "xs", "xe", "ys", "ye", "al", "be", "ga", "de", "svec", "bx", "by")

    def __init__(self, r, B, C, box):
        self.r, self.B, self.C = r, B, C
        self.xs, self.ys = box.x0, box.y0
        self.xe, self.ye = box.x0 + box.I, box.y0 + box.J
        (self.al, self.be), (self.ga, self.de) = fk_basis(r, (-B) % r, box.I)
        e0, e1 = _gauss2d((r, 0), ((-B) % r, 1))
        vx = (-C) % r
        if vx > r // 2:
            vx -= r
        det = e0[0] * e1[1] - e0[1] * e1[0]
        a = (vx * e1[1]) // det if det > 0 else (-(vx * e1[1])) // (-det)
        b = (-(e0[1] * vx)) // det if det > 0 else (e0[1] * vx) // (-det)
        cands = []
        for i in (0, 1):
            for j in (0, 1):
                cx = (a + i) * e0[0] + (b + j) * e1[0]
                cy = (a + i) * e0[1] + (b + j) * e1[1]
                cands.append(((vx - cx) ** 2 + cy * cy, (vx - cx, -cy)))
        cands.sort()
        self.svec = [c[1] for c in cands[:3]]
        if -self.al > self.ga:
            self.bx, self.by = self.al, self.be
        else:
            self.bx, self.by = self.ga, self.de

    def to_strip(self, x, y):
        bx, by = self.bx, self.by
        if bx < 0:
            bx, by = -bx, -by
        while x >= self.xe:
            x -= bx
            y -= by
        while x < self.xs:
            x += bx
            y += by
        return x, y

    def first_above(self, x, y):
        xs, xe, ys = self.xs, self.xe, self.ys
        al, be, ga, de = self.al, self.be, self.ga, self.de
        while y < ys:
            if x >= xs - al:
                x += al
                y += be
            elif x < xe - ga:
                x += ga
                y += de
            else:
                x += al + ga
                y += be + de
        while True:
            if x < xe + al:
                nx, ny = x - al, y - be
            elif x >= xs + ga:
                nx, ny = x - ga, y - de
            else:
                nx, ny = x - al - ga, y - be - de
            if ny >= ys:
                x, y = nx, ny
            else:
                return x, y

    def next_plane(self, x, y):
        xs, xe, ys, ye = self.xs, self.xe, self.ys, self.ye
        for sx, sy in self.svec:
            qx, qy = x + sx, y + sy
            if xs <= qx < xe and ys <= qy < ye:
                return self.first_above(qx, qy)
        sx, sy = self.svec[0]
        qx, qy = self.to_strip(x + sx, y + sy)
        return self.first_above(qx, qy)


def sieve(r, B, C, box, arr, val=1, stats=None):
    pm = PlaneMachine(r, B, C, box)
    I = box.I
    IJ = box.IJ
    off = box.off
    xs, xe, ye = pm.xs, pm.xe, pm.ye
    al, be, ga, de = pm.al, pm.be, pm.ga, pm.de
    su = al + I * be
    sv = ga + I * de
    suv = su + sv
    tu = xs - al
    tv = xe - ga
    z0 = box.z0
    if box.contains(0, 0, z0) and z0 == 0:
        px, py = pm.first_above(0, 0)
    else:
        x = box.x0 + ((-(B * box.y0 + C * z0) - box.x0) % r)
        px, py = pm.first_above(*pm.to_strip(x, box.y0))
    planes = hits = 0
    for z in range(z0, z0 + box.K):
        if z > z0:
            px, py = pm.next_plane(px, py)
        planes += 1
        x, y = px, py
        i = x + I * y + IJ * z + off
        while y < ye:
            arr[i] += val
            hits += 1
            if x >= tu:
                x += al
                y += be
                i += su
            elif x < tv:
                x += ga
                y += de
                i += sv
            else:
                x += al + ga
                y += be + de
                i += suv
    if stats is not None:
        stats["planes"] = stats.get("planes", 0) + planes
        stats["hits"] = stats.get("hits", 0) + hits
    return hits
