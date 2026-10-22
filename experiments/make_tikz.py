import math
import os
import sys

import numpy as np
from sympy import nextprime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from latsieve.core import Box
from latsieve import lws
from latsieve.transitions import extremal_family, phi_grid, box_points

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

COLORS = ["blue!55", "orange!80", "green!55!black!70", "red!60", "violet!60", "cyan!60!black!60"]


def transition_figure():
    I, J = 4, 6
    r = nextprime((I + 1) * (I - 1) + I * (J - 1) + I)
    B, C = extremal_family(I, J, r)
    x, y, z = phi_grid(r, B, C, I, J)
    o = np.argsort(z)
    xs, ys, zs = x[o], y[o], z[o]
    T = {}
    for k in range(len(zs)):
        k2 = (k + 1) % len(zs)
        T[(int(xs[k]), int(ys[k]))] = (int(xs[k2] - xs[k]), int(ys[k2] - ys[k]), int((zs[k2] - zs[k]) % r))
    vecs = sorted(set(T.values()), key=lambda v: (v[2], v[0]))
    s = 0.62
    lines = ["\\begin{tikzpicture}[x=%.2fcm,y=%.2fcm,every node/.style={font=\\rmfamily\\scriptsize,text=black}]" % (s, s)]
    for (a, b), v in T.items():
        c = COLORS[vecs.index(v)]
        lines.append("\\fill[%s] (%d,%d) rectangle (%d,%d);" % (c, a, b, a + 1, b + 1))
        lines.append("\\node at (%.1f,%.1f) {%d};" % (a + 0.5, b + 0.5, vecs.index(v) + 1))
    lines.append("\\draw[black!70] (0,0) grid (%d,%d);" % (I, J))
    lines.append("\\draw[very thick,black] (0,0) rectangle (%d,%d);" % (I - 1, J - 1))
    for a in range(I):
        lines.append("\\node[below] at (%.1f,0) {%d};" % (a + 0.5, a))
    for b in range(J):
        lines.append("\\node[left] at (0,%.1f) {%d};" % (b + 0.5, b))
    lines.append("\\node at (%.1f,-0.95) {$q_1$};" % (I / 2))
    lines.append("\\node[rotate=90] at (-1.0,%.1f) {$q_2$};" % (J / 2))
    for k, v in enumerate(vecs):
        yy = J - 0.6 - k * 0.98
        lines.append("\\fill[%s] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (COLORS[k], I + 0.5, yy - 0.3, I + 1.1, yy + 0.3))
        lines.append("\\draw[black!70] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (I + 0.5, yy - 0.3, I + 1.1, yy + 0.3))
        lines.append("\\node at (%.2f,%.2f) {%d};" % (I + 0.8, yy, k + 1))
        lines.append("\\node[anchor=west] at (%.2f,%.2f) {$(%d,%d,%d)$};" % (I + 1.15, yy, v[0], v[1], v[2]))
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines), (I, J, r, B, C, vecs)


def zonogon_figure():
    r, B, C = 13099, 2753, 6346
    box = Box(64, 64, 64)
    D, b = lws.setup(r, B, C, box)
    smin, smax, low, high = lws.chains(D[0], D[1], box)

    def path(ch):
        pts = [(c[0], c[1]) for c in ch]
        last = ch[-1]
        pts.append((last[0] + last[2], last[1] + last[3]))
        return pts

    lowp = path(low)
    highp = path(high)
    poly = lowp + list(reversed(highp))
    poly = [(a / r, m / r) for a, m in poly]
    pts = box_points(r, B, C, box)
    occ = {}
    for p in pts:
        k1 = (D[0][0] * p[0] + D[0][1] * p[1] + D[0][2] * p[2]) // r
        k2 = (D[1][0] * p[0] + D[1][1] * p[1] + D[1][2] * p[2]) // r
        occ[(k1, k2)] = occ.get((k1, k2), 0) + 1

    def evalc(ch, s, lowside):
        best = None
        for c in ch:
            if c[0] <= s <= c[0] + c[2]:
                v = (c[1] * c[2] + c[3] * (s - c[0])) / c[2]
                best = v if best is None else (min(best, v) if lowside else max(best, v))
        return best

    lines_k = []
    for k1 in range(math.ceil(smin / r), smax // r + 1):
        a = evalc(low, k1 * r, True)
        e = evalc(high, k1 * r, False)
        for k2 in range(math.ceil(a / r), math.floor(e / r) + 1):
            lines_k.append((k1, k2, occ.get((k1, k2), 0)))
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    W, Hh = 6.4, 4.4
    sx = W / (x1 - x0)
    sy = Hh / (y1 - y0)

    def T(px, py):
        return ((px - x0) * sx, (py - y0) * sy)

    L = ["\\begin{tikzpicture}[every node/.style={font=\\rmfamily\\scriptsize,text=black}]"]
    L.append("\\fill[blue!8] " + " -- ".join("(%.3f,%.3f)" % T(*p) for p in poly) + " -- cycle;")
    L.append("\\draw[blue!70!black,thick] " + " -- ".join("(%.3f,%.3f)" % T(*p) for p in poly) + " -- cycle;")
    for k1 in range(math.ceil(x0), math.floor(x1) + 1):
        a, _ = T(k1, y0)
        L.append("\\draw[densely dashed,black!45] (%.3f,%.3f) -- (%.3f,%.3f);" % (a, -0.1, a, Hh + 0.1))
        L.append("\\node[below] at (%.3f,-0.1) {$%d$};" % (a, k1))
    for k1, k2, n in lines_k:
        a, c = T(k1, k2)
        if n > 0:
            L.append("\\fill[black] (%.3f,%.3f) circle (0.06);" % (a, c))
            L.append("\\node[anchor=west] at (%.3f,%.3f) {$%d$};" % (a + 0.04, c, n))
        else:
            L.append("\\filldraw[fill=white,draw=red!75!black,thick] (%.3f,%.3f) circle (0.07);" % (a, c))
    ky0 = math.ceil(y0)
    ky1 = math.floor(y1)
    step = max(1, (ky1 - ky0) // 4)
    for k2 in range(ky0, ky1 + 1, step):
        _, c = T(x0, k2)
        L.append("\\draw[black!45] (-0.08,%.3f) -- (0,%.3f);" % (c, c))
        L.append("\\node[left] at (-0.08,%.3f) {$%d$};" % (c, k2))
    L.append("\\node at (%.2f,-0.75) {$k_1=\\langle D_1,x\\rangle/r$};" % (W / 2))
    L.append("\\node[rotate=90] at (-0.85,%.2f) {$\\langle D_2,x\\rangle/r$};" % (Hh / 2))
    L.append("\\end{tikzpicture}")
    info = {"r": r, "B": B, "C": C, "D": D, "b": b, "planes": len(set(k[0] for k in lines_k)), "lines": len(lines_k), "dead": sum(1 for k in lines_k if k[2] == 0), "hits": len(pts)}
    return "\n".join(L), info


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a, ia = transition_figure()
    with open(os.path.join(OUT, "tikz_tv.tex"), "w") as f:
        f.write(a + "\n")
    z, iz = zonogon_figure()
    with open(os.path.join(OUT, "tikz_zono.tex"), "w") as f:
        f.write(z + "\n")
    print(ia)
    print(iz)
