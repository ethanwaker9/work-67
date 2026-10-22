import json
import math
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from latsieve.core import Box
from latsieve import lws
from latsieve.transitions import box_points

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures")

NAMES = ["LS", "PS", "SS", "HAKT", "MR", "NTV-G", "NTV-L", "ENUM", "LWS"]
STYLE = {
    "LS": ("tab:gray", "v", "-"),
    "PS": ("tab:blue", "s", "-"),
    "SS": ("tab:orange", "^", "-"),
    "HAKT": ("tab:brown", "x", "-"),
    "MR": ("tab:green", "D", "-"),
    "NTV-G": ("tab:purple", "<", "-"),
    "NTV-L": ("tab:pink", ">", "--"),
    "ENUM": ("tab:olive", "P", "-"),
    "LWS": ("black", "o", "-"),
}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif", "mathtext.fontset": "cm", "lines.linewidth": 1.0, "lines.markersize": 3.2})


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    eps = os.path.join(OUT, name + ".eps")
    fig.savefig(eps, format="eps", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    subprocess.run(["epstopdf", eps], check=True)


def plot_method(ax, xs, ys, name):
    c, m, ls = STYLE[name]
    lw = 1.6 if name == "LWS" else 0.9
    ax.plot(xs, ys, color=c, marker=m, linestyle=ls, linewidth=lw, label=name)


def levels(ax, I, IJ):
    for v in (math.log2(I), math.log2(IJ)):
        ax.axvline(v, color="0.6", linestyle=":", linewidth=0.8)


def bench_figure(tag, fname):
    d = load("random_%s.json" % tag)
    if d is None:
        return
    I, J, K = d["box"]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for name in NAMES:
        xs, t, tp, rec, mem = [], [], [], [], []
        for row in d["bins"]:
            m = row["methods"].get(name)
            if m is None:
                continue
            xs.append(row["b"] + 0.5)
            t.append(m["us"])
            frac = m["found"] / max(1, m["points"])
            nmean = m["points"] / min(m["n"], 6)
            tp.append(m["us"] / max(frac * nmean, 1e-9))
            rec.append(100 * frac)
            mem.append(m["mem"] / 1024)
        if not xs:
            continue
        plot_method(axs[0, 0], xs, t, name)
        plot_method(axs[0, 1], xs, tp, name)
        plot_method(axs[1, 0], xs, rec, name)
        plot_method(axs[1, 1], xs, mem, name)
    axs[0, 0].set_yscale("log")
    axs[0, 1].set_yscale("log")
    axs[1, 1].set_yscale("log")
    axs[0, 0].set_ylabel(r"time per ideal ($\mu$s)")
    axs[0, 1].set_ylabel(r"time per point found ($\mu$s)")
    axs[1, 0].set_ylabel(r"points found (%)")
    axs[1, 1].set_ylabel(r"auxiliary memory per ideal (KiB)")
    for ax, lab in zip(axs.flat, "abcd"):
        ax.set_xlabel(r"$\log_2 r$")
        levels(ax, I, I * J)
        ax.set_title("(%s)" % lab, loc="left")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axs[1, 0].set_ylim(-3, 105)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=9, bbox_to_anchor=(0.5, 1.03), frameon=False, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, fname)


def speedup_figure():
    ds = [(tag, load("random_%s.json" % tag)) for tag in ("cube", "flat")]
    if any(d is None for tag, d in ds):
        return
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for row, (tag, d) in enumerate(ds):
        I, J, K = d["box"]
        for name in NAMES[:-1]:
            xs, t, tp = [], [], []
            for b in d["bins"]:
                m = b["methods"].get(name)
                if m is None:
                    continue
                w = b["methods"]["LWS"]
                xs.append(b["b"] + 0.5)
                t.append(m["us"] / w["us"])
                pm = m["us"] / max(m["found"] / min(m["n"], 6), 1e-9)
                pw = w["us"] / max(w["found"] / min(w["n"], 6), 1e-9)
                tp.append(pm / pw)
            plot_method(axs[row, 0], xs, t, name)
            plot_method(axs[row, 1], xs, tp, name)
        for col in range(2):
            ax = axs[row, col]
            ax.axhline(1.0, color="black", linewidth=1.4)
            ax.set_yscale("log")
            ax.set_xlabel(r"$\log_2 r$")
            levels(ax, I, I * J)
            ax.grid(True, linewidth=0.3, alpha=0.5)
        axs[row, 0].set_ylabel(r"time / time of LWS, $%d\times%d\times%d$" % (I, J, K))
        axs[row, 1].set_ylabel(r"time per point / that of LWS")
    for ax, lab in zip(axs.flat, "abcd"):
        ax.set_title("(%s)" % lab, loc="left")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.03), frameon=False, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "speedup")


def setup_figure():
    R = load("setup.json")
    if R is None:
        return None
    xs = [r["b"] + 0.5 for r in R]
    tot = np.array([r["total"] for r in R])
    st = np.array([r["setup"] for r in R])
    lp = tot - st
    A = np.array([[r["N"], r["lines"], r["planes"]] for r in R], dtype=float)
    c, _, _, _ = np.linalg.lstsq(A / lp[:, None], np.ones(len(R)), rcond=None)
    fit = A @ c
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axs[0]
    ax.plot(xs, tot, "k-o", linewidth=1.6, label="LWS, total")
    ax.plot(xs, st, "--s", color="tab:blue", label="reduction and zonogon")
    ax.plot(xs, lp, "-^", color="tab:green", label="loops")
    ax.set_ylabel(r"time per ideal ($\mu$s)")
    ax.legend(loc="upper right")
    ax = axs[1]
    ax.plot(xs, lp, "^", color="tab:green", label="loops, measured")
    ax.plot(xs, fit, "-", color="black", linewidth=1.0, label=r"$%.3fN+%.2f\ell+%.2fP$" % tuple(c))
    ax.set_ylabel(r"time of the loops per ideal ($\mu$s)")
    ax.legend(loc="upper right")
    for a, lab in zip(axs, "ab"):
        a.set_yscale("log")
        a.set_xlabel(r"$\log_2 r$")
        levels(a, 256, 256 * 256)
        a.set_title("(%s)" % lab, loc="left")
        a.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    save(fig, "setup")
    return [round(float(x), 4) for x in c], float(np.max(np.abs(fit / lp - 1)))


def theory_figure():
    b = load("bounds.json")
    s = load("scaling.json")
    if b is None:
        return
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0))
    N = np.array([x["N"] for x in b], dtype=float)
    P = np.array([x["planes"] for x in b], dtype=float)
    Lm = np.array([x["lines"] for x in b], dtype=float)
    ax = axs[0, 0]
    ax.scatter(np.cbrt(N), P, s=4, color="black", label="LWS planes")
    xx = np.linspace(1, np.cbrt(N).max() * 1.05, 50)
    ax.plot(xx, 36 * math.sqrt(3) * xx + 1, "r--", label=r"$36\sqrt{3}N^{1/3}+1$")
    ax.axhline(256, color="tab:blue", linestyle=":", label=r"PS planes $K$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N^{1/3}$")
    ax.set_ylabel("planes")
    ax.legend(loc="lower right")
    ax = axs[0, 1]
    El = np.array([x["Elines"] for x in b])
    ok = np.array([x["valid"] for x in b])
    ax.scatter(N, Lm, s=4, color="black", label="LWS lines")
    ax.scatter(N[ok], El[ok], s=4, color="tab:green", marker="x", label="Proposition 5.7")
    xx = np.logspace(0, math.log10(N.max() * 1.2), 50)
    ax.plot(xx, (36 * math.sqrt(3) + 1) ** 2 * xx, "r--", label=r"$(36\sqrt{3}+1)^2N$")
    ax.plot(xx, xx, color="0.5", linestyle=":", label=r"$N$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("lines")
    ax.legend(loc="upper left")
    ax = axs[1, 0]
    bins = sorted(set(x["b"] for x in b))
    md, pdd, ml, pl = [], [], [], []
    for bb in bins:
        rows = [x for x in b if x["b"] == bb and x["valid"]]
        if not rows:
            md.append(np.nan); pdd.append(np.nan); ml.append(np.nan); pl.append(np.nan)
            continue
        md.append(sum(x["dead"] for x in rows) / len(rows))
        pdd.append(sum(x["Edead"] for x in rows) / len(rows))
        ml.append(sum(x["lines"] for x in rows) / len(rows))
        pl.append(sum(x["Elines"] for x in rows) / len(rows))
    xb = [bb + 0.5 for bb in bins]
    ax.plot(xb, ml, "k-o", label="lines, measured")
    ax.plot(xb, pl, "g--x", label=r"lines, $\mathbb{E}\,l_y$")
    ax.plot(xb, md, "-s", color="tab:red", label="empty lines, measured")
    ax.plot(xb, pdd, "--^", color="tab:orange", label=r"empty lines, $\mathbb{E}\,e_y$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_2 r$")
    ax.set_ylabel("lines per ideal")
    ax.legend(loc="upper right")
    ax = axs[1, 1]
    if s is not None:
        Ks = [row["K"] for row in s]
        for name in NAMES:
            ys = [row["methods"].get(name) for row in s]
            if any(v is None for v in ys):
                continue
            plot_method(ax, Ks, ys, name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"time per ideal ($\mu$s)")
        ax.legend(loc="upper left", ncol=3)
    for a, lab in zip(axs.flat, "abcd"):
        a.set_title("(%s)" % lab, loc="left")
        a.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    save(fig, "theory")


def transitions_figure():
    t = load("transitions.json")
    if t is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axs[0]
    Is = [x["I"] for x in t["random"]]
    data = [x["counts"] for x in t["random"]]
    pos = np.log2(Is)
    ax.boxplot(data, positions=pos, widths=0.45, showfliers=True, flierprops={"markersize": 2})
    ax.plot(pos, [max(d) for d in data], "k^", markersize=3, label="largest observed")
    fam = t["family"]
    ax.set_xticks(pos)
    ax.set_xticklabels([str(i) for i in Is])
    ax.set_xlabel(r"$I=J$")
    ax.set_ylabel(r"$|\mathcal{T}|$")
    ax.set_yscale("log")
    xx = np.array(Is, dtype=float)
    ax.plot(pos, xx + 2, "r--", label=r"$\min(I,J)+2$")
    ax.plot(pos, 3 * xx, "b:", label=r"$3\min(I,J)$")
    ax.legend(loc="upper left")
    ax.set_title("(a)", loc="left")
    ax = axs[1]
    for x in t["random"]:
        if x["I"] in (16, 64, 256):
            c = np.bincount(x["counts"])
            ks = np.arange(len(c))
            ax.plot(ks[c > 0], c[c > 0] / len(x["counts"]), "-o", markersize=2.5, label=r"$I=J=%d$" % x["I"])
    ax.plot([f["I"] for f in fam], [0.02] * len(fam), "|", color="white", alpha=0)
    ax.set_xlabel(r"$|\mathcal{T}|$")
    ax.set_ylabel("fraction of lattices")
    ax.legend(loc="upper right")
    ax.set_title("(b)", loc="left")
    for a in axs:
        a.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    save(fig, "transitions")


def lattice3d_figure():
    r, B, C = 13099, 2753, 6346
    box = Box(64, 64, 64)
    D, bb = lws.setup(r, B, C, box)
    pts = np.array(box_points(r, B, C, box))
    k1 = (pts @ np.array(D[0])) // r
    fig = plt.figure(figsize=(7.0, 3.3))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    lo, hi = box.lo, box.hi
    cs = box.corners()
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if sum(1 for t in range(3) if cs[i][t] != cs[j][t]) == 1:
                edges.append((cs[i], cs[j]))
    ax.add_collection3d(Line3DCollection(edges, colors="0.55", linewidths=0.6))
    cols = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    ks = sorted(set(k1.tolist()))
    for idx, k in enumerate(range(min(ks) - 1, max(ks) + 2)):
        poly = section(D[0], k * r, box)
        if len(poly) >= 3:
            P = np.array(poly + [poly[0]])
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=cols[idx % len(cols)], linewidth=0.9)
        m = k1 == k
        if m.any():
            ax.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=7, color=cols[idx % len(cols)], depthshade=False)
    ax.set_xlabel(r"$p_1$", labelpad=-4)
    ax.set_ylabel(r"$p_2$", labelpad=-4)
    ax.set_zlabel(r"$p_3$", labelpad=-4)
    ax.tick_params(pad=-2)
    ax.view_init(elev=18, azim=-58)
    ax.set_title("(a)", loc="left")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.add_collection3d(Line3DCollection(edges, colors="0.55", linewidths=0.6))
    zs = pts[:, 2]
    for z in range(lo[2], hi[2] + 1, 4):
        ax2.plot([lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]], [z] * 5, color="tab:blue", linewidth=0.25, alpha=1.0)
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=7, color="black", depthshade=False)
    ax2.set_xlabel(r"$p_1$", labelpad=-4)
    ax2.set_ylabel(r"$p_2$", labelpad=-4)
    ax2.set_zlabel(r"$p_3$", labelpad=-4)
    ax2.tick_params(pad=-2)
    ax2.view_init(elev=18, azim=-58)
    ax2.set_title("(b)", loc="left")
    fig.tight_layout()
    save(fig, "lattice3d")
    return len(pts), len(ks)


def section(d, s, box):
    cs = box.corners()
    out = []
    for i in range(8):
        for j in range(i + 1, 8):
            P, Q = cs[i], cs[j]
            if sum(1 for t in range(3) if P[t] != Q[t]) != 1:
                continue
            fp = sum(d[t] * P[t] for t in range(3)) - s
            fq = sum(d[t] * Q[t] for t in range(3)) - s
            if fp == fq:
                continue
            if (fp > 0 and fq > 0) or (fp < 0 and fq < 0):
                continue
            lam = fp / (fp - fq)
            out.append(tuple(P[t] + lam * (Q[t] - P[t]) for t in range(3)))
    if len(out) < 3:
        return out
    c = np.mean(np.array(out), axis=0)
    n = np.array(d, dtype=float)
    n /= np.linalg.norm(n)
    a = np.cross(n, [1.0, 0, 0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(n, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    bvec = np.cross(n, a)
    out.sort(key=lambda p: math.atan2(np.dot(np.array(p) - c, bvec), np.dot(np.array(p) - c, a)))
    return out


if __name__ == "__main__":
    bench_figure("cube", "benchmark")
    speedup_figure()
    theory_figure()
    transitions_figure()
    print(setup_figure())
    print(lattice3d_figure())
