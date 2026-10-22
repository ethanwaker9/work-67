import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from latsieve.core import Box
from latsieve import lws

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tables")

NAMES = ["LS", "PS", "SS", "HAKT", "MR", "NTV-G", "NTV-L", "ENUM", "LWS"]
SRC = {"LS": "\\citep{Zajac2008}", "PS": "\\citep{GaudryGremyVideau2016}", "SS": "\\citep{GaudryGremyVideau2016}", "HAKT": "\\citep{HAKT2015}", "MR": "\\citep{McGuireRobinson2021}", "NTV-G": "\\citep{Gremy2019}", "NTV-L": "\\citep{Gremy2019}", "ENUM": "\\citep{DeMicheliGaudryPierrot2021}", "LWS": "(ours)"}


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def lev(b, I, IJ):
    if 2 ** (b + 1) <= I:
        return 0
    if 2 ** b >= IJ:
        return 2
    if 2 ** b >= I and 2 ** (b + 1) <= IJ:
        return 1
    return None


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def random_summary(tag):
    d = load("random_%s.json" % tag)
    I, J, K = d["box"]
    out = {}
    for n in NAMES:
        e = {}
        for L in (0, 1, 2):
            rows = [r for r in d["bins"] if lev(r["b"], I, I * J) == L and n in r["methods"]]
            if not rows:
                continue
            t = gm([r["methods"][n]["us"] for r in rows])
            f = sum(r["methods"][n]["found"] for r in rows) / sum(r["methods"][n]["points"] for r in rows)
            tp = gm([r["methods"][n]["us"] / max(1e-9, r["methods"][n]["found"] / min(r["methods"][n]["n"], 6)) for r in rows])
            sp = [r["methods"][n]["us"] / r["methods"]["LWS"]["us"] for r in rows]
            e[L] = {"us": t, "found": 100 * f, "uspp": tp, "spmin": min(sp), "spmax": max(sp)}
        mem = np.mean([r["methods"][n]["mem"] for r in d["bins"] if n in r["methods"]]) / 1024
        e["mem"] = mem
        out[n] = e
    return out, (I, J, K)


def fmt(x, dig=0):
    if x is None:
        return "--"
    if dig == 0:
        return "{:,}".format(int(round(x))).replace(",", "\\,")
    return ("{:.%df}" % dig).format(x)


def table_random(tag):
    s, box = random_summary(tag)
    lines = []
    for n in NAMES:
        e = s[n]
        cells = [n if n != "LWS" else "\\textbf{LWS}"]
        for L in (0, 1, 2):
            if L in e:
                cells.append(fmt(e[L]["us"]))
                cells.append(fmt(e[L]["uspp"], 3) if e[L]["uspp"] < 10 else fmt(e[L]["uspp"], 1))
                cells.append(fmt(e[L]["found"], 1))
            else:
                cells += ["--", "--", "--"]
        cells.append(fmt(e["mem"], 2))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines), s, box


def nfs_summary():
    d = load("nfs.json")
    out = {}
    for n in NAMES:
        e = {0: [0.0, 0.0, 0, 0], 1: [0.0, 0.0, 0, 0], 2: [0.0, 0.0, 0, 0]}
        mem = []
        sampled = False
        for sq in d["specialq"]:
            m = sq["methods"].get(n)
            if m is None:
                continue
            for L in ("0", "1", "2"):
                if L in m:
                    e[int(L)][0] += m[L]["time"]
                    e[int(L)][1] += m[L]["hits"]
                    e[int(L)][2] += m[L]["n"]
                    if m[L]["sampled"]:
                        sampled = True
            if m.get("mem"):
                mem.append(m["mem"])
        out[n] = {"lev": e, "mem": max(mem) if mem else None, "sampled": sampled}
    ref = out["LWS"]["lev"]
    for n in NAMES:
        for L in (0, 1, 2):
            h = out[n]["lev"][L][1]
            out[n]["lev"][L].append(100 * h / ref[L][1] if ref[L][1] else None)
    return out, d


def table_nfs():
    s, d = nfs_summary()
    lines = []
    for n in NAMES:
        e = s[n]["lev"]
        cells = [n if n != "LWS" else "\\textbf{LWS}"]
        tot = 0.0
        hits = 0.0
        allh = 0.0
        for L in (0, 1, 2):
            if e[L][2] == 0:
                cells += ["--", "--"]
                continue
            tot += e[L][0]
            hits += e[L][1]
            allh += s["LWS"]["lev"][L][1]
            star = "$^\\ast$" if s[n]["sampled"] and L == 2 and n == "HAKT" else ""
            cells.append(fmt(e[L][0], 1) + star)
            cells.append(fmt(e[L][4], 2))
        cells.append(fmt(tot, 1))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines), s, d


def table_tv():
    t = load("tvlists.json")
    tr = load("transitions.json")
    lines = []
    for row in t:
        I = row["I"]
        rs = row["rows"]
        cnt = [x["ntv"] for x in rs]
        cells = [str(I), str(len(rs)), fmt(np.mean(cnt), 2), str(max(cnt)), fmt(100 * np.mean([x["ss_all"] for x in rs]), 1), fmt(100 * np.mean([x["ss_cov"] for x in rs]), 1), fmt(100 * np.mean([x["hakt_cov"] for x in rs]), 1), fmt(100 * np.mean([x["ntv_cov"] for x in rs]), 1), fmt(1e3 * np.mean([x["t_alg"] for x in rs]), 2), fmt(np.mean([x["slab"] for x in rs]), 1)]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


THEORY = {
    "LS": ("yes", "$\\Theta(JK+N)$", "$O(1)$"),
    "PS": ("yes", "$\\Omega(K+N)$", "$O(1)$"),
    "SS": ("no", "$O(N(|L_+|+|L_0|)+K)$", "$O(|L_+|+|L_0|)$"),
    "HAKT": ("no", "$O((N+K/w_3)\\eta^2)$", "$O(\\eta^2)$"),
    "MR": ("no", "$O(N+P+\\sum t+L)$", "$O(1)$"),
    "NTV-G": ("no", "$O(N|T|+fK|S|)$", "$O(|S|+|T|+f)$"),
    "NTV-L": ("no", "$O(N|T|+fK|S|)$", "$O(|S|+|T|+f)$"),
    "ENUM": ("yes", "$\\Theta(n_E)$", "$O(1)$"),
    "LWS": ("yes", "$O(N)$", "$O(1)$"),
}


def table_overview():
    s, box = random_summary("cube")
    nfs, d = nfs_summary()
    lines = []
    for n in NAMES:
        ex, tm, mm = THEORY[n]
        e = s[n]
        tot = sum(nfs[n]["lev"][L][0] for L in (0, 1, 2))
        f1 = fmt(e[1]["us"]) if 1 in e else "--"
        f2 = fmt(e[2]["us"]) if 2 in e else "--"
        p2 = fmt(e[2]["found"], 1) if 2 in e else "--"
        name = n + " " + SRC[n] if n != "LWS" else "\\textbf{LWS (ours)}"
        star = "$^\\ast$" if nfs[n]["sampled"] else ""
        lines.append(" & ".join([name, ex, tm, mm, f1, f2, p2, fmt(tot, 1) + star]) + " \\\\")
    return "\n".join(lines)


def table_counts():
    c = load("counts.json")
    keys = [("LS", "lines"), ("PS", "planes"), ("SS", "fallbacks"), ("HAKT", "wsteps"), ("MR", "planes"), ("MR", "lines"), ("MR", "ilp"), ("NTV-G", "fallbacks"), ("NTV-L", "fallbacks"), ("ENUM", "nodes"), ("ENUM", "leaves"), ("LWS", "planes"), ("LWS", "lines"), ("LWS", "dead")]
    labels = {"lines": "lines", "planes": "planes", "fallbacks": "fallback iterations", "wsteps": "steps along $w$", "ilp": "lines tested by the integer program", "nodes": "inner nodes", "leaves": "leaves", "dead": "lines without points"}
    lines = []
    for n, k in keys:
        cells = [n if n != "LWS" else "\\textbf{LWS}", labels[k]]
        for b in ("12", "18"):
            v = c[b].get(n, {}).get(k)
            cells.append(("--" if v is None else (fmt(v) if abs(v - round(v)) < 1e-9 else "{:,.1f}".format(v).replace(",", "\\,"))))
        lines.append(" & ".join(cells) + " \\\\")
    hits = []
    for b in ("12", "18"):
        hits.append("{:,.1f}".format(c[b]["LWS"]["hits"]).replace(",", "\\,"))
    lines.append("all & points $N$ & " + " & ".join(hits) + " \\\\")
    return "\n".join(lines)


def table_example(r=13099, B=2753, C=6346, n=64):
    box = Box(n, n, n)
    D, b = lws.setup(r, B, C, box)
    smin, smax, low, high = lws.chains(D[0], D[1], box)
    b1, b2, b3 = b
    lines = []
    for k1 in range(-((-smin) // r), smax // r + 1):
        s = k1 * r
        sec = []
        for ch in (low, high):
            j = 0
            while j + 1 < len(ch) and ch[j + 1][0] <= s:
                j += 1
            sa, ma, gs, gm = ch[j]
            sec.append((ma * gs + gm * (s - sa), gs * r))
        k2lo = -((-sec[0][0]) // sec[0][1])
        k2hi = sec[1][0] // sec[1][1]
        first = True
        for k2 in range(k2lo, k2hi + 1):
            c = [k1 * b1[i] + k2 * b2[i] for i in range(3)]
            tl, th = -(1 << 62), 1 << 62
            for i in range(3):
                e = b3[i]
                if e > 0:
                    tl = max(tl, -((c[i] - box.lo[i]) // e))
                    th = min(th, (box.hi[i] - c[i]) // e)
                elif e < 0:
                    tl = max(tl, -((c[i] - box.hi[i]) // e))
                    th = min(th, (box.lo[i] - c[i]) // e)
            pts = [tuple(c[i] + t * b3[i] for i in range(3)) for t in range(tl, th + 1)]
            cells = ["$%d$" % k1 if first else "", "$[%.2f,\\,%.2f]$" % (sec[0][0] / sec[0][1], sec[1][0] / sec[1][1]) if first else "", "$%d$" % k2, "$(%d,%d,%d)$" % tuple(c), "$%d,\\,%d$" % (tl, th) if pts else "empty", ", ".join("$(%d,%d,%d)$" % p for p in pts) if pts else "--"]
            lines.append(" & ".join(cells) + " \\\\")
            first = False
    return "\n".join(lines)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for tag in ("cube", "flat"):
        if os.path.exists(os.path.join(RES, "random_%s.json" % tag)):
            body, s, box = table_random(tag)
            open(os.path.join(OUT, "table_random_%s.tex" % tag), "w").write(body + "\n")
            json.dump({k: {str(kk): vv for kk, vv in v.items()} for k, v in s.items()}, open(os.path.join(OUT, "summary_%s.json" % tag), "w"), indent=1)
    if os.path.exists(os.path.join(RES, "nfs.json")):
        body, s, d = table_nfs()
        open(os.path.join(OUT, "table_nfs.tex"), "w").write(body + "\n")
        json.dump({k: {"lev": {str(L): v["lev"][L] for L in (0, 1, 2)}, "mem": v["mem"], "sampled": v["sampled"]} for k, v in s.items()}, open(os.path.join(OUT, "summary_nfs.json"), "w"), indent=1)
    if os.path.exists(os.path.join(RES, "nfs.json")) and os.path.exists(os.path.join(RES, "random_cube.json")):
        open(os.path.join(OUT, "table_overview.tex"), "w").write(table_overview() + "\n")
    if os.path.exists(os.path.join(RES, "counts.json")):
        open(os.path.join(OUT, "table_counts.tex"), "w").write(table_counts() + "\n")
    if os.path.exists(os.path.join(RES, "tvlists.json")):
        open(os.path.join(OUT, "table_tv.tex"), "w").write(table_tv() + "\n")
    open(os.path.join(OUT, "table_example.tex"), "w").write(table_example() + "\n")
    print("ok")
