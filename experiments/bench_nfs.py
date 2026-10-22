import os
import time
import tracemalloc

from common import Box, METHODS, ORDER, applicable, save, load, RESULTS
from latsieve import nfs


def factor_base():
    path = os.path.join(RESULTS, "fb_f0.json")
    if os.path.exists(path):
        return [tuple(x) for x in load("fb_f0.json")]
    fb = nfs.factor_base(nfs.F0, 2, 1 << 20)
    save("fb_f0.json", fb)
    return fb


def run():
    fb = factor_base()
    box = Box(256, 256, 128)
    sqs = nfs.special_qs(nfs.F0, 1 << 21, 2)
    out = {"box": [256, 256, 128], "fb": len(fb), "specialq": []}
    cap = 600.0
    for q, rho in sqs:
        M = nfs.special_q_basis(q, rho)
        ideals, skipped = nfs.ideal_lattices(M, fb, q)
        lev = {0: [], 1: [], 2: []}
        for r, B, C, t in ideals:
            L = 0 if r < box.I else (1 if r < box.I * box.J else 2)
            lev[L].append((r, B, C))
        entry = {"q": q, "rho": rho, "M": M, "skipped": skipped, "counts": {k: len(v) for k, v in lev.items()}, "methods": {}}
        out["specialq"].append(entry)
        for name in ORDER:
            fn = METHODS[name]
            res = {}
            for L in (0, 1, 2):
                ids = [x for x in lev[L] if applicable(name, x[0], box)]
                if not ids:
                    continue
                arr = bytearray(box.size)
                probe = ids[:: max(1, len(ids) // 200)][:200]
                t = time.perf_counter()
                hp = 0
                for r, B, C in probe:
                    hp += fn(r, B, C, box, arr, 0)
                tp = time.perf_counter() - t
                est = tp / len(probe) * len(ids)
                if est > cap:
                    sample = ids[:: max(1, len(ids) // 2000)][:2000]
                    t = time.perf_counter()
                    hs = 0
                    for r, B, C in sample:
                        hs += fn(r, B, C, box, arr, 0)
                    ts = time.perf_counter() - t
                    res[L] = {"time": ts / len(sample) * len(ids), "hits": hs / len(sample) * len(ids), "n": len(ids), "sampled": len(sample)}
                else:
                    t = time.perf_counter()
                    h = 0
                    for r, B, C in ids:
                        h += fn(r, B, C, box, arr, 0)
                    res[L] = {"time": time.perf_counter() - t, "hits": h, "n": len(ids), "sampled": 0}
                print(q, name, L, res[L], flush=True)
            mem = []
            for L in (0, 1, 2):
                ids = [x for x in lev[L] if applicable(name, x[0], box)][:: 997][:12]
                for r, B, C in ids:
                    tracemalloc.start()
                    tracemalloc.reset_peak()
                    fn(r, B, C, box, arr, 0)
                    mem.append(tracemalloc.get_traced_memory()[1])
                    tracemalloc.stop()
            res["mem"] = max(mem) if mem else None
            entry["methods"][name] = res
            save("nfs.json", out)
    save("nfs.json", out)


if __name__ == "__main__":
    run()
