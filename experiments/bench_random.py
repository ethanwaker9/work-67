import random
import sys

from common import Box, METHODS, ORDER, applicable, random_ideals, time_batch, peak_memory, recall, save


def run(I, J, K, bmin, bmax, tag, seed):
    box = Box(I, J, K)
    arr = bytearray(box.size)
    rng = random.Random(seed)
    out = {"box": [I, J, K], "bins": []}
    for b in range(bmin, bmax + 1):
        n = 8 if b < 6 else (16 if b < 10 else (24 if b < 14 else 48))
        ideals = random_ideals(rng, 1 << b, 1 << (b + 1), n)
        row = {"b": b, "n": n, "methods": {}}
        nrec = min(n, 6)
        for name in ORDER:
            fn = METHODS[name]
            ids = [x for x in ideals if applicable(name, x[0], box)]
            if not ids:
                continue
            dt = time_batch(fn, ids, box, arr, repeat=1 if b < 6 else 2)
            mem_mean, mem_max = peak_memory(fn, ids[:4], box, arr)
            tot = good = extra = 0
            for x in ids[:nrec]:
                t, g, e = recall(fn, x, box)
                tot += t
                good += g
                extra += e
            row["methods"][name] = {"us": dt / len(ids) * 1e6, "mem": mem_mean, "memmax": mem_max, "points": tot, "found": good, "extra": extra, "n": len(ids)}
            print(tag, b, name, round(dt / len(ids) * 1e6), round(mem_mean), good, tot, flush=True)
        out["bins"].append(row)
        save("random_%s.json" % tag, out)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "cube":
        run(256, 256, 256, 2, 21, "cube", 11)
    elif which == "flat":
        run(512, 512, 128, 2, 22, "flat", 12)
