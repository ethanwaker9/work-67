import random
import time

from common import Box, random_ideals, save
from latsieve import lws


def best(f, reps=3):
    b = None
    for _ in range(reps):
        t = time.perf_counter()
        f()
        dt = time.perf_counter() - t
        b = dt if b is None or dt < b else b
    return b


def run():
    box = Box(256, 256, 256)
    arr = bytearray(box.size)
    rng = random.Random(71)
    rows = []
    for b in range(8, 22):
        ideals = random_ideals(rng, 1 << b, 1 << (b + 1), 48)

        def setup_all():
            for r, B, C in ideals:
                D, bb = lws.setup(r, B, C, box)
                lws.chains(D[0], D[1], box)

        def sieve_all():
            for r, B, C in ideals:
                lws.sieve(r, B, C, box, arr, 0)

        ts = best(setup_all) / len(ideals) * 1e6
        tt = best(sieve_all) / len(ideals) * 1e6
        st = {}
        for r, B, C in ideals:
            lws.sieve(r, B, C, box, arr, 0, st)
        n = len(ideals)
        row = {"b": b, "setup": ts, "total": tt, "N": st["hits"] / n, "planes": st["planes"] / n, "lines": st["lines"] / n}
        rows.append(row)
        print(b, round(ts, 1), round(tt, 1), round(row["N"], 1), round(row["lines"], 1), flush=True)
    save("setup.json", rows)


if __name__ == "__main__":
    run()
