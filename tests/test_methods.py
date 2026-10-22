import random
import sys
import time

import numpy as np
from sympy import isprime

from latsieve.core import Box, truth_counts
from latsieve.methods import METHODS, ORDER, applicable


def rand_ideal(rng, lo, hi):
    while True:
        r = rng.randrange(lo, hi)
        if isprime(r):
            return r, rng.randrange(1, r), rng.randrange(1, r)


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ORDER
    rng = random.Random(7)
    for box in [Box(32, 32, 32), Box(64, 32, 16), Box(32, 64, 64)]:
        for lvl, (lo, hi) in [(1, (box.I + 1, box.I * box.J)), (2, (box.I * box.J + 1, box.size // 4))]:
            stat = {n: [0, 0, 0, 0.0, 0] for n in names}
            for _ in range(60):
                r, B, C = rand_ideal(rng, lo, hi)
                T = truth_counts(r, B, C, box)
                for n in names:
                    if not applicable(n, r, box):
                        continue
                    arr = bytearray(box.size)
                    t = time.perf_counter()
                    METHODS[n](r, B, C, box, arr)
                    stat[n][3] += time.perf_counter() - t
                    stat[n][4] += 1
                    A = np.frombuffer(bytes(arr), dtype=np.uint8)
                    stat[n][0] += int(T.sum())
                    stat[n][1] += int(np.maximum(T.astype(int) - A, 0).sum())
                    stat[n][2] += int(np.maximum(A.astype(int) - T, 0).sum())
            for n in names:
                s = stat[n]
                if s[4]:
                    print(box.I, box.J, box.K, "level", lvl, n, "points", s[0], "missed", s[1], "extra", s[2], "us per ideal", round(s[3] / s[4] * 1e6))


if __name__ == "__main__":
    main()
