import math
import random

from common import Box, random_ideals, save
from latsieve import lws


def run():
    I = J = K = 256
    box = Box(I, J, K)
    arr = bytearray(box.size)
    rng = random.Random(21)
    rows = []
    for b in range(4, 22):
        n = 6 if b < 8 else (20 if b < 12 else 60)
        for r, B, C in random_ideals(rng, 1 << b, 1 << (b + 1), n):
            st = {}
            lws.sieve(r, B, C, box, arr, 0, st)
            u = st["u"]
            d1, d2 = st["d"]
            w1 = (abs(d1[0]) * (I - 1) + abs(d1[1]) * (J - 1) + abs(d1[2]) * (K - 1)) / r
            w2 = (abs(d2[0]) * (I - 1) + abs(d2[1]) * (J - 1) + abs(d2[2]) * (K - 1)) / r
            ub = (abs(u[0]) / I, abs(u[1]) / J, abs(u[2]) / K)
            nexp = I * J * K / r
            e1 = ub[0] + ub[1] + ub[2]
            e2 = ub[0] * ub[1] + ub[0] * ub[2] + ub[1] * ub[2]
            e3 = ub[0] * ub[1] * ub[2]
            rows.append({"b": b, "r": r, "N": st["hits"], "planes": st["planes"], "lines": st["lines"], "dead": st["dead"], "w1": w1, "w2": w2, "Nexp": nexp, "Elines": nexp * e1, "Edead": nexp * (e2 - e3), "valid": max(ub) <= 1})
        print(b, flush=True)
    save("bounds.json", rows)


if __name__ == "__main__":
    run()
