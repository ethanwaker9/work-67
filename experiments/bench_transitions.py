import random

import numpy as np
from sympy import isprime, nextprime

from common import save
from latsieve.transitions import transition_vectors, extremal_family


def run():
    rng = random.Random(31)
    out = {"random": [], "family": []}
    for I in (8, 16, 32, 64, 128, 256):
        J = I
        counts = []
        lengths = []
        trials = 400 if I <= 64 else (150 if I == 128 else 60)
        while len(counts) < trials:
            r = rng.randrange(4 * I * J, 64 * I * J)
            if not isprime(r):
                continue
            B, C = rng.randrange(1, r), rng.randrange(1, r)
            T = transition_vectors(r, B, C, I, J)
            if T is None:
                continue
            counts.append(len(T))
            lengths.append(len(set(k[2] for k in T)))
        out["random"].append({"I": I, "counts": counts, "lengths": lengths})
        print(I, np.mean(counts), max(counts), flush=True)
    for I in range(2, 33):
        J = 2 * I
        r = nextprime((I + 1) * (I - 1) + I * (J - 1) + I + 1)
        B, C = extremal_family(I, J, r)
        T = transition_vectors(r, B, C, I, J)
        out["family"].append({"I": I, "J": J, "r": r, "count": len(T)})
    save("transitions.json", out)


if __name__ == "__main__":
    run()
