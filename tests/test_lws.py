import random, time, sys
import numpy as np
from sympy import isprime
from latsieve.core import Box, truth_counts
from latsieve import lws

def rand_ideal(rng, lo, hi):
    while True:
        r = rng.randrange(lo, hi)
        if isprime(r):
            return r, rng.randrange(1, r), rng.randrange(1, r)

rng = random.Random(1)
bad = 0
for box in [Box(16, 16, 16), Box(32, 16, 8), Box(64, 64, 32), Box(8, 32, 64)]:
    for trial in range(200):
        lo = 3
        hi = 4 * box.size
        r, B, C = rand_ideal(rng, lo, hi)
        T = truth_counts(r, B, C, box)
        arr = bytearray(box.size)
        lws.sieve(r, B, C, box, arr)
        A = np.frombuffer(bytes(arr), dtype=np.uint8)
        if not np.array_equal(A, T):
            bad += 1
            print("MISMATCH", box.I, box.J, box.K, r, B, C, int(A.sum()), int(T.sum()))
            if bad > 5:
                sys.exit(1)
print("bad", bad)
