import json
import os
import random
import sys
import time
import tracemalloc

import numpy as np
from sympy import isprime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from latsieve.core import Box, truth_counts
from latsieve.methods import METHODS, ORDER, applicable
from latsieve.transitions import Recorder

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def random_ideals(rng, lo, hi, n):
    out = []
    while len(out) < n:
        r = rng.randrange(lo, hi)
        if isprime(r):
            out.append((r, rng.randrange(1, r), rng.randrange(1, r)))
    return out


def time_batch(fn, ideals, box, arr, repeat=1):
    best = None
    for _ in range(repeat):
        t = time.perf_counter()
        for r, B, C in ideals:
            fn(r, B, C, box, arr, 0)
        dt = time.perf_counter() - t
        best = dt if best is None or dt < best else best
    return best


def peak_memory(fn, ideals, box, arr):
    peaks = []
    for r, B, C in ideals:
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn(r, B, C, box, arr, 0)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    return float(np.mean(peaks)), float(np.max(peaks))


def recall(fn, ideal, box):
    r, B, C = ideal
    T = np.nonzero(truth_counts(r, B, C, box))[0]
    rec = Recorder()
    fn(r, B, C, box, rec, 0)
    found = np.array(rec.idx, dtype=np.int64)
    u = np.unique(found)
    good = np.intersect1d(u, T, assume_unique=True).size
    return int(T.size), int(good), int(found.size - good)


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)
