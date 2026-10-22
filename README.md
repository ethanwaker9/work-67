# Exhaustive Three Dimensional Lattice Sieving via Lattice Width

This repository contains our implementation and the experiments of the research *Exhaustive Three Dimensional Lattice Sieving via Lattice Width*. The relation collection of the number field sieve for discrete logarithms in finite fields
`F_{p^n}` of medium characteristic enumerates, for every prime ideal of norm `r` of the factor
base, the points of the sieving lattice

    Λ(r, B, C) = { (x, y, z) in Z^3 : x + B y + C z ≡ 0 (mod r) }

inside the search box `H = [-I/2, I/2) × [-J/2, J/2) × [0, K)`. The package implements the
lattice width sieve (LWS) of the paper, which enumerates `Λ ∩ H` exactly with `O(N)` operations
after one three dimensional lattice reduction, and eight sieves from the literature, all in one
framework with the same box conventions, the same sieve array and the same lattice reduction.

## Contents

```
latsieve/
  core.py          box, ground truth, Minkowski reduction in dimension three (greedy algorithm), dual bases
  lws.py           lattice width sieve, for sieving lattices and for any lattice given by a basis
  line_sieve.py    line sieve (LS)
  plane_sieve.py   plane sieve of Gaudry, Grémy and Videau (PS)
  space_sieve.py   space sieve of Gaudry, Grémy and Videau (SS)
  hakt.py          three dimensional lattice sieve of Hayasaka, Aoki, Kobayashi and Takagi (HAKT)
  mr.py            sieve of McGuire and Robinson (MR)
  ntv.py           globalntv and localntv of Grémy (NTV-G, NTV-L)
  enum_dgp.py      lattice enumeration in the circumscribed ellipsoid, as used by De Micheli, Gaudry and Pierrot (ENUM)
  fk.py            basis and successor rule of Franke and Kleinjung
  transitions.py   transition vectors, corner rectangle covering, extremal family
  nfs.py           factor base and special-q lattices of a 240-bit field F_{p^6}
  methods.py       registry of the nine sieves
tests/
  test_lws.py      exactness of LWS against a brute force count on random lattices
  test_methods.py  points found, points missed and time of the sieves on random lattices
experiments/
  bench_random.py      time, memory and points found per ideal on random lattices
  bench_nfs.py         all degree one ideals of two special-q of a 240-bit field F_{p^6} 
  bench_tvlists.py     transition vectors missed by the candidate lists of SS, HAKT and NTV
  bench_counts.py      mean numbers of planes, lines, nodes and fallback steps per ideal 
  bench_transitions.py number of transition vectors of random lattices 
  bench_bounds.py      planes, lines and empty lines of LWS against the bounds 
  bench_scaling.py     time per ideal against the box height K 
  bench_setup.py       time of the reduction and of the loops of LWS 
  results/             JSON files written by the benchmarks
```

## Usage
It requires Python 3.10 or newer with NumPy, SymPy and Matplotlib; `epstopdf` (from a TeX distribution) for the
conversion of the figures to PDF.
```python
from latsieve.core import Box
from latsieve import lws

box = Box(256, 256, 256)
arr = bytearray(box.size)
stats = {}
lws.sieve(1048583, 123457, 654321, box, arr, 1, stats)
print(stats["planes"], stats["lines"], stats["hits"])
```
`arr[i]` is increased by the given value at the index `i = x + I*y + I*J*z + box.off` of every
point `(x, y, z)` of `Λ ∩ H`. Every sieve of `latsieve.methods.METHODS` has the same signature.

## Running the experiments
From the root of the repository:
```
PYTHONPATH=. python3 tests/test_lws.py
PYTHONPATH=. python3 tests/test_methods.py
cd experiments
python3 bench_random.py cube
python3 bench_random.py flat
python3 bench_nfs.py
python3 bench_tvlists.py
python3 bench_counts.py
python3 bench_transitions.py
python3 bench_bounds.py
python3 bench_scaling.py
python3 bench_setup.py
python3 make_figures.py
python3 make_tikz.py
python3 make_tables.py
```
The benchmarks write their results to `experiments/results`, and the figure scripts read them.
Timings depend on the machine; the ratios between the sieves are stable.

