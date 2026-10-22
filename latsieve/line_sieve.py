def sieve(r, B, C, box, arr, val=1, stats=None):
    I = box.I
    J = box.J
    x0, y0, z0 = box.lo
    Bm = B % r
    lines = hits = 0
    base = 0
    for z in range(z0, z0 + box.K):
        s = (-(B * y0 + C * z) - x0) % r
        for _ in range(J):
            lines += 1
            i = base + s
            e = base + I
            while i < e:
                arr[i] += val
                i += r
                hits += 1
            s -= Bm
            if s < 0:
                s += r
            base += I
    if stats is not None:
        stats["lines"] = stats.get("lines", 0) + lines
        stats["hits"] = stats.get("hits", 0) + hits
    return hits
