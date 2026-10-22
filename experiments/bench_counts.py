import random

from common import Box, METHODS, ORDER, random_ideals, save


def run():
    box = Box(256, 256, 256)
    arr = bytearray(box.size)
    rng = random.Random(61)
    out = {}
    for b in (12, 18):
        ideals = random_ideals(rng, 1 << b, 1 << (b + 1), 100)
        res = {}
        for name in ORDER:
            ids = ideals if name not in ("HAKT", "LS") else ideals[:20]
            st = {}
            for r, B, C in ids:
                METHODS[name](r, B, C, box, arr, 0, st)
            res[name] = {k: v / len(ids) for k, v in st.items() if isinstance(v, (int, float))}
            print(b, name, res[name], flush=True)
        out[b] = res
    save("counts.json", out)


if __name__ == "__main__":
    run()
