import random

from common import Box, METHODS, ORDER, applicable, random_ideals, time_batch, save


def run():
    rng = random.Random(41)
    I = J = 256
    rows = []
    for K in (64, 128, 256, 512, 1024):
        box = Box(I, J, K)
        arr = bytearray(box.size)
        vol = I * J * K
        ideals = random_ideals(rng, vol // 96, vol // 48, 24)
        row = {"K": K, "methods": {}}
        for name in ORDER:
            ids = [x for x in ideals if applicable(name, x[0], box)]
            if name in ("HAKT", "LS"):
                ids = ids[:6]
            dt = time_batch(METHODS[name], ids, box, arr, repeat=2)
            row["methods"][name] = dt / len(ids) * 1e6
            print(K, name, round(dt / len(ids) * 1e6), flush=True)
        rows.append(row)
        save("scaling.json", rows)


if __name__ == "__main__":
    run()
