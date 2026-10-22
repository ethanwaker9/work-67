from functools import partial
from . import line_sieve, plane_sieve, space_sieve, hakt, mr, ntv, enum_dgp, lws

METHODS = {
    "LS": line_sieve.sieve,
    "PS": plane_sieve.sieve,
    "SS": space_sieve.sieve,
    "HAKT": hakt.sieve,
    "MR": mr.sieve,
    "NTV-G": partial(ntv.sieve, variant="global"),
    "NTV-L": partial(ntv.sieve, variant="local"),
    "ENUM": enum_dgp.sieve,
    "LWS": lws.sieve,
}

ORDER = ["LS", "PS", "SS", "HAKT", "MR", "NTV-G", "NTV-L", "ENUM", "LWS"]


def applicable(name, r, box):
    if name in ("PS", "SS", "HAKT"):
        return r > box.I
    return True
