def fk_basis(r, rho, I):
    a0, b0, a1, b1 = -r, 0, rho % r, 1
    while a1 >= I:
        k = (-a0) // a1
        a0 += k * a1
        b0 += k * b1
        if a0 > -I:
            break
        k = a1 // (-a0)
        a1 += k * a0
        b1 += k * b0
    if a1 < I and a0 <= -I:
        k = (-a0 - I) // a1 + 1
        a0 += k * a1
        b0 += k * b1
    elif a0 > -I and a1 >= I:
        k = (a1 - I) // (-a0) + 1
        a1 += k * a0
        b1 += k * b0
    return (a0, b0), (a1, b1)


def fk_next(x, y, xs, xe, u, v):
    if x >= xs - u[0]:
        return x + u[0], y + u[1]
    if x < xe - v[0]:
        return x + v[0], y + v[1]
    return x + u[0] + v[0], y + u[1] + v[1]


def fk_prev(x, y, xs, xe, u, v):
    if x < xe + u[0]:
        return x - u[0], y - u[1]
    if x >= xs + v[0]:
        return x - v[0], y - v[1]
    return x - u[0] - v[0], y - u[1] - v[1]
