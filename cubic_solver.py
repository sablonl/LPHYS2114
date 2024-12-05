from cmath import *
import numpy as np


"""
    Code from https://stackoverflow.com/questions/1829330/solving-a-cubic-equation
"""

def linear(a, b):
    solutions = set()
    if a == 0 and b == 0:
        solutions.add(True)

    if a == 0 and b != 0:
        solutions.add(False)

    if a != 0:
        solutions.add(-b / a)
    return solutions


def quadratic(a, b, c):
    solutions = set()
    if a != 0:
        D = b ** 2 - 4 * a * c
        x1 = (-b + sqrt(D)) / (2 * a)
        x2 = (-b - sqrt(D)) / (2 * a)
        solutions.update({x1, x2})
    else:
        solutions.update(linear(b, c))
    return solutions


def cubic(a, b, c, d):
    solutions = set()
    if a != 0:
        p = (3 * a * c - b ** 2) / (3 * a ** 2)
        q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
        for n in range(3):
            solutions.add((2 * sqrt(-p / 3) * cos(acos((-3 * q) * sqrt(-3 * p) / (2 * p ** 2)) / 3 + 2 * pi * n / 3))
                          - (b / (3 * a)))
    else:
        solutions.update(quadratic(b, c, d))

    sols = list(solutions)

    res = np.zeros((3))
    for i in range(3):
        if np.abs(sols[i].imag) < 1e-5:
            res[i] = sols[i].real
        else:
            res[i] = np.nan

    return np.sort(res)



