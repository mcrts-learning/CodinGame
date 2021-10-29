import sys
import math
import re
import numpy as np


def make_transformation(shift_array):
    n = len(shift_array)
    permutations = np.array([(x + i) % n for x, i in zip(shift_array, range(n))])
    idx = np.empty_like(permutations)
    idx[permutations] = np.arange(n)
    matrix = np.identity(n)[:, idx]
    return matrix

def make_permutation(shift_array):
    n = len(shift_array)
    permutations = np.array([(x + i) % n for x, i in zip(shift_array, range(n))])
    idx = np.empty_like(permutations)
    idx[permutations] = np.arange(n)
    return idx

F, B, R, L, U, D = (0, 1, 2, 3, 4, 5)
F, B, R, L, U, D = tuple("FBRLUD")
FRONT_SHIFT = np.array([19, 19, 0, 0, 0, 0, 17, 17, 17, 0, 0, 19, 19, 19, 0, 0, 0, 0, 17, 17, 17, 0, 0, 19])

FRONT = make_permutation(FRONT_SHIFT)
#FRONT = make_transformation(FRONT_SHIFT)

cube = np.array([F, R, R, B, B, L, L, F, U, U, D, D, F, L, L, B, B, R, R, F, D, D, U, U])
new_cube = cube[FRONT]
print(cube, file=sys.stderr)
print(new_cube, file=sys.stderr)

print("", file=sys.stderr)
print("Front Face", file=sys.stderr)
face = [7, 0, 12, 19]
print(cube[face].reshape((2, 2)), file=sys.stderr)
print(new_cube[face].reshape((2, 2)), file=sys.stderr)
print("Left Face", file=sys.stderr)
face = [5, 6, 14, 13]
print(cube[face].reshape((2, 2)), file=sys.stderr)
print(new_cube[face].reshape((2, 2)), file=sys.stderr)
