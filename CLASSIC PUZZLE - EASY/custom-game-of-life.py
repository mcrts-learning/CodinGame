import sys
import math
import numpy as np
import itertools as it
from collections import Counter
# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

def compute_counter(grid):
    counter = Counter()
    indices = np.ndindex(grid.shape)
    for dx, dy in it.product([-1, 0, 1], repeat=2):
        if (dx, dy) == (0, 0):
            continue
        for (x, y), v in np.ndenumerate(grid):
            counter[(x + dx, y + dy)] += v
    return counter

def iterate(grid, alive_rules, dead_rules):
    counter = compute_counter(grid)
    newgrid = np.zeros(grid.shape, dtype=int)
    for index, v in np.ndenumerate(grid):
        if v and counter[index] in alive_rules:
            newgrid[index] = 1
        if (not v) and counter[index] in dead_rules:
            newgrid[index] = 1
    return newgrid

h, w, n = [int(i) for i in input().split()]
grid = np.full((h, w), 0, dtype=int)

alive = list(it.compress(range(9), map(lambda s: bool(int(s)), input())))
dead = list(it.compress(range(9), map(lambda s: bool(int(s)), input())))

for i in range(h):
    line = list(map(lambda c: c == "O", input()))
    grid[i] = line
    
for _ in range(n):
    grid = iterate(grid, alive, dead)

grid = grid.astype(str)
grid = np.where(grid=="0", ".", grid) 
grid = np.where(grid=="1", "O", grid) 

print("\n".join(["".join(l) for l in grid]))
