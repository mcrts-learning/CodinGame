import sys
import math

def transform(s):
    return s.replace('|--|', '→←').replace(' ', '')

def next(grid, pos):
    r, c = pos
    v = grid[r][c]
    if v == '→':
        next_pos = (r + 1, c + 1)
    elif v == '←':
        next_pos = (r + 1, c - 1)
    else:
        next_pos = (r + 1, c)
    return next_pos

def run(grid, pos):
    for _ in range(len(grid) - 1):
        pos = next(grid, pos)
    res = grid[pos[0]][pos[1]]
    return res

w, h = [int(i) for i in input().split()]

grid = []
for i in range(h):
    grid.append(transform(input()))

for c, char in enumerate(grid[0]):
    value = run(grid, (0, c))
    print(str(char) + str(value))
