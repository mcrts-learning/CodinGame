import sys
from collections import namedtuple

def log(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)

Vector2D = namedtuple('Vector2D', ['x', 'y'])

UP = Vector2D(0, -1)
DOWN = Vector2D(0, 1)
LEFT = Vector2D(-1, 0)
RIGHT = Vector2D(1, 0)
DIRECTIONS = {
    ('>', 'R'): [DOWN, RIGHT, UP, LEFT],
    ('>', 'L'): [UP, RIGHT, DOWN, LEFT],
    ('v', 'R'): [LEFT, DOWN, RIGHT, UP],
    ('v', 'L'): [RIGHT, DOWN, LEFT, UP],
    ('<', 'R'): [UP, LEFT, DOWN, RIGHT],
    ('<', 'L'): [DOWN, LEFT, UP, RIGHT],
    ('^', 'R'): [RIGHT, UP, LEFT, DOWN],
    ('^', 'L'): [LEFT, UP, RIGHT, DOWN],
}
DIRETIONS_MAP = {
    UP: '^',
    DOWN: 'v',
    LEFT: '<',
    RIGHT: '>'
}

width, height = [int(i) for i in input().split()]

def get_next_cells(current, direction, side, maze):
    h, w = len(maze), len(maze[0])
    vectors = DIRECTIONS[(direction, side)]
    x, y = current.x, current.y
    positions = [Vector2D(x + v.x, y + v.y) for v in vectors]
    positions = filter(lambda v: (0 <= v.x < w) and (0 <= v.y < h), positions)
    positions = filter(lambda v: maze[v.y][v.x] != '#', positions)
    nextpos = next(positions)
    vector = Vector2D(nextpos.x - x, nextpos.y - y)
    return nextpos, DIRETIONS_MAP[vector]

def get_start(maze):
    for y, row in enumerate(maze):
        for x, c in enumerate(row):
            if c in ('>', 'v', '<', '^'):
                return Vector2D(x, y), c

def increment(pos, maze, value):
    line = list(maze[pos.y])
    cell = line[pos.x]
    if cell in ('>', 'v', '<', '^'):
        cell = 0 + value
    else:
        cell = int(cell) + value
    line[pos.x] = str(cell)
    maze[pos.y] = ''.join(line)
    return maze

maze = []
for i in range(height):
    line = input()
    maze.append(line)
side = input()

start, direction = get_start(maze)
try:
    pos, direction = get_next_cells(start, direction, side, maze)
except StopIteration:
    increment(start, maze, 0)
else:
    increment(pos, maze, 1)
    while pos != start:
        pos, direction = get_next_cells(pos, direction, side, maze)
        increment(pos, maze, 1)

for row in maze:
    print(row)
