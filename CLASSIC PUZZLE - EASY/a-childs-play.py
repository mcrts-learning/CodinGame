import sys
import math

DIRMAP = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
}

NEXTDIR = {
    'UP': 'RIGHT',
    'RIGHT': 'DOWN',
    'DOWN': 'LEFT',
    'LEFT': 'UP',
}

SHORTCUTS = dict()
CYCLES = dict()

w, h = [int(i) for i in input().split()]
n = int(input())
grid = []
for y in range(h):
    row = input()
    if 'O' in row:
        x = row.index('O')
        pos = (x, y)
    grid.append(row.replace('O', '.'))
direction = 'UP'
start_step = None
end_step = None

while n > 0:
    step = (pos, direction)
    if step in SHORTCUTS and step not in CYCLES:
        c = 0
        s0 = step
        new_pos, new_direction, cost = SHORTCUTS[s0]
        si = new_pos, new_direction
        c += cost
        while si != s0:
            new_pos, new_direction, cost = SHORTCUTS[si]
            si = new_pos, new_direction
            c += cost
        CYCLES[s0] = c

    if step in CYCLES and CYCLES[step] <= n:
        cost = CYCLES[step]
        n = n % cost
    elif step in SHORTCUTS and SHORTCUTS[step][2] <= n:
        pos, direction, cost = SHORTCUTS[step]
        n -= cost
    else:
        n -= 1
        dx, dy = DIRMAP[direction]
        x, y = pos
        pos = x, y = (x + dx, y + dy)
        if start_step:
            end_step = (None, None, end_step[2] + 1)

        nx, ny = (x + dx, y + dy)
        if grid[ny][nx] == '#':
            while grid[ny][nx] != '.':
                direction = NEXTDIR[direction]
                dx, dy = DIRMAP[direction]
                nx, ny = (x + dx, y + dy)
                if end_step:
                    end_step = (pos, direction, end_step[2])

            if start_step and end_step:
                SHORTCUTS[start_step] = end_step
            start_step = (pos, direction)
            end_step = (None, None, 0)

print('{} {}'.format(pos[0], pos[1]))
