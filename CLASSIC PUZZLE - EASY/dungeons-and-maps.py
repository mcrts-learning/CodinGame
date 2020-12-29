import sys
import math
import heapq as hq

def log(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)

W, H = [int(i) for i in input().split()]

def evaluate(m, r, c):
    p = 0
    positions = set([r, c])
    while True:
        char = m[r][c]
        if char == 'T': return p
        elif char == '>': c += 1
        elif char == '<': c -= 1
        elif char == '^': r -= 1
        elif char == 'v': r += 1
        else: return None
        if (r, c) in positions:
            return None
        else:
            positions.add((r, c))
        if not ((0 <= r <= H) and (0 <= c < W)):
            return None
        p += 1
    return p

r, c = [int(i) for i in input().split()]
n = int(input())
maps = []
for i in range(n):
    m = [input() for _ in range(H)]
    value = evaluate(m, r, c)
    if value is not None:
        hq.heappush(maps, (value, i))

if not maps:
    print("TRAP")
else:
    print(hq.heappop(maps)[1])
