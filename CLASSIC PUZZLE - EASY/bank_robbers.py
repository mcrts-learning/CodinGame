import sys
import math
import heapq as hq
C = 5
N = 10

r = int(input())
v = int(input())
robbers = [0] * r
hq.heapify(robbers)
for i in range(v):
    c, n = [int(j) for j in input().split()]
    c = c - n
    time = (C**c) * (N**n)
    r = hq.heappop(robbers)
    hq.heappush(robbers, r+time)

print(max(robbers))
