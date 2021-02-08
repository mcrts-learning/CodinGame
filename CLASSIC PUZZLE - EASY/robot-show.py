import sys
import math

size = int(input())
n = int(input())
bots = list(map(int, input().split()))
left = min(bots)
right = max(bots)
t = max([size - left, right])
print(t)
