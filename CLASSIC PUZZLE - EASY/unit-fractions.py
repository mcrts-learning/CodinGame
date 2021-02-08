import sys
import math

n = int(input())
for a in range(n+1, n*2+1):
    b = (n*a) / (a-n)
    if b.is_integer():
        print(f"1/{n} = 1/{b:0.0f} + 1/{a}")
