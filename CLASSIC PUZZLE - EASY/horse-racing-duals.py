import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

n = int(input())
pi=[]
for i in range(n):
    pi.append(int(input()))
pi.sort()
P=pi
for i in range(n-1):
   P[i]=abs(P[i]-P[i+1])

# Write an action using print
# To debug: print("Debug messages...", file=sys.stderr)

print(min(P))
