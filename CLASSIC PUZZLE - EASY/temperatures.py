import sys
import math

# Write an action using print
# To debug: print("Debug messages...", file=sys.stderr)

n = int(input())  # the number of temperatures to analyse
temps = input().split()  # the n temperatures expressed as integers ranging from -273 to 5526

T = []
for t in temps:
    T.append(int(t))

if n == 0:
    print(0)

elif n!=0:
    TARGET = T[0]
    for t in T:
        if abs(t)<abs(TARGET):
            TARGET = t
        elif abs(t)==abs(TARGET):
            if t>0:
                TARGET = t
    print(TARGET)
