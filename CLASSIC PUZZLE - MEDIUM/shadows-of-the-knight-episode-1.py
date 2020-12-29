import sys
import math

# w: width of the building.
# h: height of the building.
w, h = [int(i) for i in input().split()]
n = int(input())  # maximum number of turns before game over.
x0, y0 = [int(i) for i in input().split()]
w_max=w
h_max=h
w_min=0
h_min=0
# game loop
while True:
    bomb_dir = input()  # the direction of the bombs from batman's current location (U, UR, R, DR, D, DL, L or UL)
    for i in bomb_dir:
        if i=="D":
            h_min=y0
            y0=math.floor((h_max+y0)/2)
        if i=="U":
            h_max=y0
            y0=math.floor((h_min+y0)/2)
        if i=="R":
            w_min=x0
            x0=math.floor((w_max+x0)/2)
        if i=="L":
            w_max=x0
            x0=math.floor((w_min+x0)/2)
    print(x0,y0)
