import sys
import math
import itertools as it

def is_square(tripod, yaxis):
    (x1, y1), x2 = tripod
    y2 = y1 + x2 - x1
    return x1 < x2 and y2 in yaxis

w, h, count_x, count_y = [int(i) for i in input().split()]
xaxis = set([0, w] + list(map(int, input().split())))
yaxis = set([0, h] + list(map(int, input().split())))
points = it.product(xaxis, yaxis)
tripods = it.product(points, xaxis)

fun = lambda x: is_square(x, yaxis)
squares = filter(fun, tripods)
print(len(list(squares)))
