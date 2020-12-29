import sys
import math
import itertools
import decimal


def volume(d):
    return (math.pi * d**3) / 6

def diameter(v):
    r = (3 * v / (4 * math.pi)) ** (1./3.)
    return math.floor(r*2)

def surface(d):
    return math.pi * d**2

def evaluate(d1, d2):
    return round(volume(d1) + volume(d2), 4)

def is_valid(target, c1, c2):
    res = target == evaluate(c1, c2)
    return res

def is_fun(k1, k2, c1, c2):
    res = len(set([k1, k2, c1, c2])) == 4
    return res

def interest(c1, c2):
    res = surface(c2) - surface(c1)
    return res

def combination(dmin, dmax, target):
    for d1 in range(dmin, dmax + 1):
        v = target - volume(d1)
        d2min = max([diameter(v), d1])
        d2max = min([d2min + 1, dmax]) + 1
        for d2 in range(d2min, d2max):
            yield (d1, d2)

orb_size_min, orb_size_max = [int(i) for i in input().split()]
k1, k2 = [int(i) for i in input().split()]

print((orb_size_min, orb_size_max), file=sys.stderr, flush=True)
print((k1, k2), file=sys.stderr, flush=True)
target = evaluate(k1, k2)
cmax = min([diameter(target), orb_size_max])
valid = filter(lambda x: is_valid(target, x[0], x[1]), combination(orb_size_min, cmax, target))
fun = filter(lambda x: is_fun(k1, k2, x[0], x[1]), valid)
try:
    res = max(fun, key=lambda x: interest(x[0], x[1]))
    print('{} {}'.format(*res))
except ValueError:
    print("VALID")
