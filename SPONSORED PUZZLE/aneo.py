import sys
import math
from decimal import Decimal

# To avoid rounding error we will keep the velocity as km per h
# but convert it to m per s when calculation is needed.
# Velocity should always be converted to km per h
# and casted into an Integer.
# We will also try to avoid division as much as possible.
#
# Given a light at a distance "D" and a flickering period "P",
# there is a time "T" at with the light is green.
# We get a first window to pass through the light when " T c [0; P[ ".
# A second one will open for "T c [2P; 3P[".
# So for any given integer "i" we have " Ti c [2iP; (2i+1)P[ "
# We will call "i" the Interval Index
#
# For a given velocity, the corresponding interval index "i" can be deduced.
# " i = D mod (2PV) "
#
# For a given velocity, distance and period we can then evaluate
# if the car will reach the light during a green time window or not.

def log(*args, **kwargs):
    print(*args, flush=True, file=sys.stderr, **kwargs)

def t_i(i, p):
    return 2*i*p, (2*i + 1)*p

def compute_index(d, p, v):
    i = (3.6 * d) / (2*p*v)
    return int(i)

def time_to_target(d, v):
    t = 3.6 * d / v
    return t

def success(d, p, v):
    i = compute_index(d, p, v)
    tmin, tmax = t_i(i, p)
    t = time_to_target(d, v)
    return tmin <= t < tmax

def find_valid_velocity_below(d, p, v):
    i = compute_index(d, p, v)
    tmin, tmax = t_i(i, p)
    t = time_to_target(d, v)
    if t >= tmax:
        i += 1
        tmin, tmax = t_i(i, p)
    else:
        pass
    v = int(3.6 * d / tmin)
    return v

maxspeed = int(input())
light_count = int(input())

lights = []
for i in range(light_count):
    light = tuple(map(int, input().split()))
    lights.append(light)

v = maxspeed
log('start')

while any(map(lambda l: not success(*l, v), lights)):
    for d, p in filter(lambda l: not success(*l, v), lights):
        v = find_valid_velocity_below(d, p, v)

print(v)
