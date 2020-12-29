import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

def split(part, st, ed):
    pst, ped = part
    if st <= pst and ed >= ped:
        return []
    elif st > pst and ed < ped:
        return [(pst, st), (ed, ped)]
    elif pst < st < ped:
        return [(pst, st)]
    elif pst < ed < ped:
        return [(ed, ped)]
    elif st >= ped:
        return [part]
    elif ed <= pst:
        return [part]

l = int(input())
fence = [(0, l)]
n = int(input())
for i in range(n):
    st, ed = [int(j) for j in input().split()]
    newfence = []
    for f in fence:
        p = split(f, st, ed)
        newfence.extend(p)
    fence = newfence
    print(fence, file=sys.stderr)

if fence:
    for fst, fed in fence:
        print(f'{fst} {fed}')
else:
   print('All painted')
