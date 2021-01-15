import sys

n = int(input())
v = int(input())
m = int(input())
array = [None for _ in range(n**2)]

def get_path(array, idx):
    if idx == 0:
        return []
    else:
        pidx = int((idx - 1) / 2)
        path = get_path(array, pidx)
        return path + ['Right' if idx % 2 == 0 else 'Left']

for i in range(m):
    p, l, r = [int(j) for j in input().split()]
    if i == 0:
        array[0] = p
        idx = 0
    else:
        idx = array.index(p)
    array[2*idx + 1] = l
    array[2*idx + 2] = r

idx = array.index(v)
if idx == 0:
    print('Root')
else:
    path = get_path(array, idx)
    print(' '.join(path))
