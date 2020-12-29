import sys
import math
import numpy as np
g = np.array([input().split() for _ in range(9)])
gt = g.transpose()
gx = g.reshape(9, 3, 3).swapaxes(0, 1).reshape(9, 9)
valid = lambda x: len(set(x)) == 9
res = all(map(valid, np.concatenate([g, gt, gx])))
print(str(res).lower())
