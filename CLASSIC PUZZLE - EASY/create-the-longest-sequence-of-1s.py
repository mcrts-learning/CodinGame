import sys
import math

s = input()
counter = list(map(len, s.split('0')))
zipped = map(sum, zip(counter[:-1], counter[1:]))
maximum = max(zipped)
print(maximum + 1)
