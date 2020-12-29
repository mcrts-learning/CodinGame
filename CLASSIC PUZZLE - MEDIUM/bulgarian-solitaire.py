import sys
import math
from collections import Counter

def debug(msg):
    print("DEBUG : {}".format(msg), file=sys.stderr, flush=True)

def next_step(piles):
    new_piles = list(filter(bool, map(lambda x: x - 1, piles)))
    new_piles.append(len(piles))
    return new_piles

history = list()

n = int(input())
piles = list(filter(bool, map(int, input().split())))
c = Counter(piles)

while c not in history:
    history.append(c)
    piles = next_step(piles)
    c = Counter(piles)

index = history.index(c)
length = len(history) - index
print(length)
