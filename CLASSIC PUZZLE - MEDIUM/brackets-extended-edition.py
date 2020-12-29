import sys
import math
from functools import reduce

def log(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)

FAMILY = {
    '(': 0,
    ')': 0,
    '[': 1,
    ']': 1,
    '{': 2,
    '}': 2,
    '<': 3,
    '>': 3
}

def reduction(expression):
    def combination(c1, c2):
        if c1 and FAMILY[c1[-1]] == FAMILY[c2]:
            return c1[:-1]
        else:
            return c1 + c2
    return reduce(combination, expression)

N = int(input())
for _ in range(N):
    expression = ''.join(filter(lambda x: x in FAMILY.keys(), input()))
    if reduction(expression) is '':
        print('true')
    else:
        print('false')
