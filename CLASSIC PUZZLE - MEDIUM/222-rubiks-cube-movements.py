import sys
import math
import re
import numpy as np

FRONT = [
    ( (2, 2), (2, 3), (3, 3), (3, 2), (2, 2) ),
    ( (1, 2), (2, 4), (4, 3), (3, 1), (1, 2) ),
    ( (1, 3), (3, 4), (4, 2), (2, 1), (1, 3) )
]
BACK = [
    ( (2, 6), (2, 7), (3, 7), (3, 6), (2, 6) ),
	( (2, 5), (0, 2), (3, 0), (5, 3), (2, 5) ),
	( (0, 3), (2, 0), (5, 2), (3, 5), (0, 3) )
]
RIGHT = [
    ( (2, 4), (2, 5), (3, 5), (3, 4), (2, 4) ),
    ( (5, 3), (3, 3), (1, 3), (2, 6), (5, 3) ),
    ( (4, 3), (2, 3), (0, 3), (3, 6), (4, 3) )
]
LEFT = [
    ( (2, 0), (2, 1), (3, 1), (3, 0), (2, 0) ),
	( (0, 2), (2, 2), (4, 2), (3, 7), (0, 2) ),
	( (1, 2), (3, 2), (5, 2), (2, 7), (1, 2) )
]
UP = [
    ( (0, 2), (0, 3), (1, 3), (1, 2), (0, 2) ),
	( (2, 7), (2, 5), (2, 3), (2, 1), (2, 7) ),
	( (2, 6), (2, 4), (2, 2), (2, 0), (2, 6) )
]
DOWN = [
    ( (4, 2), (4, 3), (5, 3), (5, 2), (4, 2) ),
	( (3, 0), (3, 2), (3, 4), (3, 6), (3, 0) ),
	( (3, 1), (3, 3), (3, 5), (3, 7), (3, 1) )
]

def transform(array, moves) -> np.ndarray:
    new_array = np.array(array)
    for cycle in moves:
        for i1, i2 in zip(cycle[0:-1], cycle[1:]):
            new_array[i2] = array[i1]
    return new_array

def operation(array, moves, rev=False):
    if rev:
        moves = [c[::-1] for c in moves]
    return transform(array, moves)

def make_a_move(cube, op, rev, twice):
    moves = {
        'F': FRONT,
        'B': BACK,
        'R': RIGHT,
        'L': LEFT,
        'U': UP,
        'D': DOWN,
    }
    moves = moves[op]
    if twice:
        return operation(operation(cube, moves, rev), moves, rev)
    else:
        return operation(cube, moves, rev)

def parse(string, cube):
    pattern1 = """[FBRLUD]'?2?"""
    pattern2 = """(?P<op>[FBRLUD])(?P<rev>'?)(?P<twice>2?)"""
    
    print(cube, file=sys.stderr)
    for action in re.findall(pattern1, string):
        op, rev, twice = re.match(pattern2, action).groups()
        rev = bool(rev)
        twice = bool(twice)
        cube = make_a_move(cube, op, rev, twice)
    return cube

move = input()
cube = np.full((6, 8), ' ')
cube[0:2, 2:4] = 'U'
cube[2:4, 2:4] = 'F'
cube[4:6, 2:4] = 'D'
cube[2:4, 0:2] = 'L'
cube[2:4, 4:6] = 'R'
cube[2:4, 6:8] = 'B'
cube = parse(move, cube)

print(''.join(cube[2, 2:4]))
print(''.join(cube[3, 2:4]))
