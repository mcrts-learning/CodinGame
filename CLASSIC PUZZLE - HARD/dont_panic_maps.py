from typing import Iterable
import numpy as np

INPUTS = {
    "2 MISSING ELEVATORS": [
        "6 13 100 5 1 10 2 3",
        "4 1",
        "0 4",
        "2 7",
        "0 10 RIGHT",
    ],
    "TRAP": [
        "13 36 67 11 12 41 4 34",
        "6 34",
        "2 23",
        "10 3",
        "7 34",
        "5 4",
        "10 23",
        "1 24",
        "11 11",
        "10 34",
        "8 23",
        "6 13",
        "6 22",
        "11 4",
        "9 2",
        "4 9",
        "0 34",
        "2 34",
        "3 17",
        "4 23",
        "4 34",
        "1 17",
        "1 4",
        "9 17",
        "5 34",
        "2 3",
        "8 9",
        "8 1",
        "7 17",
        "9 34",
        "1 34",
        "2 24",
        "3 34",
        "11 13",
        "8 34",
        "0 6 RIGHT",
    ],
    "GIANT MAP": [
        "13 69 109 11 47 100 4 36",
        "2 56",
        "4 23",
        "8 1",
        "3 30",
        "4 9",
        "9 17",
        "11 45",
        "6 9",
        "1 24",
        "7 48",
        "3 24",
        "8 63",
        "10 45",
        "9 2",
        "2 23",
        "3 17",
        "10 3",
        "1 36",
        "2 9",
        "10 23",
        "1 62",
        "1 17",
        "1 4",
        "8 23",
        "2 43",
        "2 3",
        "6 3",
        "6 23",
        "5 4",
        "6 35",
        "11 4",
        "11 50",
        "8 9",
        "1 50",
        "2 24",
        "3 60",
        "0 6 RIGHT",
    ],
    "BEST PATH": [
        "10 19 47 9 9 41 0 17",
        "3 4",
        "4 3",
        "7 4",
        "1 17",
        "8 9",
        "4 9",
        "2 3",
        "0 3",
        "5 4",
        "7 17",
        "1 4",
        "3 17",
        "2 9",
        "6 9",
        "5 17",
        "0 9",
        "6 3",
        "0 6 RIGHT",
    ],
    "FEW CLONES": [
        "13 69 79 11 39 8 5 30",
        "6 65",
        "11 4",
        "8 34",
        "8 56",
        "7 17",
        "8 1",
        "2 24",
        "11 13",
        "10 23",
        "6 13",
        "6 34",
        "5 4",
        "1 50",
        "5 46",
        "3 17",
        "10 3",
        "11 42",
        "1 17",
        "1 4",
        "2 23",
        "8 66",
        "2 3",
        "1 24",
        "1 34",
        "8 9",
        "2 58",
        "11 11",
        "11 38",
        "8 23",
        "6 57",
        "0 33 RIGHT",
    ],
    "BEST PATH MISSING ELEVATORS": [
        "10 19 42 9 9 41 1 16",
        "4 3",
        "7 4",
        "1 17",
        "8 9",
        "4 9",
        "2 3",
        "0 3",
        "5 4",
        "7 17",
        "1 4",
        "3 17",
        "2 9",
        "6 9",
        "5 17",
        "0 9",
        "6 3",
        "0 6 RIGHT",
    ]
}

def parse(inputs: Iterable[str]):
    height, width, time, exit_y, exit_x, n_clone, n_elevator, n_starting_elevator = [int(i) for i in next(inputs).split()]
    grid = np.full((height, width), " ", dtype=str)
    grid[(exit_y, exit_x)] = 'X'
    for _ in range(n_starting_elevator):
        y, x = next(inputs).split()
        grid[(int(y), int(x))] = '^'

    y, x, direction = next(inputs).split()
    grid[(int(y), int(x))] = 'O'
    return grid, (height, width, time, exit_y, exit_x, n_clone, n_elevator, n_starting_elevator)

def TRAP():
    return parse(iter(INPUTS["TRAP"]))

def GIANT_MAP():
    return parse(iter(INPUTS["GIANT MAP"]))

def BEST_PATH():
    return parse(iter(INPUTS["BEST PATH"]))

def FEW_CLONES():
    return parse(iter(INPUTS["FEW CLONES"]))

def TWO_MISSING_ELEVATOR():
    return parse(iter(INPUTS["2 MISSING ELEVATORS"]))
    
def BEST_PATH_MISSING_ELEVATORS():
    return parse(iter(INPUTS["BEST PATH MISSING ELEVATORS"]))
    


SOLUTIONS = {
    "ELEVATOR": ('WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'ELEVATOR', 'WAIT', 'WAIT', 'WAIT', 'WAIT'),
    "BEST PATH MISSING ELEVATORS": ('WAIT', 'WAIT', 'WAIT', 'WAIT', 'BLOCK', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'ELEVATOR', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'BLOCK', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'BLOCK', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'BLOCK', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT'),
    "GIANT MAP": ('BLOCK', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'ELEVATOR', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'BLOCK', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'BLOCK', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'BLOCK', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'ELEVATOR', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'WAIT', 'ELEVATOR', 'WAIT', 'WAIT', 'WAIT', 'ELEVATOR', 'WAIT', 'WAIT', 'WAIT', 'WAIT'),
}