import sys
import math
from collections import defaultdict
import heapq as hq
import numpy as np

def debug(msg, *args):
    print(msg, args, file=sys.stderr, flush=True)

ROW, COL, COUNTER = map(int, input().split())

DIRECTIONS = {
    (0, 1): 'RIGHT',
    (1, 0): 'DOWN',
    (-1, 0): 'UP',
    (0, -1): 'LEFT'
}


def compute_direction(p1, p2):
    v = (p2[0] - p1[0], p2[1] - p1[1])
    return DIRECTIONS[v]

class Heuristic:
    @staticmethod
    def euclidian_distance(target, pos):
        targetx, targety = target
        posx, posy = pos
        return math.sqrt((targetx -posx)**2 + (targety -posy)**2)

    @staticmethod
    def distance(target, pos):
        targetx, targety = target
        posx, posy = pos
        return abs((targetx - posx)) + abs((targety - posy))


class Grid:
    def __init__(self, shape, grid, terminal=None, control=None):
        self.grid = grid
        self.terminal = terminal
        self.control = control
        self.shape = shape
        self.unexplored = set()

    @classmethod
    def fromstrings(cls, strings, shape):
        grid = np.array([list(row) for row in strings])
        terminal = None
        control = None
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                if value == 'T':
                    terminal = (r, c)
                elif value == 'C':
                    control = (r, c)

        return cls(shape, grid, terminal, control)

    def compute_unreacheable(self):
        frontier = set([self.terminal])
        path = set([self.terminal])

        while frontier:
            current = frontier.pop()
            childs = list(filter(lambda x: x not in path, self.get_adjacent(current)))
            frontier.update(filter(lambda x: self.get(x) != '?', childs))
            path.update(childs)

        for (row, col), _ in np.ndenumerate(self.grid):
            if (row, col) not in path:
                self.grid[row, col] = '#'

        self.unexplored = set(filter(lambda x: self.get(x) == '?', path))

    def display(self):
        for row in self.grid:
            debug(''.join(row))

    def get(self, pos):
        row, col = pos
        return self.grid[row, col]

    def get_adjacent(self, pos):
        row, col = pos
        adjacents = [
            (row, col - 1),
            (row, col + 1),
            (row - 1, col),
            (row + 1, col),
        ]
        res = filter(lambda x: 0 <= x[0] < self.shape[0], adjacents)
        res = filter(lambda x: 0 <= x[1] < self.shape[1], res)
        res = filter(lambda x: self.get(x) != '#', res)
        return res

    def is_explored(self):
        return not bool(self.unexplored)


class PathFinder:
    def __init__(self, grid):
        self.grid = grid

    def bfs(self, pos, target):
        current = pos
        frontier = []
        explored = set([kirk])
        tree = dict()
        grid = self.grid

        while grid.get(current) != target:
            childs = grid.get_adjacent(current)
            for child in childs:
                if child not in explored:
                    tree[child] = current
                    explored.add(child)
                    frontier.append(child)

                if grid.get(child) == target:
                    return tree, child
            current = frontier.pop(0)

    def astar(self, pos, target, heuristic):
        current = pos
        grid = self.grid
        frontier = []
        explored = set()
        costmap = dict()
        tree = dict()

        d = heuristic(current, target)
        costmap[current] = 0
        hq.heappush(frontier, (d, current))

        while frontier:
            _, current = hq.heappop(frontier)
            cost = costmap[current]

            if current == target:
                break
            childs = grid.get_adjacent(current)
            for child in childs:
                d = heuristic(child, target)
                child_cost = cost + 1
                if (child not in explored) or (costmap.get(child, float('inf')) > child_cost):
                    tree[child] = current
                    costmap[child] = child_cost
                    hq.heappush(frontier, (child_cost + d, child))
                explored.add(child)
        return tree, current

    @staticmethod
    def compute_path(tree, start, end):
        path = []
        while start != end:
            prev = end
            end = tree[prev]
            path.insert(0, compute_direction(end, prev))
        return path


stage = 'explore'
FUEL = 1000
while FUEL:
    FUEL -= 1
    kirk = tuple(map(int, input().split()))
    grid = Grid.fromstrings([input() for _ in range(ROW)], (ROW, COL))
    grid.compute_unreacheable()
    grid.display()

    if stage == 'explore' and grid.is_explored():
        stage = 'to_control'
    elif stage == 'to_control' and kirk == grid.control:
        stage = 'to_terminal'

    pathfinder = PathFinder(grid)
    if stage == 'explore':
        debug(stage)
        tree, end = pathfinder.bfs(kirk, '?')
        path = pathfinder.compute_path(tree, kirk, end)
    elif stage == 'to_control':
        debug(stage)
        tree, end = pathfinder.astar(kirk, grid.control, Heuristic.distance)
        path = pathfinder.compute_path(tree, kirk, end)
    elif stage == 'to_terminal':
        debug(stage)
        tree, end = pathfinder.astar(kirk, grid.terminal, Heuristic.distance)
        path = pathfinder.compute_path(tree, kirk, end)

    print(path[0])
