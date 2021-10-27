import sys
import numpy as np
import heapq as hq

from dataclasses import dataclass, field
from typing import Iterable, Tuple, Mapping, Set
from collections import defaultdict

Node = Tuple[int, int]

@dataclass(frozen=True, order=True)
class Path:
    node: Node
    sequence: Tuple[Node, ...]
    distance: int

    def append(self, node: Node, d: int) -> "Path":
        sequence = self.sequence + tuple([node])
        return type(self)(node, sequence, d)

@dataclass
class GridGraph:
    grid: np.ndarray

    def neighbours(self, index: Node) -> Iterable[Node]:
        nmax = self.grid.shape[0]
        x, y = index
        v = self.grid[index]
        neighbours = list()
        if x > 0:
            neighbours.append((x-1, y))
        if x < nmax-1:
            neighbours.append((x+1, y))
        if y > 0:
            neighbours.append((x, y-1))
        if y < nmax-1:
            neighbours.append((x, y+1))
        return [n for n in neighbours if self.grid[n].lower() == v.lower()]
        

    def bfs(self, index: Node) -> Mapping[Node, int]:
        paths = list()
        queue = list()
        distance = self.grid[index].islower()
        pstart = Path(index, (index, ), distance)
        hq.heappush(queue, (pstart.distance, pstart))

        while queue:
            d, path = hq.heappop(queue)
            is_leaf = True
            for child in self.neighbours(path.node):
                if child not in path.sequence:
                    is_leaf = False
                    p = path.append(child, path.distance + self.grid[child].islower())
                    hq.heappush(queue, (p.distance, p))
            
            if is_leaf:
                paths.append(path)
        
        return {p.node: p.distance for p in paths}

    def compute_longest_distance(self, index: Node) -> Tuple[int, Set[Node]]:
        distances = self.bfs(index)
        distances = self.bfs(max(distances, key=distances.get))
        d = max(distances.values())
        return d, set(distances.keys())

    def find_longest_roads(self) -> Mapping[str, int]:
        longest_roads = defaultdict(int)
        discovered = set()
        for index, v in np.ndenumerate(self.grid):
            if v != '#' and index not in discovered:
                d, indexes = self.compute_longest_distance(index)
                discovered |= indexes
                if longest_roads[v.upper()] < d:
                    longest_roads[v.upper()] = d

        return longest_roads

n = int(input())
arr = []
for i in range(n):
    line = input()
    arr.append(list(line))
grid = np.array(arr, dtype=str)

graph = GridGraph(grid)
longest_roads = graph.find_longest_roads()

player = max(longest_roads, key=longest_roads.get)
if longest_roads[player] < 5:
    print(0)
else:
    print(player, longest_roads[player])