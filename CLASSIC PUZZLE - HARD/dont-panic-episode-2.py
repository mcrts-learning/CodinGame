from typing import DefaultDict, Iterable, MutableSet, Tuple, FrozenSet
from dataclasses import dataclass, field
from collections import Counter
import heapq as hq

import numpy as np

WAIT = ("WAIT",)
ELEVATOR = ("ELEVATOR",) + 3*WAIT
BLOCK = ("BLOCK",) + 2*WAIT

@dataclass(frozen=True, order=True)
class Node:
    position: Tuple[int, int, int]
    label: str = field(default='', compare=False)

    @property
    def y(self):
        return self.position[0]

    @property
    def x(self):
        return self.position[1]

    @property
    def direction(self):
        return self.position[2]

@dataclass
class NodeProperties:
    is_elevator: bool = False
    is_exit: bool = False
    is_start: bool = False
    can_walk: bool = False
    can_block: bool = False
    can_use_elevator: bool = False
    can_put_elevator: bool = False
    can_exit: bool = False

    def __repr__(self):
        return f"""{self.__class__.__name__}({', '.join([f"{k}={v}" for k, v in self.__dict__.items() if v is True])})"""

@dataclass(frozen=True, order=True)
class Edge:
    node_from: Node
    node_to: Node
    label: str = field(default="", compare=False)
    sequence: Iterable[str] = field(default_factory=tuple, compare=False)

    @classmethod
    def walk_edge(cls, n1, n2):
        return cls(n1, n2, "WALK", WAIT*abs(n1.x - n2.x))

    @classmethod
    def block_edge(cls, n1, n2):
        return cls(n1, n2, "BLOCK", BLOCK + WAIT*abs(n1.x - n2.x))
    
    @classmethod
    def exit_edge(cls, n1, n2):
        return cls(n1, n2, "EXIT", WAIT)

    @classmethod
    def use_elevator_edge(cls, n1, n2):
        return cls(n1, n2, "USE ELEVATOR", WAIT)

    @classmethod
    def put_elevator_edge(cls, n1, n2):
        return cls(n1, n2, "PUT ELEVATOR", ELEVATOR)


@dataclass
class Graph:
    nodes: MutableSet[Node] = field(default_factory=set)
    node_properties: DefaultDict[Node, NodeProperties] = field(
        default_factory=lambda: DefaultDict(NodeProperties)
    )
    edges: MutableSet[Edge] = field(default_factory=set)

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, edge):
        self.edges.add(edge)

    def get_start(self):
        return next(filter(lambda n: self.node_properties[n].is_start, self.nodes))

    def get_exit(self):
        return next(filter(lambda n: self.node_properties[n].is_exit, self.nodes))

    def get_out_edges(self, node):
        return filter(lambda e: e.node_from == node, self.edges)

    def get_in_edges(self, node):
        return filter(lambda e: e.node_to == node, self.edges)

@dataclass
class Factory:
    time: int
    n_clones: int
    n_elevators: int

    def add_step_node(self, graph, node):
        for i in range(0, self.n_elevators):
            y = node.y - i - 1
            n = Node((y, node.x, node.direction))
            graph.add_node(n)
            graph.node_properties[n].can_put_elevator = True
            if y != 0:
                graph.node_properties[n].can_block = True
                graph.node_properties[n].can_walk = True
    
    def add_exit_node(self, graph, y, x):
        n = Node((y, x, 0), "EXIT")
        graph.add_node(n)
        graph.node_properties[n].is_exit = True

        n1 = Node((y, x, 1), "EXIT PATH")
        graph.add_node(n1)
        graph.node_properties[n1].can_exit = True
        self.add_step_node(graph, n1)
        n2 = Node((y, x, -1), "EXIT PATH")
        graph.add_node(n2)
        graph.node_properties[n2].can_exit = True
        self.add_step_node(graph, n2)

    def add_start_node(self, graph, y, x):
        n = Node((y, x, 1), "START")
        graph.add_node(n)
        graph.node_properties[n].is_start = True
        graph.node_properties[n].can_block = True
        graph.node_properties[n].can_walk = True

    def add_elevator_node(self, graph, y, x, d):
        n = Node((y, x, d), "ELEVATOR")
        graph.add_node(n)
        graph.node_properties[n].is_elevator = True
        graph.node_properties[n].can_use_elevator = True
        graph.node_properties[n].can_block = False
        graph.node_properties[n].can_walk = False
        graph.node_properties[n].can_put_elevator = False

        n_out = Node((y + 1, x, d))
        graph.add_node(n_out)
        if not graph.node_properties[n_out].is_elevator:
            graph.node_properties[n_out].can_block = True
            graph.node_properties[n_out].can_walk = True

        self.add_step_node(graph, n)
        

    def add_edges(self, graph):
        n_exit = next(filter(lambda n: graph.node_properties[n].is_exit, graph.nodes))
        for n in filter(lambda n: graph.node_properties[n].can_exit, graph.nodes):
            edge = Edge.exit_edge(n, n_exit)
            graph.add_edge(edge)

        for n1 in filter(lambda n: graph.node_properties[n].can_walk, graph.nodes):
            nodes = filter(lambda n: n1.y == n.y, graph.nodes)
            nodes = filter(lambda n: n1.x * n1.direction < n.x * n1.direction and n1.direction == n.direction, nodes)
            for n2 in nodes:
                edge = Edge.walk_edge(n1, n2)
                graph.add_edge(edge)

        for n1 in filter(lambda n: graph.node_properties[n].can_block, graph.nodes):
            nodes = filter(lambda n: n1.y == n.y, graph.nodes)
            nodes = filter(lambda n: n1.x * n1.direction > n.x * n1.direction and n1.direction != n.direction, nodes)
            for n2 in nodes:
                edge = Edge.block_edge(n1, n2)
                graph.add_edge(edge)
        
        for n1 in filter(lambda n: graph.node_properties[n].can_use_elevator, graph.nodes):
            n2 = next(filter(lambda n: n.position == (n1.y + 1, n1.x, n1.direction), graph.nodes))
            edge = Edge.use_elevator_edge(n1, n2)
            graph.add_edge(edge)

        for n1 in filter(lambda n: graph.node_properties[n].can_put_elevator, graph.nodes):
            n2 = next(filter(lambda n: n.position == (n1.y + 1, n1.x, n1.direction), graph.nodes))
            edge = Edge.put_elevator_edge(n1, n2)
            graph.add_edge(edge)

    def parse_grid(self, grid):
        graph = Graph()
        for (y, x), value in sorted(np.ndenumerate(grid), reverse=True):
            if value == 'X':
                self.add_exit_node(graph, y, x)
            if value == 'O':
                self.add_start_node(graph, y, x)
            if value == '^':
                self.add_elevator_node(graph, y, x, 1)
                self.add_elevator_node(graph, y, x, -1)
        self.add_edges(graph)
        return graph

@dataclass(frozen=True)
class Path:
    node: Node
    sequence: Tuple[str] = field(default_factory=tuple)
    nodes: FrozenSet[Node] = field(default_factory=frozenset)

    @property
    def time(self):
        return len(self.sequence)

    @property
    def cost(self):
        counter = Counter(self.sequence)
        return self.time, counter['BLOCK'] + counter['ELEVATOR'], counter['ELEVATOR']

    def __gt__(self, other):
        return self.time > other.time

    def append_edge(self, edge):
        node = edge.node_to
        sequence = self.sequence + edge.sequence
        nodes = self.nodes | set([node])
        return type(self)(node, sequence, nodes)
    
    def get_childs(self, graph):
        childs = list()
        for e in graph.get_out_edges(self.node):
            if e.node_to not in self.nodes:
                childs.append(self.append_edge(e))
        return childs

@dataclass
class Solver:
    time: int
    clone: int
    elevator: int

    @property
    def cost(self):
        return self.time, self.clone, self.elevator

    def check_cost(self, path):
        return all([a <= b for a, b in zip(path.cost, self.cost)])

    def bfs(self, graph, start, end):
        discovered = set()
        queue = list()
        pstart = Path(start)
        discovered.add(pstart)
        hq.heappush(queue, (pstart.time, pstart))

        while queue:
            _, current = hq.heappop(queue)
            if graph.node_properties[current.node].is_exit:
                return True, current

            for child in current.get_childs(graph):
                if (child not in discovered and self.check_cost(child)):
                    discovered.add(child)
                    hq.heappush(queue, (child.time, child))

        return False, None

from pprint import pprint
grid = np.flip(np.array([
[" ", "X", " ", " ", " "],
[" ", " ", " ", " ", " "],
[" ", " ", " ", " ", " "],
[" ", " ", " ", " ", " "],
[" ", " ", " ", "O", " "],
]), axis=0)
time, clones, elevators = (1000, 20, 2)
factory = Factory(time, clones, elevators)
graph = factory.parse_grid(grid)
print(f"{time=} {clones=} {elevators=}")
print(f"Nodes {len(graph.nodes)} Edges {len(graph.edges)}")

pprint([g for g in graph.nodes if len(list(graph.get_in_edges(g))) == 0])

solver = Solver(time, clones, elevators)
start = graph.get_start()
end = graph.get_exit()
print(start, end)
success, path = solver.bfs(graph, start, end)

if success:
    print("Path found")
    print(path.node, path.cost, path.time)
    print(path.sequence)
    print(path.nodes)
else:
    print("Path not found")