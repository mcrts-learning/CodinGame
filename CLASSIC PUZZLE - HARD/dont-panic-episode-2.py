from __future__ import annotations
import sys

def log(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)

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
    is_start: bool = False
    is_end: bool = False
    is_elevator: bool = False
    can_end: bool = False
    is_in: bool = False
    is_out: bool = False

    def __repr__(self):
        return f"""{self.__class__.__name__}({', '.join([f"{k}={v}" for k, v in self.__dict__.items() if v is True])})"""

@dataclass(frozen=True, order=True)
class Edge:
    node_from: Node
    node_to: Node
    label: str = field(default="", compare=False)
    sequence: Iterable[str] = field(default_factory=tuple)

    def __add__(self, other: Edge) -> Edge:
        node_from = self.node_from
        node_to = other.node_to
        label = f"{self.label} {other.label}"
        sequence = self.sequence + other.sequence
        return type(self)(node_from, node_to, label, sequence)

    @classmethod
    def walk_edge(cls, n1, n2):
        return cls(n1, n2, "WALK", WAIT*abs(n1.x - n2.x))

    @classmethod
    def block_edge(cls, n1, n2):
        return cls(n1, n2, "BLOCK", BLOCK + WAIT*abs(n1.x - n2.x))
    
    @classmethod
    def end_edge(cls, n1, n2):
        return cls(n1, n2, "END", WAIT)

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

    def remove_node(self, node):
        self.nodes.remove(node)

    def remove_edge(self, edge):
        self.edges.remove(edge)

    def get_start(self):
        return next(filter(lambda n: self.node_properties[n].is_start, self.nodes))

    def get_end(self):
        return next(filter(lambda n: self.node_properties[n].is_end, self.nodes))

    def out_edges(self, node: Node) -> Iterable[Node]:
        return filter(lambda e: e.node_from == node, self.edges)

    def in_edges(self, node: Node) -> Iterable[Node]:
        return filter(lambda e: e.node_to == node, self.edges)

    def graph_reduction(self):
        end = self.get_end()
        for node in list(
            filter(lambda n: n.position[0] >= end.position[0],
            filter(lambda n: not(
                self.node_properties[n].is_end
                or self.node_properties[n].is_start
                or self.node_properties[n].can_end
                ), self.nodes))
        ):
            in_edges = list(self.in_edges(node))
            out_edges = list(self.out_edges(node))
            for e in out_edges:
                self.remove_edge(e)
            for e in in_edges:
                self.remove_edge(e)
            self.remove_node(node)

        for node in list(filter(lambda n: not(
                self.node_properties[n].is_end
                or self.node_properties[n].is_start
            ), self.nodes)):
            in_edges = list(self.in_edges(node))
            out_edges = list(self.out_edges(node))
            if (len(in_edges) == 1 and len(out_edges) == 1):
                in_edge = in_edges[0]
                out_edge = out_edges[0]
                edge = Edge(
                    node_from=in_edge.node_from,
                    node_to=out_edge.node_to,
                    label=f"{in_edge.label} + {out_edge.label}",
                    sequence=in_edge.sequence + out_edge.sequence
                )
                self.add_edge(edge)
                self.remove_edge(in_edge)
                self.remove_edge(out_edge)
                self.remove_node(node)
        
        for node in list(filter(lambda n: not(
                self.node_properties[n].is_end
                or self.node_properties[n].is_start
            ), self.nodes)):
            in_edges = list(self.in_edges(node))
            out_edges = list(self.out_edges(node))
            if (len(in_edges) == 0 or len(out_edges) == 0):
                for e in out_edges:
                    self.remove_edge(e)
                for e in in_edges:
                    self.remove_edge(e)
                self.remove_node(node)

@dataclass
class Factory:
    time: int
    n_clones: int
    n_elevators: int

    def add_step_node(self, graph, node):
        for _ in range(1, self.n_elevators):
            y = node.y - 1
            if y <= 0:
                continue
            n = Node((y, node.x, node.direction))
            graph.add_node(n)
            graph.node_properties[n].is_in = True
            
            edge = Edge.put_elevator_edge(n, node)
            graph.add_edge(edge)
            node = n

    def add_end_node(self, graph, y, x):
        end_node = Node((y, x, 0), "END")
        graph.add_node(end_node)
        graph.node_properties[end_node].is_end = True

        n1 = Node((y, x, 1), "END PATH")
        n2 = Node((y, x, -1), "END PATH")
        for n in [n1, n2]:
            graph.add_node(n)
            graph.node_properties[n].is_in = True
            graph.node_properties[n].can_end = True
            e = Edge.end_edge(n, end_node)
            graph.add_edge(e)
            self.add_step_node(graph, n)

    def add_start_node(self, graph, y, x):
        n = Node((y, x, 1), "START")
        graph.add_node(n)
        graph.node_properties[n].is_start = True
        graph.node_properties[n].is_out = True

    def add_elevator_node(self, graph, y, x, d):
        n = Node((y, x, d), "ELEVATOR")
        graph.add_node(n)
        graph.node_properties[n].is_elevator = True
        graph.node_properties[n].is_in = True

        n_out = Node((y + 1, x, d))
        graph.add_node(n_out)
        graph.node_properties[n_out].is_in = True
        if not (graph.node_properties[n_out].is_elevator or graph.node_properties[n_out].can_end):
            graph.node_properties[n_out].is_out = True
        
        edge = Edge.use_elevator_edge(n, n_out)
        graph.add_edge(edge)

        self.add_step_node(graph, n)
        

    def add_edges(self, graph: Graph):
        for n1 in filter(lambda n: graph.node_properties[n].is_out, graph.nodes):
            nodes = filter(lambda n: graph.node_properties[n].is_in, graph.nodes)
            nodes = filter(lambda n: n1.y + 1 == n.y, nodes)
            nodes = filter(lambda n: n1.x * n1.direction <= n.x * n1.direction and n1.direction == n.direction, nodes)
            nodes = filter(
                lambda n: not any([graph.node_properties[Node((n1.y, x, n.direction))].is_elevator for x in range(n1.x, n.x, n.direction)]),
                nodes
            )
            for n2 in nodes:
                n = Node((n1.y, n2.x, n2.direction), "ELEVATOR")
                e1 = Edge.walk_edge(n1, n)
                if graph.node_properties[n].is_elevator:
                    e2 = Edge.use_elevator_edge(n, n2)
                else:
                    e2 = Edge.put_elevator_edge(n, n2)
                e = e1 + e2
                graph.add_edge(e)

        
        for n1 in filter(lambda n: graph.node_properties[n].is_out, graph.nodes):
            nodes = filter(lambda n: graph.node_properties[n].is_in, graph.nodes)
            nodes = filter(lambda n: n1.y + 1 == n.y, nodes)
            nodes = filter(lambda n: n1.x * n1.direction > n.x * n1.direction and n1.direction != n.direction, nodes)
            nodes = filter(
                lambda n: not any([graph.node_properties[Node((n1.y, x, n.direction))].is_elevator for x in range(n1.x, n.x, n.direction)]),
                nodes
            )
            for n2 in nodes:
                n = Node((n1.y, n2.x, n2.direction), "ELEVATOR")
                e1 = Edge.block_edge(n1, n)
                if graph.node_properties[n].is_elevator:
                    e2 = Edge.use_elevator_edge(n, n2)
                else:
                    e2 = Edge.put_elevator_edge(n, n2)
                e = e1 + e2
                graph.add_edge(e)

    def parse_grid(self, grid):
        graph = Graph()
        for (y, x), value in sorted(np.ndenumerate(grid), reverse=True):
            if value == 'X':
                self.add_end_node(graph, y, x)
            if value == 'O':
                self.add_start_node(graph, y, x)
            if value == '^':
                self.add_elevator_node(graph, y, x, 1)
                self.add_elevator_node(graph, y, x, -1)
        self.add_edges(graph)
        return graph


@dataclass(frozen=True, order=True)
class SolverPath:
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

    def append_edge(self, edge: Edge) -> SolverPath:
        node = edge.node_to
        sequence = self.sequence + edge.sequence
        nodes = self.nodes | set([node])
        return type(self)(node, sequence, nodes)
    
    def get_childs(self, graph: Graph):
        childs = list()
        for e in graph.out_edges(self.node):
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

    def check_cost(self, path: SolverPath):
        return all([a <= b for a, b in zip(path.cost, self.cost)])

    def bfs(self, graph: Graph, start: Node, end: Node) -> Tuple[bool, SolverPath | None, int]:
        discovered = set()
        queue = list()
        pstart = SolverPath(start)
        discovered.add(pstart)
        hq.heappush(queue, (pstart.time, pstart))

        counter = 0
        while queue:
            counter += 1
            _, current = hq.heappop(queue)
            if current.node == end:
                return True, current, counter

            for child in current.get_childs(graph):
                if (child not in discovered and self.check_cost(child)):
                    discovered.add(child)
                    hq.heappush(queue, (child.time, child))

        return False, None, counter

    def heuristic(self, target: Node, p: SolverPath) -> int:
        return p.time + 3*abs(target.y - p.node.y) + abs(target.x- p.node.x)
    
    def astar(self, graph: Graph, start: Node, end: Node) -> Tuple[bool, SolverPath | None, int]:
        discovered = set()
        queue = list()
        pstart = SolverPath(start)
        discovered.add(pstart)
        hq.heappush(queue, (self.heuristic(end, pstart), pstart))

        counter = 0
        while queue:
            counter += 1
            _, current = hq.heappop(queue)
            if current.node == end:
                return True, current, counter

            for child in current.get_childs(graph):
                if (child not in discovered and self.check_cost(child)):
                    discovered.add(child)
                    hq.heappush(queue, (self.heuristic(end, child), child))

        return False, None, counter


height, width, time, exit_y, exit_x, n_clone, n_elevator, n_starting_elevator = [int(i) for i in input().split()]
grid = np.full((height, width), " ", dtype=str)
grid[(exit_y, exit_x)] = 'X'

for _ in range(n_starting_elevator):
    y, x = input().split()
    grid[(int(y), int(x))] = '^'
y, x, direction = input().split()
grid[(int(y), int(x))] = 'O'

factory = Factory(time, n_clone, n_elevator)
graph = factory.parse_grid(grid)
graph.graph_reduction()
graph.graph_reduction()

start = graph.get_start()
end = graph.get_end()
solver = Solver(time, n_clone, n_elevator)
success, path, counter = solver.astar(graph, start, end)

for s in path.sequence:
    print(s)
    input()
