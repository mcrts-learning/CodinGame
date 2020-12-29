import sys
import math
import pprint
from collections import namedtuple, defaultdict


def log(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)


class UndirectedGraph(namedtuple('Graph', 'nodes, edges')):
    def __init__(self, nodes, edges):
        self._map = defaultdict(set)
        for k1, k2 in edges:
            self._map[k1].add(k2)
            self._map[k2].add(k1)

    @classmethod
    def construct(cls, nodes, edges):
        nodes = frozenset(nodes)
        edges = frozenset(edges)
        return cls(nodes, edges)

    @staticmethod
    def to_edge(e1, e2):
        return frozenset([e1, e2])

    def remove_edge(self, edge):
        if not isinstance(edge, frozenset):
            edge = self.to_edge(edge[0], edge[1])
        edges = self.edges.difference(set([edge]))
        return self.construct(self.nodes, edges)


class State(namedtuple('State', 'graph, exits, skynet')):
    def __init__(self, graph, exits, skynet):
        self.node_weight = dict()
        self.node_distance = dict()
        self.edge_weight = dict()
        self.removable_links = set()

    @classmethod
    def construct(cls, graph, exits, skynet=None):
        exits = frozenset(exits)
        return cls(graph, exits, skynet)

    def update_skynet(self, skynet):
        return self.construct(self.graph, self.exits, skynet)

    def remove_link(self, link):
        graph = self.graph.remove_edge(link)
        return self.construct(graph, self.exits, self.skynet)

    def get_adjacent(self, node):
        return self.graph._map[node]

    def compute_properties(self):
        self.removable_links = self.compute_removable_edges()
        self.node_weight = self.compute_node_weight()
        self.node_distance = self.compute_node_distance()
        self.edge_weight = self.compute_edge_weight()

    def compute_removable_edges(self):
        links = set()
        for e in self.exits:
            nodes = self.get_adjacent(e)
            edges = map(lambda x: self.graph.to_edge(e, x), nodes)
            links = links.union(edges)
        return links

    def compute_node_distance(self):
        skynet = self.skynet
        prop = defaultdict(lambda: float('inf'))
        prop[skynet] = 0
        frontier = [skynet]
        discovered = set()
        while frontier:
            node = frontier.pop(0)
            distance = prop[node]
            discovered.add(node)
            for child in self.get_adjacent(node):
                if child not in discovered and child not in self.exits:
                    frontier.append(child)
                    discovered.add(child)
                    prop[child] = distance + 1
        return prop

    def compute_node_weight(self):
        prop = defaultdict(lambda: 0)
        for node in self.exits:
            for child in self.get_adjacent(node):
                prop[child] += 1
        return prop

    def compute_edge_weight(self):
        prop = dict()
        for edge in self.removable_links:
            n = tuple(edge.difference(self.exits))[0]
            distance = self.node_distance[n]
            weight = self.node_weight[n]
            adj_weight = sum([self.node_weight[x] for x in self.get_adjacent(n)])
            prop[edge] = (
                distance > 0,
                -weight,
                distance - weight >= 0,
                -adj_weight,
                distance
            )
        return prop

N, L, E = map(int, input().split())

links = [UndirectedGraph.to_edge(*input().split()) for _ in range(L)]
exits = [input() for _ in range(E)]
graph = UndirectedGraph.construct(range(N), links)
state = State.construct(graph, exits)


while True:
    skynet = input()
    state = state.update_skynet(skynet)
    state.compute_properties()

    link = min(state.removable_links, key=lambda x: state.edge_weight.get(x))
    print(' '.join(link))
    state = state.remove_link(link)
