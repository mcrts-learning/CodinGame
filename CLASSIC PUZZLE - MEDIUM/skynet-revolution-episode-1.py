import sys
import itertools
import math
from collections import defaultdict

def debug(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)


class Network:
    def __init__(self, links=None, gateways=None):
        self._map = defaultdict(set)
        for link in links:
            self.add(link)

        self.gateways = gateways or set()
        self.clusters = self.compute_clusters()
        self.bridges = self.compute_bridges()

    def compute_clusters(self):
        clusters = list()
        for gateway in self.gateways:
            g = set([gateway])
            cluster = frozenset(self.get(gateway).union(g))
            clusters.append(cluster)
        return clusters

    def compute_bridges(self):
        bridges = set()
        for cluster in self.clusters:
            for n in cluster:
                childs = self.get(n) - cluster
                bridges.update(map(frozenset, zip(childs, itertools.repeat(n))))
        return bridges

    def add(self, link):
        k1, k2 = link
        self._map[k1].add(k2)
        self._map[k2].add(k1)

    def remove(self, link):
        link = frozenset(link)
        k1, k2 = link
        self._map[k1].remove(k2)
        self._map[k2].remove(k1)
        if link in self.bridges:
            self.bridges.remove(link)

    def get(self, key):
        return set(self._map.get(key))

    def get_links(self, key):
        links = set(map(frozenset, zip(self.get(key), itertools.repeat(key))))
        return links

    def __repr__(self):
        return repr(self._map)


N, L, E = [int(x) for x in input().split()]

pairs = [tuple(map(int, input().split())) for _ in range(L)]
gateways = set([int(input()) for _ in range(E)])
network = Network(pairs, gateways)

for _ in range(L):
    s = int(input())
    links = network.get_links(s)
    childs = network.get(s)
    gateways = childs.intersection(network.gateways)
    bridges = links.intersection(network.bridges)

    if gateways:
        g = gateways.pop()
        link = (g, s)
    elif bridges:
        link = bridges.pop()
    elif network.bridges:
        link = set(network.bridges).pop()
    else:
        link = links.pop()

    network.remove(link)
    print(*link)
