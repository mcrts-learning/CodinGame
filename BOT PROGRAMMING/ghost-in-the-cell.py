import sys
import math
import random
from collections import namedtuple

RANDOM = random.Random(1)

class Edge(namedtuple('Edge', ['v1', 'v2', 'cost'])):
    pass

class UndirectedGraph(namedtuple('UndirectedGraph', ['vertices', 'edges'])):
    def __init__(self, vertices, edges):
        pass

    @classmethod
    def construct(cls, vertex, edges):
        return cls(vertex, edges)

    @classmethod
    def empty(cls):
        return cls(frozenset(), frozenset())

    def add_vertex(self, vertex):
        return self.construct(self.vertices | set([vertex]), self.edges)

    def add_edge(self, v1, v2, cost):
        g = self.add_vertex(v1)
        g = g.add_vertex(v2)
        e1 = Edge(v1, v2, cost)
        e2 = Edge(v2, v1, cost)
        return g.construct(g.vertices, g.edges | set([e1, e2]))

    def get_all_edges(self, v):
        return filter(lambda e: e.v1 == v, self.edges)

    def get_all_neighbors(self, v):
        return map(lambda e: e.v2, self.get_all_edges(v))

class Factory(namedtuple('Factory', ['id', 'owner', 'population', 'production'])):
    @classmethod
    def construct(cls, fid, owner, population, production):
        return cls(int(fid), int(owner), int(population), int(production))

class State:
    def __init__(self, graph):
        self.graph = graph
        self.factory_properties = dict()

    def set_factory(self, factory):
        if factory.id in self.graph.vertices:
            self.factory_properties[factory.id] = factory
        else:
            raise ValueError(f"<Factory id={factory.id}> not in graph.")

    def get_all_neighbors(self, factory):
        if isinstance(factory, Factory):
            fid = factory.id
        else :
            fid = int(factory)
        if fid not in self.graph.vertices:
            raise ValueError(f"<Factory id={factory.id}> not in graph.")

        edges =  self.graph.get_all_edges(fid)
        factories = map(lambda e: (e.cost, self.factory_properties.get(e.v2)), edges)
        return sorted(factories)

    def get_owned_factories(self):
        factories = filter(lambda f: f.owner == 1, self.factory_properties.values())
        return list(factories)

class AttackStrategy:
    @staticmethod
    def strategy1(state, factory):
        def sort_key(item):
            distance, f = item
            return (f.owner != 0, distance, -f.production, f.population)
        factories = state.get_all_neighbors(factory)
        factories = filter(lambda x: x[1].owner != 1, factories)
        factories = filter(lambda x: x[1].production > 0, factories)
        factories = sorted(factories, key=sort_key)
        distance, target = factories[0]
        return ["MOVE {} {} {}".format(factory.id, target.id, factory.population)]

    @staticmethod
    def strategy2(state, factory):
        def sort_key(item):
            distance, f = item
            return (f.owner != 0, distance, -f.production, f.population)
        factories = state.get_all_neighbors(factory)
        factories = filter(lambda x: x[1].owner != 1, factories)
        factories = filter(lambda x: x[1].production > 0, factories)
        factories = sorted(factories, key=sort_key)

        population = factory.population
        actions = []
        while population > 0 and factories:
            distance, target = factories.pop(0)
            pop = min(target.population + 5, population)
            population -= pop
            actions.append("MOVE {} {} {}".format(factory.id, target.id, pop))
        return actions

    @staticmethod
    def strategy3(state, factory):
        def sort_key(item):
            distance, f = item
            return (distance, -f.production, f.population)
        factories = state.get_all_neighbors(factory)
        factories = filter(lambda x: x[1].owner != 1, factories)
        factories = filter(lambda x: x[1].production > 0, factories)
        factories = sorted(factories, key=sort_key)

        population = factory.population
        actions = []
        while population > 0 and factories:
            distance, target = factories.pop(0)
            pop = min(target.population + 3, population)
            population -= pop
            actions.append("MOVE {} {} {}".format(factory.id, target.id, pop))
        return actions

class BombingStrategy:
    @staticmethod
    def first_bombing(state):
        factories = state.factory_properties.values()
        factories = filter(lambda x: x.owner == -1, factories)
        factories = sorted(factories, key=lambda x: (x.production, x.population), reverse=True)
        target = factories[0]
        distance, source = next(filter(lambda x: x[1].owner == 1, state.get_all_neighbors(target)))
        return ["BOMB {} {}".format(source.id, target.id)]

factory_count = int(input())
link_count = int(input())
graph = UndirectedGraph.empty()
for i in range(link_count):
    factory_1, factory_2, distance = [int(j) for j in input().split()]
    graph = graph.add_edge(factory_1, factory_2, distance)


# game loop
state = State(graph)
turn = 0
while True:
    turn += 1
    factories = list()
    entity_count = int(input())  # the number of entities (e.g. factories and troops)

    for i in range(entity_count):
        entity_id, entity_type, arg1, arg2, arg3, arg4, arg5 = input().split()
        if entity_type == 'FACTORY':
            state.set_factory(Factory.construct(entity_id, arg1, arg2, arg3))

    actions = []
    if turn == 2:
        actions.extend(BombingStrategy.first_bombing(state))
    for f in state.get_owned_factories():
        actions.extend(AttackStrategy.strategy3(state, f))
    if not actions:
        actions.append("WAIT")
    print(';'.join(actions + ["MSG pour mon pere le roi !"]))
