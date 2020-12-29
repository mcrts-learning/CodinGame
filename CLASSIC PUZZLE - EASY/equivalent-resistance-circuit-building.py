import sys
import math
from collections import namedtuple

class Resistance(namedtuple('Resistance', 'r')):
    def eval(self):
        return self.r

class SerieResistance(namedtuple('CompositeResistance', 'resistances')):
    def eval(self):
        resistances = map(lambda x: x.eval(), self.resistances)
        return sum(resistances)

class ParallelResistance(namedtuple('CompositeResistance', 'resistances')):
    def eval(self):
        resistances = map(lambda x: x.eval(), self.resistances)
        inv_resistances = map(lambda x: x**(-1), resistances)
        r_inv = sum(inv_resistances)
        return r_inv**(-1)

def parse(circuit, resistances_dict, p=0):
    resistances = []
    while p < len(circuit):
        c = circuit[p]
        if c == '(':
            resists, p = parse(circuit, resistances_dict, p+1)
            resistances.append(SerieResistance(resists))
        if c == '[':
            resists, p = parse(circuit, resistances_dict, p+1)
            resistances.append(ParallelResistance(resists))
        elif c in (')', ']'):
            break
        elif c in resistances_dict:
            resistances.append(Resistance(resistances_dict[c]))
        p += 1
    return resistances, p


resistances = dict()
n = int(input())
for i in range(n):
    name, r = input().split()
    resistances[name] = int(r)
circuit = input()

resistances, p = parse(circuit.split(), resistances)
r = resistances[0]

print(round(float(r.eval()), 1))
