import sys

from typing import Iterable, NamedTuple
from dataclasses import dataclass, field
from collections import Counter

import itertools as it
import re


@dataclass(frozen=True)
class Outcome:
    value: int
    probability: float


@dataclass
class Entity:
    values: Iterable[int]

    def __add__(self, other):
        values = list(
            map(
                lambda t: int(t[0] + t[1]),
                it.product(self.values, other.values)
            )
        )
        return Entity(values)

    def __sub__(self, other):
        values = list(
            map(
                lambda t: int(t[0] - t[1]),
                it.product(self.values, other.values)
            )
        )
        return Entity(values)

    def __gt__(self, other):
        values = list(
            map(
                lambda t: int(t[0] > t[1]),
                it.product(self.values, other.values)
            )
        )
        return Entity(values)

    def __mul__(self, other):
        values = list(
            map(
                lambda t: int(t[0] * t[1]),
                it.product(self.values, other.values)
            )
        )
        return Entity(values)
    
    def outcomes(self):
        counter = Counter(self.values)
        outcomes = []
        total = sum(counter.values())
        for value, count in sorted(counter.items()):
            outcomes.append(Outcome(value, count / total))
        return outcomes


@dataclass
class Number(Entity):
    value: int
    values: Iterable[int] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.values = [self.value]

@dataclass
class Dice(Entity):
    n: int
    values: Iterable[int] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.values = list(range(1, self.n + 1))


class Token(NamedTuple):
    type: str
    value: str


def tokenize(expr):
    token_specification = [
        ('NUMBER',   r'\d+'),          # Integer or decimal number
        ('DICE',   r'd\d+'),           # Dice
        ('OP',       r'[+\-*/()><]'),      # Arithmetic operators
        ('SKIP',     r'[ \t]+'),       # Skip over spaces and tabs
        ('MISMATCH', r'.'),            # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    for mo in re.finditer(tok_regex, expr):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'NUMBER':
            value = int(value)
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line 1')
        yield Token(kind, value)
    pass

def serialize(tokens):
    sequence = []
    for token in tokens:
        if token.type == 'NUMBER':
            sequence.append(f"Number({token.value})")
        elif token.type == 'DICE':
            sequence.append(f"Dice({token.value[1:]})")
        else:
            sequence.append(token.value)
    return ''.join(sequence)


expr = input()
print(expr, file=sys.stderr)
for o in eval(serialize((tokenize(expr)))).outcomes():
    print(f"{o.value} {100 * o.probability:.2f}")
