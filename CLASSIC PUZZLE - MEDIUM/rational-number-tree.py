import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class Rational:
    numerator: int
    denominator: int

    def __lt__(self, other):
        a = other.denominator * self.numerator
        b = other.numerator * self.denominator
        return a < b

    def __le__(self, other):
        return (self < other) or self == other

    def __gt__(self, other):
        return (self != other) and not (self < other) 
    
    def __ge__(self, other):
        return (self > other) or self == other

    @classmethod
    def smallest(cls):
        return cls(0, 1)

    @classmethod
    def largest(cls):
        return cls(1, 0)

    @classmethod
    def mediant(cls, seed1, seed2):
        numerator = seed1.numerator + seed2.numerator
        denominator = seed1.denominator + seed2.denominator
        return cls(numerator, denominator)
    
    @classmethod
    def from_lr_path(cls, path):
        left = cls.smallest()
        right = cls.largest()
        n = cls.mediant(left, right)
        for c in path:
            if c == 'L':
                right = n
            elif c == 'R':
                left = n
            n = cls.mediant(left, right)
        return n

    def to_path(self):
        left = self.smallest()
        right = self.largest()
        n = self.mediant(left, right)
        path = ''
        while n != self:
            if self > n:
                left = n
                path += 'R'
            elif self < n:
                right = n
                path += 'L'
            n = self.mediant(left, right)
        return path
    
    def to_string(self):
        return f'{self.numerator}/{self.denominator}'


class Factory:
    def from_fraction(self, string):
        print('FRACTION : ', string, file=sys.stderr)
        numerator, denominator = map(int, string.split('/', 1))
        return Rational(numerator, denominator).to_path()
    
    def from_lr_path(self, string):
        print('LR PATH : ', string, file=sys.stderr)
        return Rational.from_lr_path(string).to_string()

    def parse_string(self, string):
        print('PARSE : ', string, file=sys.stderr)
        if '/' in string:
            return self.from_fraction(string)
        else:
            return self.from_lr_path(string)


factory = Factory()

n = int(input())
for i in range(n):
    line = input()
    res = factory.parse_string(line)
    print(res)
