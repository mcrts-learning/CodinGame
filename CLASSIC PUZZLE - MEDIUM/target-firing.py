import sys
import math

class Ship:
    def __init__(self, category, hp, armor, damage):
        self.category = category
        self.hp = hp
        self.armor = armor
        self.damage = damage
        self.turns = self.turns_to_kill()

    def turns_to_kill(self):
        damage = 10 if self.category == 'CRUISER' else 20
        turns = math.ceil(self.hp / max([damage - self.armor, 1]))
        return turns

    def priority(self):
        return self.turns / self.damage

    def __repr__(self):
        return '<{} : {} {} {} {}>'.format(self.category, self.hp, self.armor, self.damage, self.turns)

n = int(input())
ships = list()
for i in range(n):
    category, hp, armor, damage = input().split()
    hp = int(hp)
    armor = int(armor)
    damage = int(damage)
    ships.append(Ship(category, hp, armor, damage))

ships.sort(key=lambda x: x.priority())
print(ships, file=sys.stderr, flush=True)

shield = 5000
while ships and shield >= 0:
    ship = ships[0]
    damage = ship.turns * sum(map(lambda x: x.damage, ships))
    shield -= damage
    ships.pop(0)

result = shield if shield >= 0 else 'FLEE'
print(result)
