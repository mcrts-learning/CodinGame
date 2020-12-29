import sys
import random
import numpy as np

from collections import namedtuple

def log(*args, **kwargs):
    print(*args, flush=True, file=sys.stderr, **kwargs)


class Delta(namedtuple("Recipe", ['array'])):
    def __new__(cls, array):
        return super().__new__(cls, np.array(array))

    @staticmethod
    def ceil(array, value=0):
        return np.array([x if x < value else value for x in array])

    def uniform_distance(self, other):
        delta = self.array + other.array
        delta = self.ceil(delta)
        return abs(sum(delta))

    def weight_distance(self, other):
        delta = self.array + other.array
        delta = self.ceil(delta)
        r = np.dot(delta, [1, 1.5, 2.5, 3.5])
        return abs(r)


class Player:
    def __init__(self, inventory, score):
        self.inventory = inventory
        self.score = score

    def can_do(self, action):
        result = self.inventory.array + action.delta.array
        return self.inventory.uniform_distance(action.delta) <= 0 and sum(result) <= 10

    def uniform_distance(self, action):
        return self.inventory.uniform_distance(action.delta)

    def weight_distance(self, action):
        return self.inventory.weight_distance(action.delta)

    def update_inventory(self, action):
        delta = Delta(self.inventory.array + action.delta.array)
        return type(self)(delta, self.score)

class Recipe(namedtuple("Recipe", ['action_id', 'delta', 'price'])):
    pass

class Spell(namedtuple("Spell", ['action_id', 'delta', 'castable'])):
    pass

class Strategy:
    @staticmethod
    def strategy1(player, recipes, spells):
        def increase(player, recipe, spell):
            d = player.uniform_distance(recipe)
            p = player.update_inventory(spell)
            return d - p.uniform_distance(recipe)
        r = min(recipes, key=player.uniform_distance)
        good_spells = list(filter(lambda s: increase(player, r, s) >= 0, spells))
        if good_spells:
            s = max(good_spells, key=lambda s: increase(player, r, s))
            action = "CAST {}".format(s.action_id)
        else:
            action = "REST"
        return action

    @staticmethod
    def strategy2(player, recipes, spells):
        def increase(player, recipe, spell):
            d = player.weight_distance(recipe)
            p = player.update_inventory(spell)
            return d - p.weight_distance(recipe)
        r = min(recipes, key=player.weight_distance)
        good_spells = list(filter(lambda s: increase(player, r, s) >= 0, spells))
        if good_spells:
            s = max(good_spells, key=lambda s: increase(player, r, s))
            action = "CAST {}".format(s.action_id)
        else:
            action = "REST"
        return action

# game loop
while True:
    action_count = int(input())  # the number of spells and recipes in play
    recipes = []
    spells = []
    for i in range(action_count):
        # action_type: in the first league: BREW, CAST, OPPONENT_CAST; LATER : LEARN, BREW
        # tome_index: in the first two leagues: always 0; later: the index in the tome if this is a tome spell, equal to the read-ahead tax; For brews, this is the value of the current urgency bonus
        # tax_count: in the first two leagues: always 0; later: the amount of taxed tier-0 ingredients you gain from learning this spell; For brews, this is how many times you can still gain an urgency bonus
        # repeatable: for the first two leagues: always 0; later: 1 if this is a repeatable player spell
        action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count, castable, repeatable = input().split()
        action_id = int(action_id)
        delta = Delta([int(delta_0), int(delta_1), int(delta_2), int(delta_3)])
        price = int(price)
        # tome_index = int(tome_index)
        # tax_count = int(tax_count)
        castable = castable != "0"
        # repeatable = repeatable != "0"

        if action_type == "BREW":
            recipes.append(Recipe(action_id, delta, price))
        if action_type == "CAST" and castable:
            spells.append(Spell(action_id, delta, castable))

    inv_0, inv_1, inv_2, inv_3, score = [int(j) for j in input().split()]
    player1 = Player(Delta([inv_0, inv_1, inv_2, inv_3]), score)

    inv_0, inv_1, inv_2, inv_3, score = [int(j) for j in input().split()]
    player2 = Player(Delta([inv_0, inv_1, inv_2, inv_3]), score)

    craftables = list(filter(player1.can_do, recipes))
    castables = list(filter(player1.can_do, spells))
    sorted_recipes = sorted(craftables, key=lambda x: x.price, reverse=True)
    log(sorted_recipes)
    if sorted_recipes:
        r = sorted_recipes[0]
        action = "BREW {}".format(r.action_id)
    else:
        action = Strategy.strategy2(player1, recipes, castables)
    print(action)
