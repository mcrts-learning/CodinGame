import sys
import math
from collections import defaultdict

WINNING = [
    ('C', 'P'),
    ('P', 'R'),
    ('R', 'L'),
    ('L', 'S'),
    ('S', 'C'),
    ('C', 'L'),
    ('L', 'P'),
    ('P', 'S'),
    ('S', 'R'),
    ('R', 'C'),
]

def winner(a, b):
    if a[1] == b[1]:
        return a[0] < b[0]
    else:
        return (a[1], b[1]) in WINNING

n = int(input())
play = []
for i in range(n):
    numplayer, signplayer = input().split()
    play.append((int(numplayer), signplayer))

wins = defaultdict(list)
while len(play) > 1:
    new_play = []
    while play:
        p1 = play.pop(0)
        p2 = play.pop(0)
        if winner(p1, p2):
            new_play.append(p1)
            wins[p1].append(str(p2[0]))
        else:
            new_play.append(p2)
            wins[p2].append(str(p1[0]))
    play = new_play
p = play[0]
print(p[0])
print(' '.join(wins[p]))
