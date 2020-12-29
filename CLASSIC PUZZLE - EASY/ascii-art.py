import sys
import math
import re

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

l = int(input())
h = int(input())
t = input()
nl=len(t)
alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ?'
L=len(alphabet)

ASCII=[input() for x in range(h)]

t=re.sub(r'[^a-zA-Z?]', "?", t)
print(t,file=sys.stderr)

LETTRE=""

for i in range(h):
    for lettre in t:
        for index,a in enumerate(alphabet):
            if lettre.casefold()==a.casefold():
                LETTRE += ASCII[i][index*l:(index+1)*l]

for i in range (h):
    print(LETTRE[i*l*nl:(i+1)*l*nl])

# Write an action using print
# To debug: print("Debug messages...", file=sys.stderr)
