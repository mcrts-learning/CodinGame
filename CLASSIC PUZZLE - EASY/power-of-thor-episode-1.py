import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
# ---
# Hint: You can use the debug stream to print initialTX and initialTY, if Thor seems not follow your orders.

# light_x: the X position of the light of power
# light_y: the Y position of the light of power
# initial_tx: Thor's starting X position
# initial_ty: Thor's starting Y position
light_x, light_y, initial_tx, initial_ty = [int(i) for i in input().split()]
DIR_x = light_x-initial_tx
DIR_y = light_y-initial_ty

# game loop
while True:
    remaining_turns = int(input())  # The remaining amount of turns Thor can move. Do not remove this line.
    DIR=""

    if DIR_y<0:
        DIR = "N"
        DIR_y +=1
    elif DIR_y>0:
        DIR = "S"
        DIR_y +=-1
    if DIR_x<0:
        DIR += "W"
        DIR_x +=1
    elif DIR_x>0:
        DIR += "E"
        DIR_x +=-1

    print(DIR_x,file=sys.stderr)
    print(DIR_y,file=sys.stderr)
    print(DIR)
    # A single line providing the move to be made: N NE E SE S SW W or NW
