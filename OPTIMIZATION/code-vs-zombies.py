import sys
import math

# Save humans, destroy zombies!

def distance(a, b):
    xa, ya = a
    xb, yb = b
    return math.sqrt((xb - xa)**2 + (yb - ya)**2)


# game loop
while True:
    x, y = [int(i) for i in input().split()]
    human_count = int(input())
    humans = dict()
    for i in range(human_count):
        human_id, human_x, human_y = [int(j) for j in input().split()]
        humans[human_id] = (human_x, human_y)
    zombie_count = int(input())
    zombies = dict()
    for i in range(zombie_count):
        zombie_id, zombie_x, zombie_y, zombie_xnext, zombie_ynext = [int(j) for j in input().split()]
        zombies[zombie_id] = (zombie_x, zombie_y)

    human_center = (sum(map(lambda x: x[0], humans.values()))/human_count, sum(map(lambda x: x[1], humans.values()))/human_count)
    distances = list(map(lambda x: distance(human_center, x), humans.values()))
    if list(filter(lambda x: x<=2000, distances)):
        target = human_center
    else:
        target = list(humans.values())[0]
    print(distances, file=sys.stderr, flush=True)
    print('{} {}'.format(round(target[0]), round(target[1])))
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    # Your destination coordinates
