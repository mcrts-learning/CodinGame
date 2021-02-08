import sys
from collections import namedtuple, Counter

def log(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)

WAIT = ("WAIT",)
ELEVATOR = ("ELEVATOR",)
BLOCK = ("BLOCK",)
# WAIT_BLOCK = ("WAIT", "BLOCK", "WAIT", "WAIT")
# (WAIT_BLOCK, Node(node.x - 2*node.direction, node.y, -node.direction))

class Node(namedtuple('Node', ['x', 'y', 'direction'])):
    pass

class State:
    def __init__(self, height, width, time, n_clone, n_elevator, exit):
        self.height = height
        self.width = width
        self.time = time
        self.n_clone = n_clone
        self.n_elevator = n_elevator
        self.elevators = [[False for _ in range(width)] for _ in range(height)]
        self.blocks = [[False for _ in range(width)] for _ in range(height)]
        self.exit = exit
        self.start = None

    def add_elevator(self, x, y):
        self.elevators[y][x] = True

    def add_block(self, x, y):
        self.blocks[y][x] = True

    def set_start(self, x, y):
        self.start = (x, y)

    def get_adjacents(self, node):
        adjacents = []
        if 0 <= node.x + node.direction <= self.width:
            adjacents.append((WAIT, Node(node.x + node.direction, node.y, node.direction)))
        if 0 <= node.y < self.height - 1:
            adjacents.append((ELEVATOR, Node(node.x, node.y + 1, node.direction)))
        adjacents.append((BLOCK, Node(node.x - node.direction, node.y, -node.direction)))
        return adjacents

    def simulate(self, node, actions):
        time, n_clone, n_elevator = cost(actions)
        if time > self.time:
            return node, "ERROR | too long"
        elif n_clone > self.n_clone:
            return node, "ERROR | not enough clone"
        elif n_elevator > self.n_elevator:
            return node, "ERROR | not enough elevator"
        else:
            for action in actions:
                if action == ELEVATOR[0]:
                    node = Node(node.x, node.y + 1, node.direction)
                elif action == BLOCK[0]:
                    node = Node(node.x - node.direction, node.y, -node.direction)
                else:
                    if 0 <= node.x + node.direction <= self.width:
                        node = Node(node.x + node.direction, node.y, node.direction)
                    else:
                        return node, "ERROR | clone died"
                if (node.x, node.y) == self.exit:
                    return node, "OK"
        return node, "OK"

def cost(actions):
    counter = Counter(actions)
    time = counter['WAIT'] + 4*counter['ELEVATOR'] + 4*counter['BLOCK']
    clone = counter['ELEVATOR'] + counter['BLOCK']
    elevator = counter['ELEVATOR']
    return time, clone, elevator


height, width, time, exit_y, exit_x, n_clone, n_elevator, n_starting_elevator = [int(i) for i in input().split()]
state = State(height, width, time, n_clone, n_elevator, (exit_x, exit_y))
for i in range(n_starting_elevator):
    y, x = [int(j) for j in input().split()]
    state.add_elevator(x, y)


SCENARIO1 = ELEVATOR + 3*WAIT + 9*WAIT
SCENARIO2 = ELEVATOR + 3*WAIT + BLOCK
SCENARIO3 = (ELEVATOR + 3*WAIT)*5
SCENARIO4 = BLOCK + 9*WAIT + ELEVATOR + 3*WAIT + BLOCK + 6*WAIT + ELEVATOR + 3*WAIT + BLOCK
SCENARIO5 = 3*WAIT + ELEVATOR + 6*WAIT + ELEVATOR + 3*WAIT + ELEVATOR
SCENARIO6 = 4*WAIT + BLOCK + 6*WAIT + (4*WAIT + BLOCK)*5
SCENARIO7 = 4*WAIT + BLOCK + 10*WAIT + ELEVATOR + 4*WAIT + BLOCK + 4*WAIT + BLOCK + 4*WAIT + BLOCK

ACTION_SEQUENCE = list(SCENARIO1)


log(height, width, time, n_clone, n_elevator)
log(cost(ACTION_SEQUENCE))
# game loop
while True:
    inputs = input().split()
    y = int(inputs[0])  # floor of the leading clone
    x = int(inputs[1])  # position of the leading clone on its floor
    direction = 1 if inputs[2] == 'RIGHT' else -1  # direction of the leading clone: LEFT or RIGHT
    node = Node(x, y, direction)

    log(state.simulate(node, ACTION_SEQUENCE))
    if ACTION_SEQUENCE:
        action = ACTION_SEQUENCE.pop(0)
    else:
        action = "END"

    log(ACTION_SEQUENCE)


    print(action)
