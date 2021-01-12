import sys
import itertools as it
from collections import namedtuple

def log(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)

SPEED = ("SPEED",)
SLOW = ("SLOW",)
UP = ("UP",)
DOWN = ("DOWN",)
JUMP = ("JUMP",)
WAIT = ("WAIT",)

SEQUENCES = list(it.product(SPEED + SLOW + UP + DOWN + JUMP, repeat=5))

class State(namedtuple('State', 'starting_bikes, minimum_bikes, length, lanes, speed, bikes')):
    @classmethod
    def construct(cls, starting_bikes, minimum_bikes, lanes, speed=1, bikes=frozenset()):
        starting_bikes = int(starting_bikes)
        minimum_bikes = int(minimum_bikes)
        length = len(lanes[0])
        lanes = tuple(lanes)
        speed = int(speed)
        bikes = frozenset(bikes)
        return cls(starting_bikes, minimum_bikes, length, lanes, speed, bikes)

    def __str__(self):
        bikes = ['x={}, y={}, active={}'.format(x, y, bool(z)) for (x, y, z) in self.bikes]
        track = list('.'*self.length)
        x = max([x[0] for x in self.bikes])
        x = min([len(track) - 1, x])
        track[x] = 'X'
        track = ''.join(track)
        s = '\n'.join([
            "starting bikes={}, minimum bikes={}",
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
            "speed={}",
            "{}"
        ]).format(
            self.starting_bikes,
            self.minimum_bikes,
            self.lanes[0],
            self.lanes[1],
            self.lanes[2],
            self.lanes[3],
            track,
            self.speed,
            '\n'.join(bikes)
        )
        return s

    def update_speed(self, speed):
        speed = int(speed)
        s = self.starting_bikes
        m = self.minimum_bikes
        return self.construct(s, m, self.lanes, speed, self.bikes)

    def update_bikes(self, bikes):
        bikes = tuple(bikes)
        s = self.starting_bikes
        m = self.minimum_bikes
        return self.construct(s, m, self.lanes, self.speed, bikes)
    
    def load(self, speed, bikes):
        return self.update_speed(speed).update_bikes(bikes)
  
    def active_bikes(self):
        return frozenset([b for b in self.bikes if b[2] == True])

    def action_wait(self):
        s = self.speed
        bikes = [(x + s, y, a and not '0' in self.lanes[y][x+1: x+s+1]) for x, y, a in self.active_bikes()]
        state = self.update_bikes(bikes)
        if len(state.active_bikes()) < state.minimum_bikes:
            status = "INVALID | insuficent bikes"
        else:
            status = "OK"
        return state, status

    def action_jump(self):
        s = self.speed
        bikes = [(x + s, y, a and (x + s >= self.length or self.lanes[y][x + s] != '0')) for x, y, a in self.active_bikes()]
        state = self.update_bikes(bikes)
        if len(state.active_bikes()) < state.minimum_bikes:
            status = "INVALID | insuficent bikes"
        else:
            status = "OK"
        return state, status
    
    def action_speed(self):
        state, status = self.update_speed(self.speed + 1).action_wait()
        return state, status
    
    def action_slow(self):
        if self.speed == 1:
            status = "INVALID | invalid SLOW command"
            state, _ = self.action_wait()
        else:
            state, status = self.update_speed(self.speed - 1).action_wait()
        return state, status
    
    def action_up(self):
        if any(map(lambda x: x[1] == 0, self.active_bikes())):
            status = "INVALID | invalid UP command"
            state, _ = self.action_wait()
        else:
            bikes = []
            for x0, y0, _ in self.active_bikes():
                x1 = x0 + self.speed
                y1 = y0 - 1
                a1 = not ('0' in self.lanes[y0][x0+1: x1] + self.lanes[y1][x0+1: x1+1])
                bikes.append((x1, y1, a1))
            state = self.update_bikes(bikes)
            if len(state.active_bikes()) < state.minimum_bikes:
                status = "INVALID | insuficent bikes"
            else:
                status = "OK"
        return state, status
    
    def action_down(self):
        if any(map(lambda x: x[1] == 3, self.active_bikes())):
            status = "INVALID | invalid DOWN command"
            state, _ = self.action_wait()
        else:
            bikes = []
            for x0, y0, _ in self.active_bikes():
                x1 = x0 + self.speed
                y1 = y0 + 1
                a1 = not ('0' in self.lanes[y0][x0+1: x1] + self.lanes[y1][x0+1: x1+1])
                bikes.append((x1, y1, a1))
            state = self.update_bikes(bikes)
            if len(state.active_bikes()) < state.minimum_bikes:
                status = "INVALID | insuficent bikes"
            else:
                status = "OK"
        return state, status
    
    def simulate_action(self, action):
        if action == "JUMP":
            return self.action_jump()
        elif action == "SPEED":
            return self.action_speed()
        elif action == "SLOW":
            return self.action_slow()
        elif action == "UP":
            return self.action_up()
        elif action == "DOWN":
            return self.action_down()
        else: 
            return self.action_wait()

    def simulate_sequence(self, sequence):
        if not sequence:
            sequence = WAIT
        state = self
        for action in sequence:
            state, status = state.simulate_action(action)
            if status != "OK":
                break
        return state, status

def get_score(state, status):
    if not status == "OK":
        sc1 = -1
    elif state.speed == 0:
        sc1 = 0
    else:
        sc1 = len(state.active_bikes())
    return sc1, state.speed

def strategy(state):
    results = []
    for sequence in SEQUENCES:
        nextstate, status = state.simulate_sequence(sequence)
        score = get_score(nextstate, status)
        results.append((score, sequence, nextstate))
    return max(results, key=lambda x: x[0])

m = int(input())
v = int(input())
lanes = [input() for _ in range(4)]

state = State.construct(m, v, lanes)
while True:
    s = int(input())
    bikes = [tuple(map(int, input().split())) for _ in range(m)]
    state = state.update_speed(s).update_bikes(bikes)
    score, sequence, nextstate = strategy(state)
    action = sequence[0]
    print(action)
    
