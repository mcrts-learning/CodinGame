import sys
import math

class Proportional:
    def __init__(self, gain, target):
        self.gain = gain
        self.target = target

    def error(self, value):
        x, y = value
        tx, ty = self.target
        return math.sqrt((tx-x)**2 + (ty-y)**2)

    def input(self, value):
        return self.gain * self.error(value)

thrust_controller = Proportional(0.5, None)

def floor(signal):
    signal = int(signal)
    if signal >= 100:
        signal = 100
    elif signal <= 20:
        signal = 20
    return signal




# game loop
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]

    thrust_controller.target = (next_checkpoint_x, next_checkpoint_y)
    signal = floor(thrust_controller.input((x, y)))
    print("{} {} {}".format(next_checkpoint_x, next_checkpoint_y, signal))
