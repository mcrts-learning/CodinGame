import sys
import math

def log(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)


class Interpreter:
    def __init__(self, size):
        self.array = [0] * size
        self.pointer = 0

    def interpret(self, source, inputs):
        self.static_check(source)
        position = 0
        source_len = len(source)
        array_size = len(self.array)
        outputs = []
        stack = []
        while position < source_len:
            code = source[position]
            if code == '>':
                self.pointer += 1
            elif code == '<':
                self.pointer -= 1
            elif code == '+':
                self.array[self.pointer] += 1
            elif code == '-':
                self.array[self.pointer] -= 1
            elif code == '.':
                outputs.append(chr(self.array[self.pointer]))
            elif code == ',':
                self.array[self.pointer] = inputs.pop(0)
            elif code == '[' and not self.array[self.pointer]:
                opened = 1
                while opened:
                    position += 1
                    if source[position] == '[': opened += 1
                    elif source[position] == ']': opened -= 1
            elif code == '[' and self.array[self.pointer]:
                stack.append(position)
            elif code == ']' and not self.array[self.pointer]:
                stack.pop()
            elif code == ']' and self.array[self.pointer]:
                position = stack[-1]
            else: pass
            position += 1

            if not (0 <= self.pointer < array_size):
                raise ValueError('POINTER OUT OF BOUNDS')
            if not (0 <= self.array[self.pointer] <= 255):
                raise ValueError('INCORRECT VALUE')
        return outputs

    @staticmethod
    def static_check(source):
        stack = 0
        for code in source:
            if code == '[': stack += 1
            elif code == ']': stack -= 1

            if stack < 0:
                raise ValueError('SYNTAX ERROR')
        if stack > 0:
            raise ValueError('SYNTAX ERROR')



l, s, i = [int(i) for i in input().split()]
source = '\n'.join([input() for _ in range(l)])
inputs = [int(input()) for _ in range(i)]
interpreter = Interpreter(s)

try:
    outputs = interpreter.interpret(source, inputs)
except ValueError as e:
    print(e)
else:
    print(''.join(outputs))
