import sys
import math
import string

def reverse(dictionary):
    keys = dictionary.keys()
    values = dictionary.values()
    return dict(zip(values, keys))

def Caesar(shift=0):
    shift = shift % 26
    alpha_input = string.ascii_uppercase
    alpha_output = string.ascii_uppercase[shift:] + string.ascii_uppercase[:shift]
    return dict(zip(alpha_input, alpha_output))

def Rotor(s):
    alpha_input = string.ascii_uppercase
    return dict(zip(alpha_input, s))

def caesar_shift(string, shift=0, inverse=False):
    new_string = ''
    for s in string:
        if inverse:
            new_string += reverse(Caesar(shift))[s]
        else:
            new_string += Caesar(shift)[s]
        shift += 1
    return new_string

def encode(s0, shift, r1, r2, r3):
    s1 = caesar_shift(s0, shift)
    s2 = ''.join(map(Rotor(r1).get, s1))
    s3 = ''.join(map(Rotor(r2).get, s2))
    s4 = ''.join(map(Rotor(r3).get, s3))
    return s4

def decode(s0, shift, r1, r2, r3):
    s1 = ''.join(map(reverse(Rotor(r3)).get, s0))
    s2 = ''.join(map(reverse(Rotor(r2)).get, s1))
    s3 = ''.join(map(reverse(Rotor(r1)).get, s2))
    s4 = caesar_shift(s3, shift, inverse=True)
    return s4

operation = input()
shift = int(input())
r1 = input()
r2 = input()
r3 = input()
message = input()

if operation == 'ENCODE':
    print(encode(message, shift, r1, r2, r3))
else:
    print(decode(message, shift, r1, r2, r3))
