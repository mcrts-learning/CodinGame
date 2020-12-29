import sys
import math

def log(msg, *args, **kwargs):
    print(msg, *args, file=sys.stderr, flush=True, **kwargs)

OPEN_TOKENS = '([{'
CLOSE_TOKENS = ')]}'
MAP = {
    '(': ')',
    ')': '(',
    '[': ']',
    ']': '[',
    '{': '}',
    '}': '{',
}

def validate_char(closure, c):
    if c in OPEN_TOKENS:
        return True
    elif c in CLOSE_TOKENS and closure and c == MAP[closure[-1]]:
        return True
    return False

def process_closure(closure, c):
    closure = list(closure)
    if c in OPEN_TOKENS:
        closure.append(c)
    elif c in CLOSE_TOKENS:
        closure.pop()
    return closure

def validate(expression):
    closure = []
    for c in expression:
        if validate_char(closure, c):
            closure = process_closure(closure, c)
        else:
            return False
    if closure:
        return False
    return True

expression = ''.join(filter(lambda x: x in MAP.keys(), input()))
log(expression)
if validate(expression):
    print('true')
else:
    print('false')
