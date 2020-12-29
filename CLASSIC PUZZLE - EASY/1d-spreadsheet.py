import sys
import math

OPERAND = {
    'VALUE': lambda x, y: x,
    'ADD': lambda x, y: int(x) + int(y),
    'SUB': lambda x, y: int(x) - int(y),
    'MULT': lambda x, y: int(x) * int(y),
}

def evaluate(arg, sheet):
    if arg.startswith('$'):
        res = eval_cell(int(arg[1:]), sheet)
    else:
        res = arg
    return res

def eval_cell(index, sheet):
    cell = sheet[index]
    operand, arg1, arg2 = cell
    arg1 = evaluate(arg1, sheet)
    arg2 = evaluate(arg2, sheet)
    result = OPERAND[operand](arg1, arg2)
    sheet[index] = ('VALUE', str(result), '_')
    return result

n = int(input())

sheet = []
for i in range(n):
    sheet.append(tuple(input().split()))

for i in range(n):
    print(eval_cell(i, sheet))
