import sys
import math


def count_digit(p1, p2):
    l1 = len(str(p1))
    l2 = len(str(p2))
    count = 0
    for i in range(l1, l2+1):
        if i == l1:
            st = p1
        else:
            st = 10**(i-1)
        if i == l2:
            ed = p2
        else:
            ed = 10**i - 1
        count += (ed - st + 1) * i
    return count

def dichotomic_search(p1, p2, target):
    pin = p1
    pout = p2
    p = p1
    while pout - pin > 1:
        p = (pin + pout) // 2
        count = count_digit(p1, p)
        if count == target:
            return p
        elif count < target:
            pin = p
        else:
            pout = p
            p = p - 1
    return p


n = int(input())
for i in range(n):
    st, ed = [int(j) for j in input().split()]
    target_count = count_digit(st, ed) // 2
    page = dichotomic_search(st, ed, target_count)
    print(page)
