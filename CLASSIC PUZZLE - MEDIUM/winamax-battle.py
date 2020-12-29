import sys
import math
from collections import deque


n = int(input())  # the number of cards for player 1
P1=deque()
for i in range(n):
    INPUT=input()[0:-1]
    if INPUT=="J":
        INPUT="11"
    elif INPUT=="Q":
        INPUT="12"
    elif INPUT=="K":
        INPUT="13"
    elif INPUT=="A":
        INPUT="14"
    P1.append(int(INPUT))  # the n cards of player 1
m = int(input())  # the number of cards for player 2
P2=deque()
for i in range(m):
    INPUT=input()[0:-1]
    if INPUT=="J":
        INPUT="11"
    elif INPUT=="Q":
        INPUT="12"
    elif INPUT=="K":
        INPUT="13"
    elif INPUT=="A":
        INPUT="14"
    P2.append(int(INPUT))  # the m cards of player 2

N=0
while len(P1)>0 and len(P2)>0:
    N+=1
    B1=[]
    B2=[]
    B1.append(P1.popleft())
    B2.append(P2.popleft())
    while B1[-1]==B2[-1]:
        print("bataille",file=sys.stderr)
        if len(P1)<4 or len(P2)<4:
            print("PAT")
            exit()
        B1.append(P1.popleft())
        B2.append(P2.popleft())
        B1.append(P1.popleft())
        B2.append(P2.popleft())
        B1.append(P1.popleft())
        B2.append(P2.popleft())
        B1.append(P1.popleft())
        B2.append(P2.popleft())

    if B1[-1]>B2[-1]:
        P1.extend(B1)
        P1.extend(B2)
    else:
        P2.extend(B1)
        P2.extend(B2)


if len(P1)>0:
    print(1,N)
else:
    print(2,N)
