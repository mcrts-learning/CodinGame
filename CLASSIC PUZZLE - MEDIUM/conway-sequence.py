import sys
import math

def Sequence(C):
    New=[]
    idx=0


    for i in C:

        if len(New)<2:

            New.append(1)
            New.append(int(i))
        else:
            if New[-1]==i:
                New[-2]+=1
            else:
                New.append(1)
                New.append(int(i))

    return New

r = int(input())
l = int(input())

CONWAY=[]
CONWAY.append([r])

for i in range(l-1):

    CONWAY.append(Sequence(CONWAY[i]))


ANS=[str(i) for i in CONWAY[-1]]

ANS=' '.join(ANS)
print(ANS)
