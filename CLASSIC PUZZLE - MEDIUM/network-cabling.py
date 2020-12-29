import sys
import math

n = int(input())
POS=[]
for i in range(n):
    POS.append([int(j) for j in input().split()])
POS.sort(key=lambda x: x[1])

print(POS,file=sys.stderr)
idx=int((n-1)/2)
Y=POS[idx][1]

POS.sort()
L=POS[n-1][0]-POS[0][0]

for i in range(n):
    L+=abs(Y-POS[i][1])


print(int(L))
