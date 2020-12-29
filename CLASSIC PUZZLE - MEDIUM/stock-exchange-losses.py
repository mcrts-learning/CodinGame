import sys
import math

n = int(input())
Valeur=[int(i) for i in input().split()]

print(Valeur,file=sys.stderr)
perte=0
for idx,i in enumerate(Valeur):
    if i>perte:
        perte=max(perte,max([i-int(j) for j in Valeur[idx:]]))
print(-perte)
