import sys
import math


n = int(input())
c = int(input())
b=[]
for i in range(n):
    b.append(int(input()))

print("prix=",c,"budget=",b,file=sys.stderr)


if sum(b)<c:
    print("IMPOSSIBLE")
    exit()

budget=[]
for i in b:
    if i<c/n:
        budget.append(i)
    else:
        budget.append(int(c/n))

while sum(budget)!=c:
    for i in range(n):
        if (budget[i]+1)<=b[i]:
            budget[i]+=1

        if sum(budget)==c:
            break

budget.sort()
for i in budget:
    print(i)

# Write an action using print
# To debug: print("Debug messages...", file=sys.stderr)
