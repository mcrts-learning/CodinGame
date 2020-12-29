import sys
import math

BASE=20
Sym2Base={}
Base2Sym={}

l, h = [int(i) for i in input().split()]
for i in range(h):
    numeral=input()
    for j in range(BASE):
        idx=l*j
        STR=numeral[idx:idx+l]
        if j in Base2Sym:
            Base2Sym[j]+=[STR]
        else:
            Base2Sym[j]=[STR]

for key,value in Base2Sym.items():
    Sym2Base[''.join(value)]=key

########################################

N1_sym=[]
N2_sym=[]
s1 = int(int(input())/h)
for i in range(s1):
    N1_sym.append(''.join([input() for i in range(h)]))
s2 = int(int(input())/h)
for i in range(s2):
    N2_sym.append(''.join([input() for i in range(h)]))

#########################################

N1=0
N2=0
for i in N1_sym:
    N1=N1*20+Sym2Base[i]
for i in N2_sym:
    N2=N2*20+Sym2Base[i]

#########################################

operation = input()
if operation=='+':
    result=N1+N2
elif operation=='*':
    result=N1*N2
elif operation=='-':
    result=N1-N2
elif operation=='/':
    result=N1/N2

result_Base=[]
if result==0:
    result_Base.append(0)

while not(result/BASE==0):
    result_Base.append(result % BASE)
    result=int(result/BASE)

result_Base.reverse()
for i in result_Base:
    for j in Base2Sym[i]:
        print(j)
    
