import sys
import math

POINT={'e':1,'a':1,'i':1,'o':1,'n':1,'r':1,'t':1,'l':1,'s':1,'u':1,'d':2,'g':2,'b':3,'c':3,'m':3,'p':3,'f':4,'h':4,'v':4,'w':4,'y':4,'k':5,'j':8,'x':8,'q':10,'z':10}

LETTRE={}
DICT={}
LISTE=[]
n = int(input())
for i in range(n):
    w = input()
    score=0
    for lettre in w:
        score+=POINT[lettre]

        if w in DICT:
            if lettre in DICT[w]:
                DICT[w][lettre]+=1
            else:
                DICT[w][lettre]=1
        else:
            DICT[w]={lettre:1}


    LISTE.append([w,score])

LISTE.sort(key=lambda x: x[1],reverse=True)

print(LISTE,file=sys.stderr)
print(DICT,file=sys.stderr)

letters = input()
for lettre in letters:
    if lettre in LETTRE:
        LETTRE[lettre]+=1
    else:
        LETTRE[lettre]=1

print(LETTRE,file=sys.stderr)

for mot in LISTE:
    for lettre in mot[0]:
        if not(lettre in LETTRE):
            MOT=None
            break
        elif LETTRE[lettre]<DICT[mot[0]][lettre]:
            MOT=None
            break
        else:
            MOT=mot[0]
    if MOT:
        print(MOT)
        exit()
