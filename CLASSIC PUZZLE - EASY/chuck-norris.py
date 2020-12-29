import sys
import math

message = input()
B=[]
for i in message:
    B+='{0:07b}'.format(ord(i))

print(B,file=sys.stderr)



CODE=[]
if B[0]== "1":
    CODE+="0 0"
    c="1"
else:
    CODE+="00 0"
    c="0"
i=1
while i<len(B):
    if B[i]==c:
        CODE+="0"
    else:
        CODE+=" "
        if B[i]=="1":
            CODE+="0 0"
            c="1"
        else:
            CODE+="00 0"
            c="0"
    i+=1

print("".join(CODE))
