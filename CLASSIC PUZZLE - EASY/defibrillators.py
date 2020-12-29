import sys
import math
import re
import numpy as np



lon = float(re.sub(",",".",input()))
lat = float(re.sub(",",".",input()))
n = int(input())
DISTANCE=[["","",""]for x in range(n)]
for i in range(n):
    defib = input()
    NOM=defib.split(";")[1]
    LON=defib.split(";")[4]
    LON=float(re.sub(",",".",LON))
    LAT=defib.split(";")[5]
    LAT=float(re.sub(",",".",LAT))
    DISTANCE[i]=[NOM,LON,LAT]

print(lon,file=sys.stderr)

D=[]
for i in DISTANCE:
    D_x=(i[1]-lon)*math.cos((i[2]+lat)/2)
    D_y=i[2]-lat
    D.append(math.sqrt(D_x*D_x+D_y*D_y)*6371)

idx = np.argmin(D)

print(DISTANCE[idx][0])
