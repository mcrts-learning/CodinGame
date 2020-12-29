import sys
import math

def RN(i,j):
    global GRID
    for x in range(j+1,width):
        if GRID[i][x]=="0":
            return(i,x)
    return(-1,-1)

def BN(i,j):
    global GRID
    for x in range(i+1,height):
        if GRID[x][j]=="0":
            return(x,j)
    return(-1,-1)


width = int(input())  # the number of cells on the X axis
height = int(input())  # the number of cells on the Y axis
GRID=[""for x in range(height)]
for i in range(height):
    GRID[i] = input()  # width characters, each either 0 or .


for i in range(height):
    for j in range(width):
        if GRID[i][j]=="0":
            y1,x1=(i,j)
            y2,x2=RN(i,j)
            y3,x3=BN(i,j)
            print(x1,y1,x2,y2,x3,y3)
