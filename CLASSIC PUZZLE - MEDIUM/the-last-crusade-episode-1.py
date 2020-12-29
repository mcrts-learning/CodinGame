import sys
import math

Type=[]
Type.append({})
Type.append({'T':'D','L':'D','R':'D'})
Type.append({'L':'R','R':'L'})
Type.append({'T':'D'})
Type.append({'T':'L','R':'D'})
Type.append({'T':'R','L':'D'})
Type.append({'L':'R','R':'L'})
Type.append({'T':'D','R':'D'})
Type.append({'L':'D','R':'D'})
Type.append({'T':'D','L':'D'})
Type.append({'T':'L'})
Type.append({'T':'R'})
Type.append({'R':'D'})
Type.append({'L':'D'})

w, h = [int(i) for i in input().split()]

GRID=[[Type[int(j)] for j in input().split()] for i in range(h)]
ex = int(input())  # the coordinate along the X axis of the exit (not useful for this first mission, but must be read).

# game loop
while True:
    xi, yi, pos = input().split()
    xi = int(xi)
    yi = int(yi)
    pos=pos[0]
    NEXT=GRID[yi][xi][pos]
    if NEXT =='L':
        xi+=-1
        yi+=0
    elif NEXT =='R':
        xi+=1
        yi+=0
    else:
        xi+=0
        yi+=1
    print(xi,yi)
