import sys
import math
import numpy as np

WALL={}
OBSTACLE={}
PANEL={}
T={}

DIRECTION={'SOUTH':np.array((0,1)),'EAST':np.array((1,0)),'NORTH':np.array((0,-1)),'WEST':np.array((-1,0))}
CHAR=[' ','N','S','E','W','I','B','T','$']
#####################################################

def Affichage():
    G=GRID[:]
    X,Y=(Bender.x,Bender.y)
    G[Y][X]='@'
    for i in range(len(G[0])):
        print(G[i],file=sys.stderr)
    print('',file=sys.stderr)
    G[Y][X]=' '

#####################################################

class robot:

    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.direction='SOUTH'
        self.casseur=False
        self.alive=True
        self.priority=['SOUTH','EAST','NORTH','WEST']

    def Panel(self):
        X,Y=(self.x,self.y)
        cell=GRID[Y][X]

        if cell=='S':
            self.direction='SOUTH'
        elif cell=='N':
            self.direction='NORTH'
        elif cell=='E':
            self.direction='EAST'
        elif cell=='W':
            self.direction='WEST'
        elif cell=='I':
            self.priority.reverse()
        elif cell=='B':
            self.casseur=not(self.casseur)
        elif cell=='T':
            print('tp',file=sys.stderr)
            (self.x,self.y)=T[(self.x,self.y)]
        elif cell=='$':
            self.alive=False

    def ChangeDir(self):
        X,Y=(self.x,self.y)+DIRECTION[self.direction]

        if (X,Y) in WALL:
            for Dir in self.priority:
                X,Y=(self.x,self.y)+DIRECTION[Dir]
                cell=GRID[Y][X]
                if cell in CHAR:
                    self.direction=Dir
                    break

        elif (X,Y) in OBSTACLE:
            print('X',X,Y,file=sys.stderr)
            if not(self.casseur):
                for Dir in self.priority:
                    X,Y=(self.x,self.y)+DIRECTION[Dir]
                    cell=GRID[Y][X]
                    if cell in CHAR:
                        self.direction=Dir
                        break

    def Move(self):
        X,Y=(self.x,self.y)+DIRECTION[self.direction]
        cell=GRID[Y][X]
        if cell=='X':
            del OBSTACLE[(X,Y)]
            GRID[Y][X]=' '
        (self.x,self.y)=(X,Y)

#####################################################

l, c = [int(i) for i in input().split()]
GRID=[]
for i in range(l):
    GRID.append(list(input()))

t=[]
for i in range(l):
    for j in range(c):
        if GRID[i][j]=='@':
            GRID[i][j]=' '
            Bender=robot(j,i)
        elif GRID[i][j]=='T':
            t.append((j,i))
        elif GRID[i][j]=='#':
            WALL[(j,i)]='#'
        elif GRID[i][j]=='X':
            OBSTACLE[(j,i)]='X'
if t:
    T[t[0]]=t[1]
    T[t[1]]=t[0]

SEQ=[]
i=0
while Bender.alive:
    #Affichage()
    Bender.ChangeDir()
    SEQ.append(Bender.direction)
    Bender.Move()
    print(Bender.direction,file=sys.stderr)
    Bender.Panel()
    i+=1
    if i>=200:
        print('LOOP')
        exit()

for i in SEQ:
        print(i)
    
