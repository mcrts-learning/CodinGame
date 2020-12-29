import sys
import math
import numpy as np

G=3.711
Y_margin=40
X_margin=250
V_margin=5;
Vy_max=40;
Vx_max=20;

class Lander:

    def Update(self,x=0,y=0,vx=0,vy=0,fuel=0,angle=0,power=0):
        self.x=x
        self.y=y
        self.vx=vx
        self.vy=vy
        self.fuel=fuel
        self.angle=angle
        self.power=power
        self.speed=math.sqrt(vx**2+vy**2)

    def Target(self,L=0,R=0,Y=0):
        self.targetL=L
        self.targetR=R
        self.targetY=Y

    def isOverTarget(self):
        return(self.x>self.targetL and self.x<self.targetR)

    def isClosetoTarget(self):
        return(self.x>(self.targetL-X_margin) and self.x<(self.targetR+X_margin))

    def isLanding(self):
        return(self.y<(self.targetY+Y_margin))

    def isSpeedSafe(self):
        return(abs(self.vx)<Vx_max and abs(self.vy)<Vy_max)

    def isWrongDir(self):
        return((self.x<self.targetL and self.vx<0) or (self.x>self.targetR and self.vx>0))

    def isVxtooFast(self):
        return(abs(self.vx)>4*Vx_max)

    def isVxtooSlow(self):
        return(abs(self.vx)<2*Vx_max)

    def AngletoTarget(self):
        angle=int(math.degrees(math.acos(G/4)))
        if self.x<self.targetL:
            return(-angle)
        elif self.x>self.targetR:
            return(angle)
        else:
            return(0)

    def AngletoSlow(self):
        return(int(math.degrees(math.asin(self.vx/self.speed))))

    def PowertoHover(self):
        if self.vy>=0:
            return(3)
        else:
            return(4)


surface_n = int(input())# the number of points used to draw the surface of Mars.

LAND_x=[]
LAND_y=[]
GROUND=[]

for i in range(surface_n):
    l_x,l_y=[int(j) for j in input().split()]
    LAND_x.append(l_x)
    LAND_y.append(l_y)

    if i>=1:
        GROUND.append(LAND_y[i-1]-LAND_y[i])
idx=GROUND.index(0)

ship=Lander()
ship.Target(LAND_x[idx],LAND_x[idx+1],LAND_y[idx])

print(ship.targetL,ship.targetR,file=sys.stderr)


# game loop

while True:
    x, y, vx, vy, fuel, angle, power = [int(i) for i in input().split()]

    ship.Update(x,y,vx,vy,fuel,angle,power)


    if not(ship.isOverTarget()):
        if ship.isWrongDir() or ship.isVxtooFast():
            print(ship.AngletoSlow(),4)
        elif ship.isVxtooSlow():
            print(ship.AngletoTarget(),4)
        elif ship.isClosetoTarget():
            print(ship.AngletoSlow(),4)
        else:
            print(0,ship.PowertoHover())

    else:
        if ship.isLanding():
            print(0,4)
        elif ship.isSpeedSafe():
            print(0,3)
        else:
            print(ship.AngletoSlow(),4)
