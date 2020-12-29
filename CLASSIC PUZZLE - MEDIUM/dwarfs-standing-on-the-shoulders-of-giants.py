import sys
import math

NODE={}


def add(A):
        if not(A in NODE):
            NODE[A]=Node()
            NODE[A].Value=A

class Node:

    def __init__(self):
        self.Value=None
        self.Pupil={}
        self.Seq=None

    def link(self,A):
        self.Pupil[A]=NODE[A]

    def influence(self):
        SEQ=[]
        if self.Seq:
            return self.Seq
        if not(self.Pupil):
            self.Seq=1
            return self.Seq
        for pupil in self.Pupil:
            seq=self.Pupil[pupil].influence()
            SEQ.append(seq)

        self.Seq=max(SEQ)+1
        return self.Seq

    def Print(self):
        print('Node : ',self.Value,file=sys.stderr)
        print('Pupil : ',self.Pupil.keys(),file=sys.stderr)
        print('Seq : ',self.Seq,file=sys.stderr)

n = int(input())  # the number of relationships of influence

INFLUENCE=Node()
for i in range(n):
    x, y = [int(j) for j in input().split()]

    if not(x in NODE):
        add(x)
    if not(y in NODE):
        add(y)
    NODE[x].link(y)

SEQ=[]
for i in NODE:
    SEQ.append(NODE[i].influence())

print(max(SEQ))
