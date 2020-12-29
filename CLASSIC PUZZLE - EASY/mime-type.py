import sys
import math

n = int(input())  # Number of elements which make up the association table.
q = int(input())  # Number Q of file names to be analyzed.

EXT={}

for i in range(n):
    # ext: file extension
    # mt: MIME type.
    ext, mt = input().split()
    EXT[ext.casefold()]=mt

for i in range(q):
    fname=input().split(".") # One file name per line
    if len(fname)>1:
        if fname[-1].casefold() in EXT:
            print(EXT[fname[-1].casefold()])
        else:
            print('UNKNOWN')
    else:
        print('UNKNOWN')
