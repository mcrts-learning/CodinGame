import sys
import math

class Node:
    def __init__(self):
        self.Children={}

    def Add(self,telephone):
        if len(telephone)>0:
            number=int(telephone[0:1])
            subtelephone=telephone[1:]

            if not(number in self.Children):
                child=Node()
                child.Add(subtelephone)
                self.Children[number]=child
            else:
                self.Children[number].Add(subtelephone)

    def Size(self):
        Nelement=1
        for child in self.Children:
            Nelement+=self.Children[child].Size()

        return Nelement


n = int(input())
PHONENUMBERS=Node()

for i in range(n):
    telephone = input()
    PHONENUMBERS.Add(telephone)

print(PHONENUMBERS.Size()-1)
