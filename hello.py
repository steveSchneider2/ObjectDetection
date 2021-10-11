# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 15:09:57 2021

@author: steve
"""

import sys
def hello(a, b):
    print('hello and that is  your sum:')
    sum = a+b
    print(sum)

if __name__== "__main__":
    hello(int(sys.argv[1]), int(sys.argv[2]))