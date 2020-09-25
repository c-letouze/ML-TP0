#! /usr/bin/env python3
# Coraline Letouzé
# 19 Sept 2020
# Machine learning for physicists - practical work 0
# Exercice 1: Factorial function

import numpy as np
import scipy.stats as stat

def factorial(n):
    """ Return n! of a positive integer n. """
    if n<0 or n%1!=0:
        return None
    elif n==0:
        return 1
    else:
        return n*factorial(n-1)

assert factorial(0) == 1
assert factorial(5) == 120
assert factorial(10) == 3628800
factorial(-1)
factorial(2.5)
factorial(-5.6)
print("Tests passed!")

