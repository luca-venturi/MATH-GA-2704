from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

a = np.array([[1,1,2,2],[7,7,3,3]])
b = np.array([3,4,5,6])
c = a[1][2:]
print a
print a.dot(b)
print c
print b[:2]
