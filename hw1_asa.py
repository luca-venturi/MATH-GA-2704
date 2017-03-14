import numpy as np
import matplotlib.pyplot as plt

### Part 1

A = np.random.rand(3,3)
b = np.random.rand(3,1)

x = np.linalg.solve(A,b)

print b
print np.dot(A,x)

if np.max(np.abs(np.dot(A,x)-b))>1e-8:
	print "Error in solving the system"
else:
	print "System solved successfully"

### Part 2

t = np.arange(0.,2.,1e-3)
plt.plot(t,np.exp(t)+np.pi/2)
plt.show()
