from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# I actually computed the constant Z using WolframAlpha (it was taking too long with python)
# alpha = 1.2
# Z = 5.59158
# I use alpha = 1.5 instead of alpha = 1.2 

alpha = 1.5
Z = 2.61238

# I get as input the number of iterations

N = input('Number of steps = ') # N = 6*(10**6)

# This function computes a vector of samples from the empirical distribution with 'small steps'-type proposal matrix and the corresponding r(t)

def small_steps(N, alpha):

	x = np.zeros((N+1))
	x[0] = 1
	for i in range(N):
		if x[i] == 1:
			U = np.random.rand()
			if U > 2**(-alpha-1):
				x[i+1] = 1
			else:
				x[i+1] = 2
		else:
			x[i+1] = np.random.choice([x[i]-1,x[i]+1])
			U = np.random.rand()
			if (x[i+1] == x[i]+1) & (U > (x[i]/x[i+1])**alpha):
				x[i+1] = x[i]
	
	R = 5
	r = np.zeros((N+1))
	r[0] = 1
	for i in range(N):
		if x[i+1] <= R:
			r[i+1] = r[i]+1
		else:
			r[i+1] = r[i]
	for i in range(1,N+1):
		r[i] /= (i+1)

	return x, r

# This function computes a vector of samples from the empirical distribution with 'large steps'-type proposal matrix and the corresponding r(t)

def large_steps(N, alpha):

	K = 1000
	aus = np.zeros((K)) 
	pi = np.zeros((K)) 
	H = np.zeros((K,K))
	A = np.zeros((K,K))
	for i in range(K):
		pi[i] = (i+1)**(-alpha)	
		aus[i] = K + sum([abs(k-i) for k in range(K)])
	for i in range(K):	
		for j in range(K):
			H[i,j] = (1 + abs(i-j))/aus[i]
			A[i,j] = np.minimum(1,pi[j]*aus[i]/(pi[i]*aus[j])) # np.minimum(1,pi[j]*H[j,i]/(pi[i]*H[i,j]))
	
	x = np.zeros((N+1))
	x[0] = 1
	for i in range(N):
		x[i+1] = np.random.choice(range(1,K+1),p = H[x[i]-1,:])
		U = np.random.rand()
		if U > A[x[i]-1,x[i+1]-1]:
			x[i+1] = x[i]
	
	R = 5
	r = np.zeros((N+1))
	r[0] = 1
	for i in range(N):
		if x[i+1] <= R:
			r[i+1] = r[i]+1
		else:
			r[i+1] = r[i]
	for i in range(1,N+1):
		r[i] /= (i+1)

	return x, r

# This function computes a vector of samples from the theoretical distribution and the theoretical r (r(t)=r in this case)

def real_dis(N, Z, alpha):

	M = 1000
	p_real = np.zeros((M))
	for i in range(M-1):
		p_real[i] = (i+1)**(-alpha)/Z
	p_real[M-1] = 1-sum(p_real[:-1])
	
	R =5
	r = np.ones((N+1))*sum(p_real[:R])
	x = np.zeros((N+1))
	for i in range(N+1):
		x[i] = np.random.choice(range(1,M+1),p = p_real)

	return x, r

type_step = input('Type of steps = ')
 
# I get as input the the type of proposal matrix I want to use: 1 for 'large steps', 0 for 'small steps'

if type_step == 1:
	
	x_large, r_large = large_steps(N, alpha)
	x_real, r_real = real_dis(N, Z, alpha)

	bins = np.linspace(1, 25, num=26)
	plt.hist([x_large,x_real], bins, normed=True)
	plt.show()

	t = range(N+1)
	plt.plot(t,r_large,'b',t,r_real,'g')
	plt.show()

elif type_step == 0:

	x_small, r_small = large_steps(N, alpha)
	x_real, r_real = real_dis(N, Z, alpha)

	bins = np.linspace(1, 25, num=26)
	plt.hist([x_small,x_real], bins, normed=True)
	plt.show()

	t = range(N+1)
	plt.plot(t,r_small,'b',t,r_real,'g')
	plt.show()
		

