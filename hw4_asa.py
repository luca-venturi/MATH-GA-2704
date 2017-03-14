from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

### part (a) ###

d = 4					# d = length of the side of the grid 
D = d**2				# D = number of squares in the grid
Q = np.zeros((D,D))		# inizialize generator matrix Q
P = np.zeros((D,D))		# inizialize transition matrix P

# we make Q and P

for i in range(D-1):
	if (i+1)%d != 0:
		Q[i][i+1] = 1
		Q[i+1][i] = 1
		P[i][i+1] = 1
		P[i+1][i] = 1
for i in range(D-d):
	Q[i][i+d] = 1
	Q[i+d][i] = 1
	P[i][i+d] = 1
	P[i+d][i] = 1

for i in range(D):
	Q[i,i] = -sum(Q[i])
	P[i] /= sum(P[i])

# we make lamb = holding time parameters vector

lamb = [-Q[i,i] for i in range(D)]

### part (b) ###

# we evaluate the mean first passage time using the backward Kolmogorov equations

rmt = np.linalg.solve(Q[:-1,:-1],-np.ones((D-1)))[0]

print rmt

### part (c) ###

# this function generate the corresponding Markov chain (using Kinetic Monte Carlo method) until Theseus hits the Minotaur and return this hitting time 

def generate_first_time(D, lamb, P):

	x = 0
	t = 0
	while x < D-1:
		t += np.random.exponential(scale=1/lamb[x])
		x = np.random.choice(range(D),p=P[x])

	return t

# this function computes the average mean first-passage time obtained by generating n=1,...,N trajectories 

def generate_mean_time_cons(N, D, lamb, P):

	t = [generate_first_time(D, lamb, P)]
	for i in range(1,N):
		t.append(t[i-1] + generate_first_time(D, lamb, P))
	for i in range(1,N):
		t[i] /= (i+1)

	return t

# we plot the value of mfpt_b obtained for N = 1,...,2**16 versus the 'true value' mfpt_a

K = 16
N = 2**K
t = generate_mean_time_cons(N, D, lamb, P)
plt.plot(range(1,N+1),rmt*np.ones((N)),range(1,N+1),t)
plt.xlim(1,N)
plt.ylim(14,16)
plt.show()

# we evaluate (and plot in log_2 scale) the error |mfpt_a - mfpt_b| for different values of N (2**k for k = 0,...,M-1)
# and we evaluate the linear regression of the datas we obtained to give an estimate of alpha

x = [abs(t[2**k-1]-rmt) for k in range(K+1)]

aus = np.vstack([range(K+1), np.ones(K+1)]).T
alpha, c = np.linalg.lstsq(aus, np.log2(x))[0]

plt.plot(range(K+1),np.log2(x),range(K+1),[alpha*k+c for k in range(K+1)])
plt.show()

print alpha
