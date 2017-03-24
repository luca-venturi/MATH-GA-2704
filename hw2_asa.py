import numpy as np

### part (a) ###

n = 3					# n = lenght of the side of the grid 
N = n**2				# N = number of squares in the grid
P = np.zeros((N,N))		# inizialize transition matrix P

# we make P = adjacency matrix (of the graph induced by the grid)

for i in range(N-1):
	if (i+1)%n != 0:
		P[i][i+1] = 1
		P[i+1][i] = 1
for i in range(N-n):
	P[i][i+n] = 1
	P[i+n][i] = 1

# we make P = transition matrix, by normalizing its rows

for i in range(N):
	P[i][:] = P[i][:]/sum(P[i][:])

# we evaluate tau_1 = E[T_A | X_0=1] (for A={7,8,9} or more generally A={N-n+1,...,N}) by solving the linear system shown in class 

tau = np.linalg.solve(P[:N-n,:N-n] - np.diag(np.ones((N-n))), -np.ones((N-n)))

print "\nThe average time to hit the top row (evaluated with method (a)) is: ", tau[0]

### part (b) ###

# we evaluate G_0i = P(X_{T_A}=i | X_0=1) (for A={7,8,9} or more generally A={N-n+1,...,N} and i in A) by solving the linear system shown at Exercise 5 

G = np.linalg.solve(np.diag(np.ones((N-n)))-P[:N-n,:N-n],P[:N-n,-n:])

print "\nThe probabilities of hitting squares ", [N-n+i+1 for i in range(n)], "are, respectively:"
print [G[0,i] for i in range(n)]

### part (c) ###

# we define the function [sim_MC_first_passage] which takes as input:
# P = adjacency matrix
# x_0 = initial state of the random walk
# A = subset of set of states
# the function simulates such a Markov chain until it reach a state in A
# the function returns as output the first-passage time and a vector with the occurred states 

def sim_MC_first_passage(P, x_0, A):
	
	x = np.array([x_0]) # vector of the occurred states
	i = 0 # time
	while x[i] not in A:
		temp = np.random.choice(np.arange(np.size(P[0,:])), p=P[x[i],:]) # moves from x[i] to x[i+1] with the law given by P
		x = np.append(x, temp)
		i += 1
	
	return i, x

# we run the above defined function M times and we evaluate the mean first-passage time

M = 10000.
sum_T = 0.
for j in range(np.int(M)):
	temp_T, temp_x = sim_MC_first_passage(P,0,range(N-n,N))
	sum_T += temp_T
mean_T = sum_T/M # mean first-passage time

print "\nThe average time to hit the top row (evaluated with method (c)) is: ", mean_T, "\n"
