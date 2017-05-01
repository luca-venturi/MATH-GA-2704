from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# we numerically solve the SDE dX = 2Xdt + XdW

# Euler-Maruyama scheme

def em(lam, mu, end_time, dt):
	
	N = int(np.floor(end_time/dt))
	x = np.ones((N))	
	x_real = np.ones((N))
	t = 0.
	W = 0.
	sqrtdt = np.sqrt(dt)
	for i in range(1,N):
		t += dt
		dW = np.random.normal(0.,sqrtdt)
		W += dW
		x_real[i] = np.exp((lam - 0.5*(mu**2))*t + mu*W)
		x[i] = x[i-1] + lam*x[i-1]*dt + mu*x[i-1]*dW

	return x, x_real		

# Milstein scheme

def mil(lam, mu, end_time, dt):
	
	N = int(np.floor(end_time/dt))
	x = np.ones((N))	
	x_real = np.ones((N))
	t = 0.
	W = 0.
	sqrtdt = np.sqrt(dt)
	for i in range(1,N):
		t += dt
		dW = np.random.normal(0.,sqrtdt)
		W += dW
		x_real[i] = np.exp((lam - 0.5*(mu**2))*t + mu*W)
		x[i] = x[i-1] + lam*x[i-1]*dt + mu*x[i-1]*dW + 0.5*(mu**2)*x[i-1]*(dW**2 - dt)

	return x, x_real

# weak convergence error

def w_conv_error(lam, mu, end_time, dt, met, M):

	N = int(np.floor(end_time/dt))
	x_num_mean = np.zeros((N))
	x_real_mean = np.zeros((N))
	if met == 0:
		for i in range(M):
			tmp1, tmp2 = em(lam, mu, end_time, dt)
			x_num_mean += tmp1
			x_real_mean += tmp2
	elif met == 1: 
		for i in range(M):
			tmp1, tmp2 = mil(lam, mu, end_time, dt)
			x_num_mean += tmp1
			x_real_mean += tmp2
	x_num_mean /= M
	x_real_mean /= M
		
	return np.amax(np.absolute(x_num_mean - x_real_mean))

# strong convergence error

def s_conv_error(lam, mu, end_time, dt, met, M):

	N = int(np.floor(end_time/dt))
	err_mean = np.zeros((N))
	if met == 0:
		for i in range(M):
			tmp1, tmp2 = em(lam, mu, end_time, dt)
			err_mean += np.abs(tmp1-tmp2)
	elif met == 1:
		for i in range(M):
			tmp1, tmp2 = mil(lam, mu, end_time, dt)
			err_mean += np.abs(tmp1-tmp2)
	err_mean /= M
		
	return np.amax(err_mean)

### (a) ###

# parameters setting

lam = 2.
mu = 1.
T = 1 
n = 5
dt = [T/(4**i) for i in range(2,n+2)]
M = 1000

# weak convergence of EM scheme

w_conv_em = np.zeros((n))
for i in range(n):
	w_conv_em[i] = w_conv_error(lam, mu, T, dt[i], 0, M)

alpha = np.polyfit(np.log2(dt), np.log2(w_conv_em), 1)

plt.plot(np.log2(dt), np.log2(w_conv_em), np.log2(dt), alpha[0]*np.log2(dt) + alpha[1])
plt.show()

print "\nEuler-Maruyama: weak convergence: alpha = ", alpha[0], "\n"

# strong convergence of EM scheme

s_conv_em = np.zeros((n))
for i in range(n):
	s_conv_em[i] = s_conv_error(lam, mu, T, dt[i], 0, M)

alpha = np.polyfit(np.log2(dt), np.log2(s_conv_em), 1)

plt.plot(np.log2(dt), np.log2(s_conv_em), np.log2(dt), alpha[0]*np.log2(dt) + alpha[1])
plt.show()

print "\nEuler-Maruyama: strong convergence: alpha = ", alpha[0], "\n"

### (b) ###

# weak convergence of Milstein scheme

w_conv_em = np.zeros((n))
for i in range(n):
	w_conv_em[i] = w_conv_error(lam, mu, T, dt[i], 1, M)

alpha = np.polyfit(np.log2(dt), np.log2(w_conv_em), 1)

plt.plot(np.log2(dt), np.log2(w_conv_em), np.log2(dt), alpha[0]*np.log2(dt) + alpha[1])
plt.show()

print "\nMilstein: weak convergence: alpha = ", alpha[0], "\n"

# strong convergence of Milstein scheme

s_conv_em = np.zeros((n))
for i in range(n):
	s_conv_em[i] = s_conv_error(lam, mu, T, dt[i], 1, M)

alpha = np.polyfit(np.log2(dt), np.log2(s_conv_em), 1)

plt.plot(np.log2(dt), np.log2(s_conv_em), np.log2(dt), alpha[0]*np.log2(dt) + alpha[1])
plt.show()

print "\nMilstein: strong convergence: alpha = ", alpha[0], "\n"

### (c) ###

# we evaluate lim_t EX_t^2 (~ EX_T^2 for T large) for different values of dt

lam = -3.
mu = np.sqrt(3.)
T = 10.
n = 3
dt = [0.4, 0.3, 0.2]
M = 50000

for i in range(n):
	lim = 0.
	for j in range(M):
		x, x_real = mil(lam, mu, T, dt[i])
		lim += x[-1]**2
	lim /= M
	print "\ndt = ", dt[i], "-> lim = ", lim, "\n"

### (d) ###

# we evaluate P( lim_t |X_t| = 0 ) (~ P( |X_T| < eps ) for T large, eps small) for different values of dt

lam = 0.5
mu = np.sqrt(6.)
T = 10.
eps = 1e-5
n = 3
dt = [0.4, 0.3, 0.2, 0.1]
M = 50000

for i in range(n):
	p = 0.
	for j in range(M):
		x, x_real = mil(lam, mu, T, dt[i])
		p += (x[-1] < eps)
	p /= M
	print "\ndt = ", dt[i], "-> p = ", p, "\n"

