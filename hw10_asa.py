from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# we numerically solve the SDE dX = 2Xdt + XdW

# Euler-Maruyama scheme

def em(lam, mu, end_time, dt):
	
	N = int(np.floor(end_time/dt))
	x = np.ones((N))
	for i in range(1,N):
		x[i] = x[i-1] + lam*x[i-1]*dt + mu*x[i-1]*np.random.normal(0.,dt)

	return x		

# Milstein scheme

def mil(lam, mu, end_time, dt):
	
	N = int(np.floor(end_time/dt))
	x = np.ones((N))
	for i in range(1,N):
		dW = np.random.normal(0.,dt)
		x[i] = x[i-1] + lam*x[i-1]*dt + mu*x[i-1]*dW + 0.5*(mu**2)*x[i-1]*(dW**2 - dt)

	return x

# real solution

def real_sol(lam, mu, end_time, dt):
	
	N = int(np.floor(end_time/dt))
	x = np.ones((N))
	x_mean = np.ones((N))
	t = 0.	
	for i in range(1,N):
		t += dt
		W = np.random.normal(0.,t)
		x[i] = np.exp((lam - 0.5*(mu**2))*t + mu*W)
		x_mean[i] = np.exp(lam*t)

	return x, x_mean

# weak convergence error

def w_conv_error(lam, mu, end_time, dt, met, M):

	N = int(np.floor(end_time/dt))
	x_num_mean = np.zeros((N))
	if met == 0:
		for i in range(M):
			x_num_mean += em(lam, mu, end_time, dt)
	elif met == 1: 
		for i in range(M):
			x_num_mean += mil(lam, mu, end_time, dt)
	x_num_mean /= M
	x_real, x_real_mean = real_sol(lam, mu, end_time, dt)
		
	return np.amax(np.absolute(x_num_mean - x_real_mean))

# strong convergence error

def s_conv_error(lam, mu, end_time, dt, met, M):

	N = int(np.floor(end_time/dt))
	err_mean = np.zeros((N))
	if met == 0:
		for i in range(M):
			real_s, real_s_mean = real_sol(lam, mu, end_time, dt)
			err_mean += (em(lam, mu, end_time, dt) - real_s)**2
	elif met == 1:
		for i in range(M):
			real_s, real_s_mean = real_sol(lam, mu, end_time, dt)
			err_mean += (mil(lam, mu, end_time, dt) - real_s)**2
	err_mean /= M
		
	return np.mean(err_mean)

# parameters setting

lam = 2.
mu = 1.
T = 1. 
n = 5
dt = [T/(4**i) for i in range(2,n+2)]
M = 1000
'''
# weak convergence of EM scheme

w_conv_em = np.zeros((n))
for i in range(n):
	w_conv_em[i] = w_conv_error(lam, mu, T, dt[i], 0, M)

alpha = np.polyfit(np.log2(dt), np.log2(w_conv_em), 1)

plt.plot(np.log2(dt), np.log2(w_conv_em), np.log2(dt), alpha[0]*np.log2(dt) + alpha[1])
plt.show()

print "\nEuler-Maruyama: weak convergence: alpha = ", alpha[0]
'''
# strong convergence of EM scheme

s_conv_em = np.zeros((n))
for i in range(n):
	s_conv_em[i] = s_conv_error(lam, mu, T, dt[i], 0, M)
	print i, "-th iteration done\n"

alpha = np.polyfit(np.log2(dt), np.log2(s_conv_em), 1)

plt.plot(np.log2(dt), np.log2(s_conv_em), np.log2(dt), alpha[0]*np.log2(dt) + alpha[1])
plt.show()

print "\nEuler-Maruyama: strong convergence: alpha = ", alpha[0]

