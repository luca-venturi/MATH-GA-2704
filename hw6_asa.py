from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# first we define two functions to evaluate the discrete fourier transform (dft) and its inverse (dft_inv)

def dft(N, L, f, x, k): # f is a function

	dft_f = np.zeros((N), dtype=complex)
	for n in range(N):
		dft_f[n] = sum([f(x[i])*np.exp(-x[i]*k[n]*1.j) for i in range(N)])
	dft_f *= L/N
	print dft_f

	return dft_f

def dft_inv(N, L, f, x, k): # f is the vector of dft of a function

	dft_f = np.zeros((N), dtype=complex)
	for i in range(N):
		dft_f[i] = sum([f[n]*np.exp(x[i]*k[n]*1.j) for n in range(N)])
	dft_f /= L

	return dft_f

def C0(x): return complex(2.*np.exp(-np.abs(x)))
C1 = lambda x: complex(2.*np.exp(-x**2/2))
C2 = lambda x: complex(2.*float(np.abs(x) <= 1))

L = 10
N = L*10

x = np.array([complex(i*L/N) for i in range(N)])
k = np.array([complex(n*2.*np.pi/L) for n in range(N)])

# covariance: C0

dft_C0 = dft(N, L, C0, x, k)
phi = np.zeros((N+1), dtype=complex)
phi[0] = np.sqrt(L*dft_C0[0])*(np.random.normal(loc=0.0, scale=1.0))
for n in range(1,N/2):
	phi[n] = np.sqrt(L*dft_C0[n]/2)*(np.random.normal(loc=0.0, scale=1.0) + np.random.normal(loc=0.0, scale=1.0)*1.j)
phi[N/2] = np.sqrt(L*dft_C0[N/2])*(np.random.normal(loc=0.0, scale=1.0))
X = np.real(dft_inv(N+1, L, phi, x, k))

plt.plot(x,X)
plt.show()


