from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# variables initialization

def C0(x): return complex(2.*np.exp(-np.abs(x)))
def C1(x): return complex(2.*np.exp(-x**2/2))
def C2(x): return complex(2.*float(np.abs(x) <= 1))

L = 100
N = L*20

x = np.array([complex(i*L/N) for i in range(N)])

# function to estimate covariance

def estimate_covariance_ft(C_vec):
	
	dft_C = np.fft.fft(C_vec)*(L/N)
	it = 10000
	X = np.zeros((N,it))
	for i in range(it):
		phi = []
		for n in range(N):
			A = np.random.normal(loc=0.0, scale=1.0)
			B = np.random.normal(loc=0.0, scale=1.0)
			phi.append((A+B*1.j)*np.sqrt(L*dft_C[n]/2))
		phi = np.array(phi)
		X[:,i] = np.real(np.fft.ifft(phi)*(N/L))

	return np.cov(X)[0,:], X[:,0]

def estimate_covariance_chol(C_vec):
	
	real_cov_matrix = np.zeros((N,N))
	for n in range(N):
		real_cov_matrix[n,n:] = np.real(C_vec[n:])*0.5
		real_cov_matrix[n,n] *= 0.5  
	real_cov_matrix = (real_cov_matrix + real_cov_matrix.T)
	K = np.linalg.cholesky(real_cov_matrix)
	it = 1000
	X = np.zeros((N,it))
	for i in range(it):
		phi = []
		for n in range(N):
			phi.append(np.random.normal(loc=0.0, scale=1.0))
		phi = np.array(phi)
		X[:,i] = K.dot(phi)

	return np.cov(X)[0,:], X[:,0]

# C0

C0_vec = np.array([C2(x[i]) for i in range(N)])
	
X_cov, X_sample = estimate_covariance_ft(C0_vec)

plt.plot(np.real(x),X_sample)
plt.show()
plt.plot(np.real(x),np.real(C0_vec)*0.5,np.real(x),X_cov)
plt.show()

"""X_cov, X_sample = estimate_covariance_chol(C0_vec)

plt.plot(np.real(x),X_sample)
plt.show()
plt.plot(np.real(x),np.real(C0_vec)*0.5,np.real(x),X_cov)
plt.show()"""



