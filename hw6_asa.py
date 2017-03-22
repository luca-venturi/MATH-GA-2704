from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# variables initialization

def C0(x): return np.exp(-np.abs(x))
def C1(x): return np.exp(-x**2/2)
def C2(x): return float(np.abs(x) <= 1)

L = 50
N = L*30

x = np.array([complex(i*L/N) for i in range(N)])

# function to estimate covariance

"""
def estimate_covariance_ft(C_vec):
	
	dft_C = np.fft.fft(C_vec)*(L/N)
	it = 1000
	X = np.zeros((N,it))
	for i in range(it):
		phi = np.zeros((N), dtype=complex)
		for n in range(N):
			A = np.random.normal(loc=0.0, scale=1.0)
			B = np.random.normal(loc=0.0, scale=1.0)
			phi[n] = (A+B*1.j)*np.sqrt(L*dft_C[n]*0.5)
		X[:,i] = np.real(np.fft.ifft(phi))*(N/L)

	return np.cov(X)[0,:], X[:,0]
"""

def estimate_covariance_ft_circ(C_vec_tilde):
	
	N_tilde = 2*N-2
	invdft_C_tilde = np.fft.ifft(C_vec_tilde)*N_tilde
	it = 1000
	X = np.zeros((N,it))
	for i in range(it):
		phi = np.zeros((N_tilde), dtype=complex)
		for n in range(N_tilde):
			A = np.random.normal(loc=0.0, scale=1.0)
			B = np.random.normal(loc=0.0, scale=1.0)
			phi[n] = (A+B*1.j)*np.sqrt(invdft_C_tilde[n])
 		X[:,i] = np.real(np.fft.fft(phi)[:N])/np.sqrt(2*N_tilde)

	return np.cov(X)[0,:], X[:,0]

def estimate_covariance_chol(C_vec):
	
	real_cov_matrix = np.zeros((N,N))
	for n in range(N):
		real_cov_matrix[n][n:] = np.real(C_vec[:N-n])*0.5
		real_cov_matrix[n,n] *= 0.5  
	real_cov_matrix = (real_cov_matrix + real_cov_matrix.T)
	K = np.linalg.cholesky(real_cov_matrix)
	it = 10000
	X = np.zeros((N,it))
	for i in range(it):
		phi = np.zeros((N))
		for n in range(N):
			phi[n] = np.random.normal(loc=0.0, scale=1.0)
		X[:,i] = K.dot(phi)

	return np.cov(X)[0,:], X[:,0]

# C1

C_vec = np.array([C1(x[i]) for i in range(N)])
C_vec_tilde = np.append(C_vec, C_vec[N-2:0:-1])

# C1: DFT

X_cov, X_sample = estimate_covariance_ft_circ(C_vec_tilde*2.0)

plt.plot(np.real(x),X_sample)
plt.show()
plt.plot(np.real(x),np.real(C_vec),np.real(x),X_cov)
plt.show()
"""
# C1: Choleski

X_cov, X_sample = estimate_covariance_chol(C_vec)

plt.plot(np.real(x),X_sample)
plt.show()
plt.plot(np.real(x),np.real(C_vec),np.real(x),X_cov)
plt.show()
"""
