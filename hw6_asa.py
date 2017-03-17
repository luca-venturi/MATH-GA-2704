from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def C0(x): return complex(2.*np.exp(-np.abs(x)))
def C1(x): return complex(2.*np.exp(-x**2/2))
def C2(x): return complex(2.*float(np.abs(x) <= 1))

L = 20
N = L*10

x = np.array([complex(i*L/N) for i in range(N)])

# covariance: C0

C0_vec = np.array([C2(x[i]) for i in range(N)])
dft_C0 = np.fft.fft(C0_vec)*(L/N)
phi = []
for n in range(N):
	A = np.random.normal(loc=0.0, scale=1.0)
	B = np.random.normal(loc=0.0, scale=1.0)
	phi.append((A+B*1.j)*np.sqrt(L*dft_C0[n]/2))
phi = np.array(phi)
X = np.real(np.fft.ifft(phi)*(N/L))

plt.plot(np.real(x),X)
plt.show() 

# estimate covariance

it = 100000
X = np.zeros((it,N))
X_mean = np.zeros((N))
X_cov = np.zeros((N))

for i in range(it):
	phi = []
	for n in range(N):
		A = np.random.normal(loc=0.0, scale=1.0)
		B = np.random.normal(loc=0.0, scale=1.0)
		phi.append((A+B*1.j)*np.sqrt(L*dft_C0[n]/2))
	phi = np.array(phi)
	X[i,:] = np.real(np.fft.ifft(phi)*(N/L))

for n in range(N):
	X_mean[n] = np.mean(X[:,n])
	print X_mean[n]	
	X_cov[n] = np.sum((X[:,n]-X_mean[n])*(X[:,0]-X_mean[0]))/(it-1)

plt.plot(np.real(x),np.real(C0_vec),np.real(x),X_cov)
plt.show()
