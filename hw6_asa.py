from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def C0(x): return complex(2.*np.exp(-np.abs(x)))
def C1(x): return complex(2.*np.exp(-x**2/2))
def C2(x): return complex(2.*float(np.abs(x) <= 1))

L = 10
N = L*10

x = np.array([complex(i*L/N) for i in range(N)])

# covariance: C0

C0_vec = np.array([C0(x[i]) for i in range(N)])
dft_C0 = np.fft.fft(C0_vec)*(L/N)
phi = []
for n in range(N):
	A = np.random.normal(loc=0.0, scale=1.0)
	B = np.random.normal(loc=0.0, scale=1.0)
	phi.append((A+B*1.j)*np.sqrt(L*dft_C0[n]/2))
phi = np.array(phi)
X = np.real(np.fft.ifft(phi)*(N/L))

plt.plot(x,X)
plt.show() 

# estimate covariance


