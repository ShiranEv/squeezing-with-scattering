"""
Created on Tue Jan 14 18:26:41 2025
@author: Shiran Even-Haim
"""
#%% imports
from qutip import *
import matplotlib.pyplot as plt
import numpy as np

#%% functions
# probability of k losses with ns scattering events
def p_k_loss_with_ns_scattering(ns, k, N):
    if k >ns:
        return 0
    if k>N:
        return  0
    if ns == 0:
        return 1
    if k==0:
        return 0
    return (k/N) * p_k_loss_with_ns_scattering(ns-1, k, N) + ((1-k+N)/N) * p_k_loss_with_ns_scattering(ns-1, k-1, N)

# returns rho_k, rho with k losses
def scattering_channel(rho, N, k):
    for i in range(k):
        rho = (1/(N-i)) * (destroy(N+1) * rho * create(N+1) + (N-i-expect(num(N+1), rho)) * rho)
    return rho

# returns rho with ns scattering events
def return_scattered_state(rho, ns, N):
    return sum([p_k_loss_with_ns_scattering(ns, k, N) * scattering_channel(rho, N, k) for k in range(ns+1)])

# Q calculations
def La(x):
    return 1/(1+x**2)

def Ld(x):
    return -x/(1+x**2)

def T0(eta, N, x_a, x_c):
    return 1/((1+0.5*N*eta*La(x_a))**2+(x_c+0.5*N*eta*Ld(x_a))**2)

def Q_per_scattered(eta, N, x_a, x_c):
    return 0.5*T0(eta, N, x_a, x_c)* (-2*eta*Ld(x_a)*(1-x_c*x_a+0.5*N*eta))

def nsc_for_desired_squeezing(x_a, N, chi_tau, Gamma=2*np.pi*0.184, kappa=2*np.pi*0.84, eta=3.2):
    return int(chi_tau * N / np.abs(Q_per_scattered(eta, N, x_a, x_a*Gamma/kappa)))

def squeezing(rho, N, chi_tau):
    return (-1j * chi_tau * jmat(N/2, 'z')**2).expm() * rho * (-1j * chi_tau * jmat(N/2, 'z')**2).expm().dag()

# This is the main function that calculates the final state
def squeezing_with_scattering(rho, chi_tau, x_a):
    N = rho.shape[0] - 1
    ns = nsc_for_desired_squeezing(x_a, N, chi_tau)
    squeezed_state = squeezing(rho, N, chi_tau)
    scattered_state = return_scattered_state(squeezed_state, ns, N)
    return scattered_state

#%% parameters
chi_tau = np.pi/4
N=20

#%% find optimal x_a
x_a_vec = np.linspace(10, 20, 20)
ns_vec = [nsc_for_desired_squeezing(x_a, N, chi_tau, Gamma=2*np.pi*0.184, kappa=2*np.pi*0.84, eta=3.2) for x_a in x_a_vec]
x_a = x_a_vec[np.argmin(ns_vec)]

#%% plot
plt.plot(x_a_vec, ns_vec, '.')
plt.xlabel("x_a")
plt.ylabel("n_sc")
plt.show()

#%% final state
theta=np.pi/4
rho_final = squeezing_with_scattering(ket2dm(spin_coherent(N/2, theta, 0)), chi_tau, x_a)
