import numpy as np
import math
from scipy import integrate
import matplotlib.pyplot as plt
import scipy.special as sps
import scipy.integrate as spi
import pandas as pd
import dm_lib
import Annihilation_Rate as anr
import sys
import warnings
if not sys.warnoptions:
   warnings.simplefilter("ignore")

smaller = 1e-3
T_c = 1e5 #Kelvin
tolerance = 0.01

################################################
# Load phase shift computations
################################################
bench_key ="B1"
eos = 'HHJ1'
print(bench_key, eos)
m_X, a_X, m_phi,Delta = dm_lib.benchmark_values[bench_key]
def load_deltas(bench_key):
    if not "3" in bench_key:
        Phase_shift_path = "C:\\Users\\nkout\\Desktop\\Thesis\\Codes\\Recombination\\Phase_shift\\" + bench_key + "_phase_shift_data.csv"
        Phase_shift_data = pd.read_csv(Phase_shift_path)
        ls = Phase_shift_data['l'].to_numpy()
        deltas = Phase_shift_data['delta_l'].to_numpy()
        lmax = ls[-1]
    return deltas, lmax
alpha = np.sqrt(dm_lib.Kinetic(bench_key))
mu = m_X/2
kappa = mu*dm_lib.v0func(bench_key)
a_0 = 1/(a_X*mu)

def E_n(n):
    return a_X**2*mu/(2*n**2)

def omega(n):
    return E_n(n) + kappa**2/(2*mu)

################################################
# Radial Solution - Step 2
################################################

# At r-> inf:
def R_l(l,r):
    deltas = load_deltas(bench_key)[0][l]
    A = dm_lib.j_l(l,kappa*r)*np.cos(deltas)
    B = dm_lib.n_l(l,kappa*r)*np.sin(deltas)
    return (A - B)

################################################
# Matching - Step 3
################################################

def chi_l(l,r):
    '''
    Solution of eq. C12
    '''
    deltas = load_deltas(bench_key)[0][l]
    x = m_X*a_X*r
    A = 1
    B = np.exp(complex(0,1)*deltas)
    C = np.cos(deltas)*dm_lib.j_l(l,alpha*x) - np.sin(deltas)*dm_lib.n_l(l,alpha*x)
    return A*B*C

################################################
# Compute Sigma_T - Step 4
################################################

def sigma_l(l):
    deltas = load_deltas(bench_key)[0]
    return (l+1)*(np.sin(deltas[l+1] - deltas[l]))**2

def sigma_T_k2_4pi(lmax):
    sigma_T = 0
    for l in range(lmax):
        sigma_T += sigma_l(l)
        print(f"sigma_{l} = {sigma_l(l)}")
    for j in range(10):
        sigma_T += sigma_l(lmax - 1)
        print(sigma_T,j+1)
    return sigma_T

################################################
# Normalisation - Step 5
################################################

def norm(l):
    def integrand(r):
        return abs(chi_l(l,r)) ** 2

    # Define the range of integration
    r_min = 0  # Replace with the lower limit of r
    r_max = 2*np.pi/np.sqrt(dm_lib.Kinetic(bench_key))  # xnorm

    # Perform the integration to calculate the normalization factor
    integral, _ = integrate.quad(integrand, r_min, r_max)
    #print(integral)
    # Define the normalized function
    def normalized_chi_l(r):
        return chi_l(l,r) / np.sqrt(integral)
    return normalized_chi_l


################################################
# Integration - Step 6
################################################

def R_nl(n,l,r):
    A = 2/( n**(l+2) * np.math.factorial(2*l+1))
    B = np.sqrt( np.math.factorial(n+l)/ np.math.factorial(n - l - 1))
    C = (2*r)**l/( a_0**(l+3/2))*np.exp(-r /(n*a_0))
    D = sps.hyp1f1(1+l-n,2+2*l, 2*r/(n*a_0))
    return A*B*C*D

# Define the integrands

# Integrate the function from 0 to infinity
def integral_1(n,l):
    def integrand_1(r, n, l):
        normalized_chi_l = norm(l+1)
        chi_l_r = normalized_chi_l(r)
        return R_nl(n, l, r) * chi_l_r *  r ** 3
    result = spi.quad(integrand_1,0, np.inf, args=(n, l))[0]
    return result

def integral_2(n,l):
    def integrand_2(r, n, l):
        normalized_chi_l = norm(l-1)
        chi_l_r = normalized_chi_l(r)
        return R_nl(n, l, r) * chi_l_r * r ** 3
    if l ==0:
        result = 0
    else:
        result = spi.quad(integrand_2, 0, np.inf, args=(n, l-1))[0]
    return result


def sigmaV_rec_numeric(n,l):
    constant = a_X/(3*np.pi)
    A = (omega(n)**2 + 0.5*m_phi**2)*np.sqrt(omega(n)**2 - m_phi**2)
    ints = ((l + 1) * integral_1(n,l) ** 2) + (l * integral_2(n,l) ** 2)
    return constant*A*ints

def n_max(a_X,m_X,m_phi):
    return round(a_X*np.sqrt(m_X/(4*m_phi)))

def n_bar(a_X,m_X,m_phi):
    return round((a_X**2*m_X/(2*m_phi))**(1/3))

#print(n_bar(a_X,m_X,m_phi))
#print(v*np.sqrt(2))
print("nmax: ",n_max(a_X,m_X,m_phi))
print("n_bar: ", n_bar(a_X, m_X, m_phi))
#print(lmax)
#print(deltas[1])

def sigma_V_rec(bench_key):
    '''
    [sigma_rec] = GeV-2
    '''
    if not "3" in bench_key:
        sigmaVs = ([])
        for n in range(0, n_bar(a_X, m_X, m_phi)):
            n += 1
            for l in range(0, n):
                sigmaVs = np.append(sigmaVs, sigmaV_rec_numeric(n, l))
        print("sigma_rec/sigma_ann: ",sigmaVs[0]* anr.thermAvgSommerfeld(m_X, m_phi, a_X)/(anr.sigmaVtree(m_X, m_phi, a_X)* anr.thermAvgSommerfeld(m_X, m_phi, a_X)))
        return sum(sigmaVs)* anr.thermAvgSommerfeld(m_X, m_phi, a_X)
    else:
            sigma_ann = anr.sigmaVtree(m_X, m_phi, a_X)* anr.thermAvgSommerfeld(m_X, m_phi, a_X)
            print('sigma_ann: ',sigma_ann)
            print("sigma_rec/sigma_ann: ",64 / (3 * np.sqrt(3) * np.pi) * np.log(a_X / 2 * np.sqrt(m_X / m_phi)))
            return 64 / (3 * np.sqrt(3) * np.pi) * np.log(a_X / 2 * np.sqrt(m_X / m_phi)) * sigma_ann

def Crec():
    sigmaVrec = sigma_V_rec(bench_key)
    prefactor = (2 * np.pi) ** (3 / 2) * (anr.r_th(eos,m_X)) ** 3
    conversion = (1.52e24)  # GeV -> Sec^-1
    print("sigma_V_rec", sigmaVrec)
    return conversion / prefactor* sigmaVrec


print("Crec: ",Crec())
