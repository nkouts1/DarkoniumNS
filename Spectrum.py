import numpy as np
import dm_lib
import Annihilation_Rate as anr
import Capture_Rate
import Verification as ver
from scipy import integrate
import sys
import warnings
if not sys.warnoptions:
   warnings.simplefilter("ignore")


D_odot = 1.5e11 #meters
R_sol = 696340000 #meters

def decay_length(epsilon,m_phi,eos,E_phi):
    #P_R, Epsilon_R, B_R, r, n_R, dr = dm_lib.choose_Eos(eos)
    BR = 1
    return BR * (1.1e-9/epsilon)**2 * (E_phi/(m_phi*1e3) / 1000 ) * (100/(m_phi*1e3)) * R_sol #r[-1]*1e3

def k_p(m_X, m_phi, a_X,eos):
    return ver.Crec(m_X, m_phi, a_X,eos)/( 4*(ver.Crec(m_X, m_phi, a_X,eos) + anr.cAnn(eos, m_X, m_phi, a_X) ) )

def k_o(m_X, m_phi, a_X,eos):
    return 3*k_p(m_X, m_phi, a_X,eos)

def k_ann(m_X, m_phi, a_X,eos):
    return anr.cAnn(eos, m_X, m_phi, a_X)/( ver.Crec(m_X, m_phi, a_X,eos) +  anr.cAnn(eos, m_X, m_phi, a_X) )

#def delta(x, x0, sigma):
#    return np.exp(-(x - x0)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def delta(x, x0):
    if x == x0:
        return 1
    else:
        return 0

def dN_phi_ann(E_phi, m_X):
    return 2 * delta(E_phi,  m_X)

def dN_phi_p(E_phi, m_X, Delta):
    return 2 * delta(E_phi,  m_X + Delta/2)

def dN_phi_o(E_phi, m_X, m_phi, Delta):
    y = E_phi /  ( m_X - Delta / 2 )
    ymin = m_phi/(m_X - Delta/2)
    ymax = 1 - 3 * m_phi**2/(16 * m_X - 8* Delta)**2
    if ymin <= y <= ymax:
        A = 9 / ( 4 * ( np.pi**2 - 9 )* y**2 )
        numerator = y * (8 - 3* y) + (y+1)/(y - 2) * (y**2 - 6*y + 16)* np.log(1 - y)
        denominator = m_X - Delta/2 - m_phi - 3 * m_phi**2/( 4 * m_X - 2* Delta)
        return A * numerator/ denominator
    else:
        return 0

def dN_phi_rec(E_phi, Delta):
    return delta(E_phi,  Delta)


def Gamma_phi(m_X, m_phi, a_X,eos, E_phi, Delta):
    ortho = k_o(m_X, m_phi, a_X,eos) * dN_phi_o(E_phi, m_X, m_phi, Delta)
    para = k_p(m_X, m_phi, a_X,eos) * dN_phi_p(E_phi, m_X, Delta)
    ann = k_ann(m_X, m_phi, a_X, eos) * dN_phi_ann(E_phi, m_X)
    rec = (k_o(m_X, m_phi, a_X, eos) + k_p(m_X, m_phi, a_X, eos) ) * dN_phi_rec(E_phi, Delta)
    return Capture_Rate.Capture_Rate(eos, m_X) * ( ortho + para + ann + rec )


def v_plus(m_phi):
    m_e = 0.9314941e-3 #MeV/c2
    return (1 - 4 * m_e**2/ m_phi**2)**(1/2)

def v_phi(m_phi, E_phi):
    return (1 - m_phi**2 / E_phi**2 ) **(1/2)

def theta_d_max(E_d):
    return np.sqrt( (5.8**2/E_d) + 0.23**2 )

def phi_plus(m_X, m_phi, a_X,  Delta, epsilon):
    def integrand(E_phi):
        A = Gamma_phi(m_X, m_phi, a_X,eos, E_phi, Delta)/ ( 4 * np.pi * E_phi * v_plus(m_phi) * v_phi(m_phi, E_phi) * D_odot**2 )
        B = ( np.exp(- R_sol/decay_length(epsilon,m_phi,eos,E_phi))  - np.exp( - D_odot / decay_length(epsilon, m_phi, eos, E_phi) ))
        return A * B
    result = integrate.quad(integrand, 0, np.inf)[0]  # , args = (m_X, m_phi, a_X,  Delta, epsilon,))[0]
    return result

#def dN_plus(E_phi):
#    alpha = v_phi(m_phi, E_phi)/v_plus(m_phi)
#    if alpha >= 1:
#
#    else:
#
#    return 0




bench_key ="B1"

eos = 'L80'
m_X, a_X, m_phi, Delta = dm_lib.benchmark_values[bench_key]
P_R, Epsilon_R, B_R, r, n_R, dr = dm_lib.choose_Eos(eos)

#print(phi_plus(m_X, m_phi, a_X,  Delta, 1.6e-10))
