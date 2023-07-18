import pandas as pd
import scipy.special as sps
import numpy as np


T_c = 1e5 #Kelvin

################################################
# Dark Matter Benchmark values
################################################

benchmark_values = {
    #The values are all in GeV, and are sorted as: m_X, a_X, m_phi, Delta
    "B1":  (400,0.011,10e-3,0.012),
    "B2a": (2000,0.04,5e-3,0.8),
    "B2b": (2000,0.04,15e-3,0.8),
    "B2c": (2000,0.04,70e-3,0.8),
    "B3a": (139000,0.54,2e-3,10133),
    "B3b": (139000,0.54,20e-3,10133),
    "B3c": (139000,0.54,100e-3,10133)
}


def choose_Eos(eos):
    # EoS data path
    EoS_path = 'C:\\Users\\nkout\\Desktop\\TOV\\EoSs\\Data\\' + eos + '_data.csv'
    # Load data from TOV.py
    EoS_data = pd.read_csv(EoS_path)
    P_R = EoS_data['P'].to_numpy()
    Epsilon_R = EoS_data['epsilon'].to_numpy()
    r = EoS_data['r'].to_numpy()
    n_R = EoS_data["n_R"].to_numpy()
    B_R = EoS_data["B_R"].to_numpy()
    dr = EoS_data["dr"].to_numpy()
    return P_R, Epsilon_R, B_R,r,n_R,dr

################################################
# Functions
################################################
def v0func(bench_key):
    m_X, a_X, m_phi, Delta = benchmark_values[bench_key]
    k =  8.617333262e-14 #GeV/Kelvin
    return np.sqrt(2*T_c*k/m_X)

def Kinetic(bench_key):
    m_X, a_X, m_phi, Delta = benchmark_values[bench_key]
    v = v0func(bench_key)
    return (v/(2*a_X))**2

# Bessel function of the first kind
def j_l(l,x):
    return sps.jn(l, x)

# Derivative of the Bessel function of the first kind
def j_l_derivative(l,x):
    return sps.jvp(l, x, 1)

# Bessel function of the first kind
def n_l(l,x):
    return sps.yvp(l,x)

# Derivative of the Bessel function of the second kind - Neumann function
def n_l_derivative(l,x):
    return sps.yvp(l,x,1)
