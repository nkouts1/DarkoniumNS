import numpy as np
import Annihilation_Rate as anr
import Capture_Rate
import dm_lib
import os
import csv

bench_keys = ["B1","B2a","B2b","B2c","B3a","B3b","B3c"]

###########################################
# Recombination cross section - analytic expression & Recombination Rate
###########################################

def sigma_rec(m_X, m_phi, a_X):
    '''
    [sigma_rec] = GeV-2
    '''
    sigma_ann = anr.sigmaVtree(m_X, m_phi, a_X) * anr.thermAvgSommerfeld(m_X, m_phi, a_X)
    return 64 / (3 * np.sqrt(3) * np.pi) * np.log(a_X / 2 * np.sqrt(m_X / m_phi)) * sigma_ann

def Crec(m_X, m_phi, a_X,eos):
    prefactor = (2 * np.pi) ** (3 / 2) * (anr.r_th(eos,m_X)) ** (3)
    conversion = (1.52e24)  # GeV -> Sec^-1
    return  conversion/prefactor * sigma_rec(m_X, m_phi, a_X) * anr.thermAvgSommerfeld(m_X, m_phi, a_X)

###########################################
# O- and P- states  Rates & Populations
###########################################
def C_o(m_X,a_X):
    return 0.06 * a_X ** 6 * m_X  # GeV

def q_o(m_X, m_phi, a_X,eos):
    conversion = (1.52e24)  # GeV -> Sec^-1
    return 3 * Crec(m_X, m_phi, a_X,eos) * Capture_Rate.Capture_Rate(eos, m_X) / (4 * ( Crec(m_X, m_phi, a_X,eos) + anr.cAnn(eos, m_X, m_phi, a_X)))/conversion  # GeV

def N_o(m_X, m_phi, a_X,eos):
    return q_o(m_X, m_phi, a_X,eos) / C_o(m_X,a_X)

def C_p(m_X,a_X):
    return a_X**5/2 * m_X

def q_p(m_X, m_phi, a_X,eos):
    return q_o(m_X, m_phi, a_X,eos)/3

def N_p(m_X, m_phi, a_X,eos):
    return q_p(m_X, m_phi, a_X,eos)/C_p(m_X,a_X)

###########################################
# NS Volume based on the EoS
###########################################

def Vol_th(eos,m_X):
    return 4 / 3 * np.pi * (anr.r_th(eos,m_X)) ** 3

def Vol_NS(eos):
    GeV2meters = 5.06e15  # GeV/m
    P_R, Epsilon_R, B_R, r, n_R, dr = dm_lib.choose_Eos(eos)
    return 4/3* np.pi * (r[-1]*1e3*GeV2meters) ** 3

###########################################
# Number of DM particles
###########################################

def N_X(m_X, m_phi, a_X,eos):
    return np.sqrt(Capture_Rate.Capture_Rate(eos, m_X) /(anr.cAnn(eos, m_X, m_phi, a_X) + Crec(m_X, m_phi, a_X, eos)) )

###########################################
# Steady State number density
###########################################
def n_D(m_X, m_phi, a_X,eos):
    return N_o(m_X, m_phi, a_X,eos)/ Vol_th(eos,m_X)

###########################################
# 1 - Darkonium ionization is negligible with respect to decay
###########################################

def sigma_ion(m_X, m_phi, a_X,Delta,T_c):
    return m_X ** 2 * anr.v_th(m_X,T_c) ** 2 / (4 * Delta ** 2) * sigma_rec(m_X, m_phi, a_X)

def lambda_ion(m_X, m_phi, a_X,Delta,eos,T_c):
    return 1/(sigma_ion(m_X, m_phi, a_X,Delta,T_c)*n_D(m_X, m_phi, a_X,eos))

def P_ion(m_X, m_phi, a_X,Delta,eos,T_c):
    return anr.r_th(eos,m_X)/lambda_ion(m_X, m_phi, a_X,Delta,eos,T_c)
###########################################
# 2 - Only ground state darkonium is appreciably populated in the Sun.
###########################################

def sigma_La(m_X,a_X):
    return 512 * np.pi / (3 * a_X ** 4 * m_X ** 2)  # GeV-2

def lambda_La(m_X, m_phi, a_X,eos):
    return 1/(sigma_La(m_X,a_X)*n_D(m_X, m_phi, a_X,eos))

def P_La(m_X, m_phi, a_X,eos):
    return anr.r_th(eos,m_X)/lambda_La(m_X, m_phi, a_X,eos)
###########################################
# 3 - Dark photons do not scatter before leaving the Sun.
###########################################
def sigma_C(m_X,a_X):
    return 8 * np.pi * a_X ** 2 / (3 * m_X ** 2)

def lambda_C(m_X, m_phi, a_X,eos):
    return 1/(sigma_C(m_X,a_X)*n_D(m_X, m_phi, a_X,eos))
def P_C(m_X, m_phi, a_X,eos):
    return anr.r_th(eos,m_X)/lambda_C(m_X, m_phi, a_X,eos)

###########################################
# 4 - DM self-capture is negligible when the DM population in the Sun is at its steady state value.
###########################################
def tau_X(m_X, m_phi, a_X, eos):
    return 1 / np.sqrt(Capture_Rate.Capture_Rate(eos, m_X)*(anr.cAnn(eos, m_X, m_phi, a_X) + Crec(m_X, m_phi, a_X, eos)))

###########################################
# Create a directory to save data
###########################################

def directory_data():
    '''
    Creates a directory in the given path.
    If the Directory EoSs\Data\eos doesn't exists, it creates it.
    '''
    path = 'C:\\Users\\nkout\\Desktop\\Thesis\\Codes\\'
    if not os.path.exists(path):
        os.makedirs(path)
    verification_data = path + '\\'
    return verification_data
###########################################
# Save data for a given EoS and Temperature
###########################################

def save_data(eos,T_c):
    '''
        Saves data in a csv file in the given directory.
    '''
    Lambda_ions = []
    Lambda_Las = []
    Lambda_Cs = []
    m_Xs = []
    a_Xs = []
    m_phis = []
    for bench_key in dm_lib.benchmark_values:
        m_X, a_X, m_phi, Delta = dm_lib.benchmark_values[bench_key]
        m_Xs = np.append(m_Xs,m_X/1000)
        a_Xs = np.append(a_Xs, a_X)
        m_phis = np.append(m_phis, m_phi*1000)
        r_th = anr.r_th(eos, m_X)
        Lambda_ions = np.append(Lambda_ions,r_th/lambda_ion(m_X, m_phi, a_X,Delta,eos,T_c))
        Lambda_Las = np.append(Lambda_Las,r_th/lambda_La(m_X, m_phi, a_X,eos))
        Lambda_Cs = np.append(Lambda_Cs,r_th/lambda_C(m_X, m_phi, a_X,eos))
    # Save data
    f = open(directory_data() + 'verification_data.csv', 'w')
    writer = csv.writer(f)
    header = ['Benchmark', 'm_X/TeV', 'a_X', 'm_phi/MeV', 'r_th/lambda_ion', 'r_th/lambda_La', 'r_th/lambda_C']
    verification_data = zip(bench_keys,m_Xs,a_Xs,m_phis,Lambda_ions,Lambda_Las,Lambda_Cs)
    writer.writerow(header)
    writer.writerows(verification_data)

#save_data('L80',1e5)
