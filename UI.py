import numpy as np
import Annihilation_Rate as anr
import Capture_Rate
import dm_lib
import TOV
import Verification as ver

GeV2meters = 5.06e15   #GeV/m
GeV2sec = 1.52e24      #GeV/s
sec2yr = 3.16887646e-8

################################################################################################
# Neutron Star EoS, Temperature and Initial Pressure
################################################################################################

P_0 = 1000                 #MeV/fm-3
T_c = 1e5                  #Kelvin

#Library of EoSs
#'L80','L95','SkI4', 'NLD', 'Ska','WFF1','WFF2','APR1', 'BL1', 'BL2','DH','BGP','W',
# 'MDI1', 'MDI2','MDI3','MDI4','HHJ1','HHJ2','PS', 'SCVBB'

eos = 'SkI4'

################################################################################################
# Solve TOV equations
################################################################################################
TOV.plot_sol(eos, P_0)
TOV.save_data(eos, P_0)

################################################################################################
# Choose Benchmark
################################################################################################

bench_key ="B3c"

m_X, a_X, m_phi, Delta = dm_lib.benchmark_values[bench_key]
P_R, Epsilon_R, B_R, r, n_R, dr = dm_lib.choose_Eos(eos)

#r_th = anr.r_th(eos,m_X)
print("Benchmark: ", bench_key, " EoS: ", eos)
print("r_th(m): {:.2e}".format(anr.r_th(eos,m_X)/GeV2meters))
print("r_th/R_NS: {:.2e}".format(anr.r_th(eos,m_X)/GeV2meters /(r[-1]*1e3)))
print("N_X: {:.2e}".format(ver.N_X(m_X, m_phi, a_X, eos)))
print("N_o: {:.2e}".format(ver.N_o(m_X, m_phi, a_X, eos)))
print("N_p: {:.2e}".format(ver.N_p(m_X, m_phi, a_X, eos)))
print("Ccap: {:.2e}".format(Capture_Rate.Capture_Rate(eos, m_X)))
print("Ccap_geo: {:.2e}".format(Capture_Rate.Capture_Rate_geo(eos, m_X)))
print("SigmaVtree_ann: {:.2e}".format(anr.sigmaVtree( m_X, m_phi, a_X)))
print("Cann: {:.2e}".format(anr.cAnn(eos, m_X, m_phi, a_X)))
print("Crec: {:.2e}".format(ver.Crec(m_X, m_phi, a_X,eos)))
print("n_D(m-3): {:.2e}".format(ver.n_D(m_X, m_phi, a_X,eos)/GeV2meters**3))
print("P_ion: {:.2e}".format(ver.P_ion(m_X, m_phi, a_X,Delta,eos,T_c)))
print("P_La: {:.2e}".format(ver.P_La(m_X, m_phi, a_X,eos)))
print("P_C: {:.2e}".format(ver.P_C(m_X, m_phi, a_X,eos)))
print("tau_X(yr): {:.2e}".format(ver.tau_X(m_X, m_phi, a_X, eos)*sec2yr))
