import numpy as np
import dm_lib
################################################
# Calculate Capture Rate of DM particles
################################################
cm2km = 1e-5                #1cm is 1e-5 kms
c = 2.998e5                 #km/s
v_0 = 220                   #km/s
sigma_ref = 1.7e-50         #km2
rho_X = 0.3/cm2km**3        #GeV/c2 /km3


# Capture Rate
def Capture_Rate(eos,m_X):
    P_R, Epsilon_R, B_R, r, n_R, dr = dm_lib.choose_Eos(eos)
    # Calculate the integral by summing the data
    TOV_int = sum(r**2 * n_R * dr * (1 - B_R) / B_R)
    return 4 * np.sqrt(6*np.pi)* c ** 2 * rho_X/ m_X * sigma_ref / v_0  * TOV_int   #s-1  /(1e-45)

# Geometric limit capture rate
def Capture_Rate_geo(eos,m_X):
    P_R, Epsilon_R, B_R, r, n_R, dr = dm_lib.choose_Eos(eos)
    return np.sqrt(6*np.pi) * rho_X/m_X *r[-1]**2/v_0*(1 - B_R[-1])/B_R[-1] *c**2

