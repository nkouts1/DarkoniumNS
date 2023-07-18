import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from matplotlib import interactive
import os
import dm_lib
import csv
import sys
import warnings
if not sys.warnoptions:
   warnings.simplefilter("ignore")

#Constants
#Dimensions: Length: km, Time: s, Mass: TeV

#Light speed
c = 2.998e5 #km/s
#Epsilon0 value
epsilon_0 = 1.60218e+29 #kg /km /s**2
#Neutron mass
m_n = 1.67492749804e-27  #kg
#Solar Mass
Mo = 1.988e30  #kg
#Gravitational Constant
G = 6.67430e-14  #km/kg/s**2


#v_0 = 220                   #km/s
#sigma_ref = 1.7e-55           #km**2
##m_X = 1                  #TeV
#rho_X = 3e11                #TeV/km**3



# Define equations of state
def lattimersoft(P): #HLPS-1
    return - 4743.2711 * (P ** (- 0.03313)) + 4885.28224 * (P ** (- 0.01027)) - 0.00535 * (P ** 2.19938) + 0.01101 * (
                P ** 2.09858)


def lattimerstiff(P): #HLPS-3
    return 81.56823 + 131.81073 * (1 - np.exp(-P / 4.41577)) + 924.14278 * (1 - np.exp(-P / 523.73573))


def lattimerintermediate(P): #HLPS-2
    return 161.55325 + 2777.74571 * (1 - np.exp(-P / 1909.97137)) + 172.85805 * (1 - np.exp(-P / 22.86436))


def nld(P):
    return 119.05736 + 304.80445 * (1 - np.exp(-P / 48.61465)) + 33722.34448 * (1 - np.exp(-P / 17499.47411))


def L80(P):
    return 5.97365 * (P ** 0.77374) + 89.24 * (P ** 0.30993)


def L95(P):
    return 15.54878 * (P ** 0.66635) + 76.70887 * (P ** 0.24734)


def ski4(P):
    return 105.72158 * (P ** 0.2745) + 4.75668 * (P ** 0.76537)


def ska(P):
    return 0.53928 * (P ** 1.01394) + 94.31452 * (P ** 0.35135)


def wff1(P):
    return 0.00127717 * (P ** 1.69617) + 135.233 * (P ** 0.331471)


def wff2(P):
    return 0.00244523 * (P ** 1.62692) + 122.076 * (P ** 0.340401)


def apr1(P):
    return 0.000719964 * (P ** 1.85898) + 108.975 * (P ** 0.340074)


def bl1(P):
    return 0.488686 * (P ** 1.01457) + 102.26 * (P ** 0.355095)


def bl2(P):
    return 1.34241 * (P ** 0.910079) + 100.756 * (P ** 0.354129)


def dh(P):
    return 39.5021 * (P ** 0.541485) + 96.0528 * (P ** 0.00401285)


def bgp(P):
    return 0.0112475 * (P ** 1.59689) + 102.302 * (P ** 0.335526)


def w(P):
    return 0.261822 * (P ** 1.16851) + 92.4893 * (P ** 0.307728)


def mdi1(P):
    return 4.1844 * (P ** 0.81449) + 95.00134 * (P ** 0.31736)


def mdi2(P):
    return 5.97365 * (P ** 0.77374) + 89.24 * (P ** 0.30993)


def mdi3(P):
    return 15.55 * (P ** 0.666) + 76.71 * (P ** 0.247)


def mdi4(P):
    return 25.99587 * (P ** 0.61209) + 65.62193 * (P ** 0.15512)


def hhj1(P):
    return 1.78429 * (P ** 0.93761) + 106.93652 * (P ** 0.31715)


def hhj2(P):
    return 1.18961 * (P ** 0.96539) + 108.40302 * (P ** 0.31264)


def ps(P):
    return 9805.95 * (1 - np.exp(0.000193624 * (-P))) + 212.072 * (1 - np.exp(0.401508 * (-P))) + 1.69483


def scvbb(P):
    return 0.371414 * P ** (1.08004) + 109.258 * P ** (0.351019)

#Library of EoSs
eos_library = {
    'L80': L80,
    'L95': L95,
    'SkI4': ski4,
    'Ska': ska,
    'WFF1': wff1,
    'WFF2': wff2,
    'APR1': apr1,
    'BL1': bl1,
    'BL2': bl2,
    'DH': dh,
    'BGP': bgp,
    'W': w,
    'MDI1': mdi1,
    'MDI2': mdi2,
    'MDI3': mdi3,
    'MDI4': mdi4,
    'HHJ1': hhj1,
    'HHJ2': hhj2,
    'SCVBB': scvbb
}


def Eps_of_P(P,eos,P0):
    '''
    Defines the function of Epsilon to Pressure, so that it can be used in the TOV function and solve the system.
    This function need the input of P0 (central pressure) and an eos name from the eos_library.
    '''
    P0 = initial(P0)[0]
    if P0 >= 0.184:
        e = eos_library[eos](P)  # Equation of state
    elif P0 >= 9.34375e-5:
        e = 0.00873 + 103.17338 * (1 - np.exp(-P / 0.38527)) + 7.34979 * (1 - np.exp(-P / 0.01211))
    elif P0 >= 4.1725e-8:
        e = 0.00015 + 0.00203 * (1 - np.exp(-P * 344827.5)) + 0.10851 * (1 - np.exp(-P * 7692.3076))
    elif P0 >= 1.44875e-11:
        e = 0.0000051 * (1 - np.exp(-P * 0.2373 * (10 ** 10))) + 0.00014 * (1 - np.exp(-P * 0.4021 * (10 ** 8)))
    else:
        e = 10 ** (31.93753 + 10.82611 * np.log10(P) + 1.29312 * (np.log10(P) ** 2) + 0.08014 * (np.log10(P) ** 3) + 0.00242 * (np.log10(P) ** 4) + 0.000028 * (np.log10(P) ** 5))
    return e


#Define TOV equations
def TOV(r, y,eos,P0):
    '''
    TOV system: dP/dr, dM/dr and dPhi/dr.
    '''
    M = y[1]
    P = y[0]
    e = Eps_of_P(P,eos,P0)
    dMdt = 11.2 * (10 ** (-6)) * (r ** 2) * e  # 1st order derivative of Mass
    dPdt = - 1.474 * (e * M / (r ** 2)) * (1 + P / e) * (1 + 11.2 * (10 ** (-6)) * (r ** 3) * P / M) * ((1 - 2.948 * M / r) ** (-1))  # 1st order derivative of Pressure
    dPhidt = 2*(1.474 * (M / (r ** 2)))*(1 + 11.2 * (10 ** (-6)) * (r ** 3) * P / M) * ((1 - 2.948 * M / r) ** (-1))
    dzdt = [dPdt,dMdt,dPhidt]  # array of derivatives
    return dzdt


#Initial Conditions
def initial(P0):
    '''
    Initial conditions of the NS.
    0: Central Pressure
    1: Mass near the center
    2: Phi at the center of the NS
    '''
    return [P0, 0.000000000001,0]

# Solve ode system
def solve_ode(eos,P0):
    '''
    Solves the TOV system with Runge - Kutta, given the initial conditions, for a given eos name.
    '''
    yol = solve_ivp(TOV, [0.0000001, 100], initial(1000), method='RK45', t_eval=np.linspace(0.0000001, 100, 1000000),args= (eos,P0,))
    return yol.t, yol.y[0], yol.y[1],yol.y[2]


def directory_plots(eos):
    '''
    Creates a directory in the given path.
    If the Directory EoSs\Plots\eos doesn't exists, it creates it.
    '''
    # Create directory to save plots
    path = 'C:\\Users\\nkout\\Desktop\\Thesis\\Codes\\EoSs\\Plots\\' + eos
    if not os.path.exists(path):
        os.makedirs(path)
    eos_folder = path + '\\'
    return eos_folder

def parameters(eos,P0):
    r = ([])
    P_R =([])
    M_R = ([])
    Phi_R =([])
    B_R = ([])
    Epsilon_R = ([])
    rho_R = ([])
    n_R = ([])
    ''''
    Parameters extracted from the ODE solution:
    r: Radius of the NS in km
    P_R: Pressure in MeV/fm**3 - r
    M_R: Mass of the NS in Mo - r
    Phi_R: Phi of the NS - r
    B_R: Exp(- Phi) - r
    Epsilon_R: Epsilon in MeV/fm**3 - r
    rho_R: Rho  in kg /km3
    n_R: Number density of the NS in 1/km**3 - r
    dr: change of r[i]
    '''
    r = solve_ode(eos,P0)[0]
    P_R = solve_ode(eos,P0)[1]
    M_R = solve_ode(eos,P0)[2]
    Phi_R = solve_ode(eos,P0)[3]
    B_R = np.exp(-Phi_R)
    Epsilon_R = Eps_of_P(P_R,eos,P0)
    rho_R = Eps_of_P(P_R,eos,P0) * epsilon_0 / c ** 2
    n_R = rho_R / m_n
    i = 1;
    dr = np.zeros(len(r))
    for i in range(len(r)):
        dr[i] = r[i] - r[i - 1]

    return r, P_R,M_R, Phi_R, B_R,Epsilon_R, rho_R, n_R, dr

def plot_Pressure(eos,P0):
    ''''
    Plots Log(Pressure) (MeV/fm**3) - Radius (km)
    '''
    r = parameters(eos,P0)[0]
    P_R = parameters(eos,P0)[1]
    plt.clf() #Clear data plotted in previous loop (see last for loop)
    plt.figure(1)
    plt.plot(r, P_R)
    plt.yscale('log')
    plt.ylabel('$P \;(MeV/fm^{3})$', fontsize=14)
    plt.xlabel('$R\;(km)$', fontsize=14)
    plt.title('EoS:' + ' ' + eos)
    plt.savefig(directory_plots(eos) + 'Pressure_Radius_' + eos + '.png')

def plot_Phi(eos,P0):
    ''''
    Plots Phi - Radius
    In the plot the Mass, Radius and Beta at r[-1] is also shown
    '''
    r = parameters(eos,P0)[0]  # Radius in km
    Phi_R = parameters(eos,P0)[3]  # Phi
    M_R = parameters(eos,P0)[2]  # Mass in Solar Masses
    Mf = M_R[-1]
    Rf = r[-1]
    plt.clf()  #Clear data plotted in previous loop (see last for loop)
    plt.figure(2)
    plt.plot(r, Phi_R)
    plt.ylabel('$\\Phi(R)$', fontsize=14)
    plt.xlabel('$R\;(km)$', fontsize=14)
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3 # create a list with two empty handles (or more if needed)
    labels = []
    labels.append("M = %.2f" % Mf + '$M_{\odot}$')
    labels.append("R = %.2f" % Rf + 'km')
    labels.append("$\\Phi(R)$ = %.2f" % Phi_R[-1])
    plt.legend(handles, labels, loc='best', fontsize='small', fancybox=True, framealpha=0.7, handlelength=0,
               handletextpad=0)
    plt.title('EoS:' + ' ' + eos)
    plt.savefig(directory_plots(eos) + 'Phi_Radius_' + eos + '.png')

def plot_B_R(eos,P0):
    '''
    Plots Beta - Radius
    In the plot the Mass, Radius and Beta at r[-1] is also shown
    '''
    r = parameters(eos,P0)[0]  # Radius in km
    Phi_R = parameters(eos,P0)[3]  # Phi
    B_R = np.exp(-Phi_R)  # B = Exp(- Phi)
    M_R = parameters(eos,P0)[2]  # Mass in Solar Masses
    Mf = M_R[-1]
    Rf = r[-1]
    # B
    plt.clf()  #Clear data plotted in previous loop (see last for loop)
    plt.figure(3)
    plt.plot(r, B_R)
    plt.ylabel('$B(R)$', fontsize=14)
    plt.xlabel('$R\;(km)$', fontsize=14)
    # create a list with two empty handles (or more if needed)
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3
    labels = []
    labels.append("M = %.2f" % Mf + '$M_{\odot}$')
    labels.append("R = %.2f" % Rf + 'km')
    labels.append("B(R) = %.3f" % B_R[-1])
    plt.legend(handles, labels, loc='best', fontsize='small', fancybox=True, framealpha=0.7, handlelength=0,
               handletextpad=0)
    plt.title('EoS:' + ' ' + eos)
    plt.savefig(directory_plots(eos) + 'Beta_Radius_' + eos + '.png')

def plot_Epsilon(eos,P0):
    '''
    Plots Epsilon - Radius
    '''
    r = parameters(eos,P0)[0]
    P_R = parameters(eos,P0)[1]
    Epsilon_R = Eps_of_P(P_R, eos,P0)
    plt.clf()  #Clear data plotted in previous loop (see last for loop)
    plt.figure(4)
    plt.plot(r, Epsilon_R)
    plt.yscale('log')
    plt.ylabel('$\epsilon(MeV/fm^{3})$', fontsize=14)
    plt.xlabel('$R\;(km)$', fontsize=14)
    plt.title('EoS:' + ' ' + eos)
    plt.savefig(directory_plots(eos) + 'Epsilon_Radius_' + eos + '.png')

def plot_rho(eos,P0):
    '''
    Plots Rho - Radius
    '''
    r = parameters(eos,P0)[0]
    P_R = parameters(eos,P0)[1]
    rho_R = Eps_of_P(P_R, eos,P0) * epsilon_0 / c ** 2
    plt.clf()  #Clear data plotted in previous loop (see last for loop)
    plt.figure(5)
    plt.plot(r, rho_R)
    plt.yscale('log')
    plt.ylabel('$\\rho(kg /km^-3)$', fontsize=14)
    plt.xlabel('$R\;(km)$', fontsize=14)
    plt.title('EoS:' + ' ' + eos)
    plt.savefig(directory_plots(eos) + 'Rho_Radius_' + eos + '.png')

def plot_n_R(eos,P0):
    '''
    Plots Number Density - Radius
    '''
    r = parameters(eos,P0)[0]
    P_R = parameters(eos,P0)[1]
    rho_R = Eps_of_P(P_R, eos,P0) * epsilon_0 / c ** 2
    n_R = rho_R / m_n
    plt.clf()  #Clear data plotted in previous loop (see last for loop)
    plt.figure(6)
    plt.plot(r, n_R)
    plt.yscale('log')
    plt.ylabel('$n(km^{-3})$', fontsize=14)
    plt.xlabel('$R\;(km)$', fontsize=14)
    plt.title('EoS:' + ' ' + eos)
    plt.savefig(directory_plots(eos) + 'Number-Density_Radius' + eos + '.png')

def plot_Mass(eos,P0):
    '''
    Plots Mass - Radius
    In the plot the Mass and Radius at r[-1] is also shown
    '''
    r = parameters(eos,P0)[0]  # Radius in km
    M_R = parameters(eos,P0)[2]  # Mass in Solar Masses
    Mf = M_R[-1]
    Rf = r[-1]
    # Mass
    plt.clf()  #Clear data plotted in previous loop (see last for loop)
    plt.figure(7)
    plt.plot(r, M_R)
    plt.ylabel('$M \; (M_{\odot})$', fontsize=14)
    plt.xlabel('$R\;(km)$', fontsize=14)
    # create a list with two empty handles (or more if needed)
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 2
    labels = []
    labels.append("M = %.2f" % Mf + '$M_{\odot}$')
    labels.append("R = %.2f" % Rf + 'km')
    plt.legend(handles, labels, loc='best', fontsize='small', fancybox=True, framealpha=0.7, handlelength=0,
               handletextpad=0)
    plt.title('EoS:' + ' ' + eos)
    # plt.yscale('log')
    plt.savefig(directory_plots(eos) + 'Mass_Radius_' + eos + '.png')
    # plt.show()

def plot_sol(eos,P0):
    '''
    Plots figures 1 to 7 and saves it to a folder in EoSs/Plots with the name of the EoS
    '''
    plot_Pressure(eos,P0)
    plot_Phi(eos,P0)
    plot_B_R(eos,P0)
    plot_Epsilon(eos,P0)
    plot_rho(eos,P0)
    plot_n_R(eos,P0)
    plot_Mass(eos,P0)
    interactive(False)
    plt.close()  # save plots in folders, doesn't show them

def directory_data():
    '''
    Creates a directory in the given path.
    If the Directory EoSs\Data\eos doesn't exists, it creates it.
    '''
    path = 'C:\\Users\\nkout\\Desktop\\Thesis\\Codes\\EoSs\\Data'
    if not os.path.exists(path):
        os.makedirs(path)
    eos_folder_data = path + '\\'
    return eos_folder_data


def save_data(eos,P0):
    '''
    Saves data in a csv file in the given directory.
    In the file the parameters:
    Pressure, Epsilon, Radius, Beta, Number Density and dr
    are saved, so that they can be used to calculate the Capture Rate of DM particles.
    '''
    # Save data
    f = open(directory_data() + eos + '_data.csv', 'w')
    writer = csv.writer(f)
    header = ['P', 'epsilon', 'r', 'B_R', 'n_R', 'dr']
    eos_data = zip(parameters(eos,P0)[1], parameters(eos,P0)[5], parameters(eos,P0)[0], parameters(eos,P0)[4], parameters(eos,P0)[7], parameters(eos,P0)[8])
    writer.writerow(header)
    writer.writerows(eos_data)


