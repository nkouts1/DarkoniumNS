import numpy as np
from scipy.integrate import solve_ivp
import scipy.special as sps
import csv
import os
import sys
import dm_lib
import warnings
if not sys.warnoptions:
   warnings.simplefilter("ignore")

# Create directory to save plots
path = 'C:\\Users\\nkout\\Desktop\\Thesis\\Codes\\Recombination\\'
if not os.path.exists(path):
    os.makedirs(path)

bench_key ="B1"
smaller = 1e-3
T_c = 1e5 #Kelvin
tolerance = 0.01

################################################
#    Relative velocity
################################################

def v0func(bench_key):
    m_X, a_X, m_phi, Delta = dm_lib.benchmark_values[bench_key]
    k =  8.617333262e-14 #GeV/Kelvin
    return np.sqrt(2*T_c*k/m_X)

def v01func(bench_key):
    m_X, a_X, m_phi, Delta = dm_lib.benchmark_values[bench_key]
    k = 8.617333262e-14  # GeV/Kelvin
    TCross = 1.5e7*k #Kelvin X GeV/Kelvin
    return np.sqrt(2*TCross/m_X)

def b(bench_key):
    m_X, a_X, m_phi, Delta = dm_lib.benchmark_values[bench_key]
    return a_X * m_X/m_phi
################################################
#    Define Potential Contributions
################################################
def Yukawa(bench_key,x):
    return (1/x)*np.exp(-x/ b(bench_key))

def Kinetic(bench_key):
    m_X, a_X, m_phi, Delta = dm_lib.benchmark_values[bench_key]
    v = v01func(bench_key)
    return (v/(2*a_X))**2

def Angular(l,x):
    return l*(l+1)/(x**2)

#Define Schrodinger Equation
def Schrodinger(x,y,l,bench_key):
    y1,y2 = y
    return [y2, (-Kinetic(bench_key)+ Angular(l,x) + Yukawa(bench_key,x) )*y1]

################################################
# Initial Conditions - Step 1
################################################

# x_i << b
def find_xi(bench_key,smaller):
    return b(bench_key)*smaller

#so we get Y0 for Step 1:
def initital_conditions(l,xi):
    return [1,(l+1)/xi]

################################################
# Radial Solution - Step 2
################################################

#Find x_m by the condition alpha**2 >> exp(-x_m/b)/x_m i.e at least 1e3 times lower
def find_xm(initial_x):
    x = np.linspace(initial_x, 1e9, 100000000)
    index = np.where(Yukawa(bench_key,x) < Kinetic(bench_key) *smaller)[0][0]
    xm = x[index]
    return xm

xi = find_xi(bench_key,smaller)
xm = find_xm(xi)
print(bench_key)
print("xi: ",xi)
print("xm: ",xm)
print("xnorm: ", 2*np.pi/np.sqrt(Kinetic(bench_key)))
print("Yukawa at xm: ",Yukawa(bench_key,xm))
print("Alpha: ", np.sqrt(Kinetic(bench_key)))
print("b: ",b(bench_key))

################################################
# Matching - Step 3
################################################

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

#Phase shift
def delta_l(l,x,x_l_derivative ,x_l):
    alpha = np.sqrt(Kinetic(bench_key))
    beta_l = x*x_l_derivative/x_l -1
    numerator = alpha*x*j_l_derivative(l,alpha*x) - beta_l *j_l(l,alpha*x)
    denominator = alpha*x*n_l_derivative(l,alpha*x) - beta_l *n_l(l,alpha*x)
    return np.arctan(numerator/denominator)

#Solve numerically the Schrodinger Equation
def nsolve(l,xi,xm):
    sol = solve_ivp(Schrodinger, t_span=(xi, xm), y0=initital_conditions(l,xi), t_eval=np.linspace(xi, xm, 10000000), args = (l,bench_key,), method='RK45')
    return sol.t, sol.y[1], sol.y[0]


################################################
# Compute Sigma_T - Step 4
################################################

def sigma_l(l,d_l_plus_1,d_l):
    return (l+1)*(np.sin(d_l_plus_1 - d_l))**2

def delta_l_final(bench_key):
    deltas = np.array([])
    xes = np.array([])
    lmax = 5000
    for l in range(lmax):
        percent_diff = np.inf
        i = 1
        xi = find_xi(bench_key,smaller)
        xm = find_xm(xi)
        previous_delta_l = delta_l(l, xm + 1000, nsolve(l, xi - 0.05, xm + 1000)[-1][1],
                                               nsolve(l, xi - 0.05, xm + 1000)[-1][2])
        while True:
            delta_new = delta_l(l, xm, nsolve(l, xi, xm)[-1][1], nsolve(l, xi, xm)[-1][2])
            percent_diff = abs(delta_new - previous_delta_l) / abs(previous_delta_l) * 100
            #print(f"xi = {xi}, xm = {xm}, delta_l = {delta_new}, percent_diff = {percent_diff}")

            if percent_diff < tolerance:
                print(
                    f"delta_{l} has converged to within 1%, after {i} iterations for xi = {xi} and xm = {xm}, and is equal to {delta_new}")
                xes = np.append(xes, xm)
                break

            if abs(delta_new) < abs(previous_delta_l):
                xm += 100
            else:
                xi -= 0.05
            previous_delta_l = delta_new
            i += 1
        deltas = np.append(deltas, delta_new)
        if abs(delta_new) < tolerance:
            lmax = l
            print(f"delta_{l} <0.01, so lmax = {lmax}.")
            break
    return deltas, xes, lmax

def directory_data():
    '''
    Creates a directory in the given path.
    '''
    path = 'C:\\Users\\nkout\\Desktop\\Thesis\\Codes\\Recombination\\Phase_shift'
    if not os.path.exists(path):
        os.makedirs(path)
    phase_shift_folder_data = path + '\\'
    return phase_shift_folder_data

def save_data(deltas,lmax):
    '''
    Saves data in a csv file in the given directory.
    In the file the parameters: l (up to lmax) and delta_l are saved.
    '''
    # Save data
    f = open(directory_data() + bench_key+'_phase_shift_data.csv', 'w')
    writer = csv.writer(f)
    header = ['l', 'delta_l']
    phase_shift_data = zip(np.arange(lmax+1),deltas)
    writer.writerow(header)
    writer.writerows(phase_shift_data)


deltas,xes,lmax = delta_l_final(bench_key)

save_data(deltas,lmax)


