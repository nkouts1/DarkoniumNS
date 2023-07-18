import dm_lib
from scipy import integrate
import numpy as np
import pandas as pd



#Temperature at the center of the NS: 10^5K
T_c = 1e5    #Kelvin
#Light speed
c = 2.998e5 #km/s

################################################
# Annihilation Rate of DM particles
################################################

################################################
# v_th at the center of a NS
################################################

def v_th(m_X,T_c):
	'''
	v_th(m_X,T_c)

	Returns the typical velocity of a dark matter particle with mass m_X at the center of the NS of temperature T_c

	[m_X] = GeV
	'''
	k =  8.617333262e-14 #GeV/K
	return np.sqrt(2*T_c*k/m_X)


################################################
# Tree-level annihilation cross section
################################################

def sigmaVtree(m_X, m_phi, a_X):
	'''
	sigmaVtree(m_X, m_A, alpha_X)

	Returns the tree-level annihilation cross section for massive dark photons fixed by relic abundance

	[m_X] = GeV
	[m_phi] = GeV
	'''
	numerator = (1 - (m_phi / m_X) ** 2) ** 1.5
	denominator = (1 - 0.5 * (m_phi / m_X) ** 2) ** 2
	prefactor = np.pi * (a_X / m_X) ** 2
	function = prefactor * numerator/denominator
	return function


################################################
# Sommerfeld Enhahcement
################################################

def sommerfeld(v, m_X, m_phi, a_X):
	'''
	sommerfeld(v, m_X, m_A, alpha_X)

	Returns the Sommerfeld enhancemement

	[m_X] = GeV
	[m_phi] = GeV
	'''
	a = v / (2 * a_X)
	c = 6 * a_X * m_X / (np.pi ** 2 * m_phi)
	# Kludge: Absolute value the argument of the square root inside Cos(...)
	function = np.pi/a * np.sinh(2*np.pi*a*c) / ( np.cosh(2*np.pi*a*c) - np.cos(2*np.pi*np.abs(np.sqrt(np.abs(c-(a*c)**2)) ) ) )
	return function

################################################
# Thermal Average Sommerfeld
################################################

def thermAvgSommerfeld(m_X, m_phi, a_X):
	'''
	thermAvgSommerfeld(m_X, m_phi, a_X):

	Returns the Thermally-averaged Sommerfeld enhancement

	[m_X] = GeV
	[m_A] = GeV
	'''
	v0 = v_th(m_X,T_c)

	def integrand(v):
		# We perform d^3v in spherical velocity space.
		# d^3v = v^2 dv * d(Omega)
		prefactor = 4*np.pi/(2*np.pi*v0**2)**(1.5)
		function = prefactor * v**2 * np.exp(-0.5*(v/v0)**2) * sommerfeld(v, m_X, m_phi, a_X)
		return function

	lowV = 0
	# Python doesn't like it when you integrate to infinity, so we integrate out to 10 standard deviations
	highV = 10*(v_th(m_X,T_c))

	integral = integrate.quad(integrand, lowV, highV)[0]
	return integral

################################################
# Thermal Radius
################################################

def r_th(eos,m_X):
	'''
		r_th(eos,m_X):

		Returns the thermal radius for a given mass of Dark matter and EoS of the NS.

		[m_X] = GeV
		[m_phi] = GeV
		'''
	#First make to natural units:
	k = 8.6173e-14 #GeV/K
	G = 6.71e-39  #GeV**-2
	conversion = 7.76239e-6 #MeV/fm**3 to GeV**4
	Geom_term =dm_lib.choose_Eos(eos)[2][0]*(dm_lib.choose_Eos(eos)[1][0]+ 3*dm_lib.choose_Eos(eos)[0][0])*conversion
	numerator = 9*T_c*k
	denominator= 4*np.pi*G * m_X * Geom_term
	return (numerator/denominator)**(1/2)

########################
#  cAnn
########################

def cAnn(eos,m_X,m_phi,a_X):
	'''
	cAnn(eos,m_X,m_phi,a_X)

	Returns the Annihilation rate in sec^-1 without Sommerfeld effects.
	To include sommerfeld effects, set thermAvgSomm = thermAvgSommerfeld(m_X, m_A, alpha_X)

	[m_X] = GeV
	[sigmaVTree] = GeV^-2
	'''
	prefactor = (2*np.pi)**(3/2)*(r_th(eos,m_X))**(3)
	conversion = (1.52e24) # GeV -> Sec^-1
	return conversion / prefactor * sigmaVtree(m_X, m_phi, a_X) * thermAvgSommerfeld(m_X, m_phi, a_X)
