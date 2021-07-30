import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad


def solve_func(x, args):
	e=args[0]
	M=args[1]
	return np.float(x-e*np.sin(x)-M)


def phase_ecc_true(phase, e):
	'''
	Calculates the true anomaly from the phase
	'''
	mean_anomaly=phase*2*np.pi
	#print(mean_anomaly)
	E=fsolve(solve_func, x0=np.pi, args=np.array([e, mean_anomaly]))[0]
	
	true_anomaly = 2*np.arctan2(np.sqrt((1+e))*np.tan(E/2), np.sqrt((1-e)))
	if true_anomaly<0:
		true_anomaly+=2*np.pi
	return true_anomaly
	




#Spherical wind model, based on Williams et al. (1990)
def flux_williams_yf_delta(years, freq, period, year_peri, e, i, omega, xi, s, norm_flux, alpha):
	'''
	This function returns the flux in the spherical wind model per combination of date and frequency.
	eta = the momentum ratio between the two winds
	nu_ref = reference frequency, to make the results easier to interpret
	'''
	eta=0.44
	phase= (years-year_peri)/period
	f = phase_ecc_true_yf(phase, e)
	if (-np.pi/2<(omega+f)<np.pi/2) or (3*np.pi/2<(omega+f)<5*np.pi/2):
		omega=-omega
		f=-f
	r = (1-e**2)/(1+e*np.cos(f))*(1-eta**(1/2)/(1+eta**(1/2)))
	delta= 1+np.tan(omega+f)**2 + np.tan(i)**2
	tau=xi*((1/np.cos(i))/(2*delta*r**3*np.cos(omega+f)**3) *(np.sin(omega+f)*np.cos(omega+f)*np.tan(i) + ((1+np.tan(i)**2)/(np.sqrt(delta)))*np.arctan2(-np.sqrt(delta),(np.tan(omega+f)*np.tan(i))))*(freq/1.38e9)**(-2.1))
	flux=norm_flux*np.exp(-np.abs(tau))*r**s *(freq/1.38e9)**(alpha)
	return flux

def flux_spherical_wind(x, period, year_peri, e, i, omega, xi, s, norm_flux, alpha):
	'''
	Flux from the WCR that is free-free absorbed by a spherical wind, based on the model from Williams et al. (1990).
	x = array wiht each element consisting of the date in years and the frequency in Hz.
	period = Period of the system in years
	year_peri= most recent year of periastron passage, in years.
	e =  eccentricity of the orbit
	i = inclination of the orbit in rad
	omega = argument of periastron of the orbit in rad
	xi = a constant that depends on the electron density at the semi-major axis, squared
	s = power-law index of the response of the intrinsic synchrotron flux to changes in separation.
	norm_flux = normalisation constant for the flux, in Jy
	alpha = spectral index of the intrinsic synchrotron emission
	'''
	flux=np.zeros(len(x))
	for j in range(len(x)):
		flux[j] = flux_spherical_wind_element(x[j][0], x[j][1], period, year_peri, e, i, omega, xi, s, norm_flux, alpha)
	return flux



# Anisotropic wind model

def int_num_func(z, args):
	z_wcr=z
	x_wcr=args[0]
	y_wcr=args[1]
	c=args[2]
	i_wind=args[3]
	
	y_wcr_wind=y_wcr
	z_wcr_wind=z_wcr*np.cos(i_wind)-x_wcr*np.sin(i_wind)
	x_wcr_wind=x_wcr*np.cos(i_wind)+z_wcr*np.sin(i_wind)
	R_prime = np.sqrt(x_wcr_wind**2+y_wcr_wind**2)
	z_prime=z_wcr_wind
	return R_prime**(-7) * np.exp(-c*z_prime**2/(R_prime**3))**2


def int_gauss(x_wcr, y_wcr, z_wcr, i_wind, c):
	
	return quad(vertical_gaussian, z_wcr, np.inf, [x_wcr, y_wcr, c, i_wind])[0]

def flux_FFA_anisotropic_wind_element(year, freq, period, year_peri, e, i, omega, xi, s, norm_flux, alpha, c,i_wind, omega_wind):
	'''
	This function returns the flux in the anisotropic wind model per combination of date and frequency.
	eta = the momentum ratio between the two winds
	nu_ref = reference frequency, to make the results easier to interpret
	'''
	eta=0.44

	phase= (year-year_peri)/period
	f = phase_ecc_true_yf(phase, e)
	r = (1-e**2)/(1+e*np.cos(f))*(1-eta**(1/2)/(1+eta**(1/2)))

	z_ellipse=0
	x_ellipse=r*np.cos(omega-f)
	y_ellipse=r*np.sin(omega-f)

	x = x_ellipse*np.cos(-i) +z_ellipse*np.sin(-i)
	z = z_ellipse*np.cos(-i) -x_ellipse*np.sin(-i)
	y = y_ellipse

	x_rotated=x*np.cos(omega_wind) + y*np.sin(omega_wind)
	y_rotated=y*np.cos(omega_wind) - x*np.sin(omega_wind)
	z_rotated=z

	
	int_res = xi*int_indef_gauss_released_num(x_rotated, y_rotated,z_rotated, i_wind, c)
	tau = int_res*(freq/nu_ref)**(-2.1)
	flux=norm_flux*np.exp(-np.abs(tau))*r**s*(freq/nu_ref)**(alpha)

	return flux

def flux_FFA_anisotropic_wind(x, period, year_peri, e, i, omega, xi, s, norm_flux, alpha, c, i_wind, omega_wind):
	'''
	Flux from the WCR that is free-free absorbed by an anisotropic wind.
	x = array wiht each element consisting of the date in years and the frequency in Hz.
	period = Period of the system in years
	year_peri= most recent year of periastron passage, in years.
	e =  eccentricity of the orbit
	i = inclination of the orbit in rad
	omega = argument of periastron of the orbit in rad
	xi = a constant that depends on the electron density at the semi-major axis, squared
	s = power-law index of the response of the intrinsic synchrotron flux to changes in separation.
	norm_flux = normalisation constant for the flux, in Jy
	alpha = spectral index of the intrinsic synchrotron emission
	c = constant in the vertical Gaussian that depends on the scael height of the anisotropic wind
	i_wind = inclination of the anisotropic wind in rad
	omega_wind = angle over which to rotate the anisotropic wind around the z-axis in rad
	'''
	flux=np.zeros(len(x))

	for j in range(len(x)):
		flux[j] = flux_FFA_anisotropic_wind_element(x[j][0], x[j][1], period, year_peri, e, i, omega, xi, s, norm_flux, alpha, c, i_wind, omega_wind)
	return flux
