import numpy as np
from scipy.special import gamma

def power_spectrum_term(k, alpha, beta):
    f = k**(alpha-1) * np.exp(-beta*k)
    return f

def fiducial_power_spectrum(k, total_variance=1e-7):
    a1 = 1.
    a2 = 10.

    alpha1 = 7.
    alpha2 = 20.

    beta1 = 100. # Mpc/h
    beta2 = 100./3. # Mpc/h



    A1 = a1/(a1+a2) * np.power(beta1, alpha1+2.) / gamma(alpha1 + 2.)
    A2 = a2/(a1+a2) * np.power(beta2, alpha2+2.) / gamma(alpha2 + 2.)

    P_k = A1 * power_spectrum_term(k, alpha1, beta1) + A2 * power_spectrum_term(k, alpha2, beta2)

    A_tot = total_variance *(2*np.pi**2.)
    P_k *= A_tot

    return P_k
