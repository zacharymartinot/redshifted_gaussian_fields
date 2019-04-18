import numpy as np
import numba
import ctypes

from scipy import LowLevelCallable, integrate, special
from astropy import cosmology

from .generator import GammaPowerSpectrum

lpmv_addr = numba.extending.get_cython_function_address("scipy.special.cython_special", "lpmv")
lpmv_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
c_lpmv = lpmv_functype(lpmv_addr)

@numba.vectorize('float64(float64, float64, float64)')
def njit_lpmv(m, v, x):
    return c_lpmv(m, v, x)

@numba.njit
def gamma(r1,r2,mu):
    return np.sqrt(r1**2. + r2**2. - 2.*r1*r2*mu)

@numba.njit
def Xi_term(g, alpha, beta):
#     pre_factor = beta/(2*np.pi**2. * alpha * (alpha+1.)) # the 2*np.pi**2 is canceled in the normalization
    pre_factor = 1./(alpha+1.)
    vphi = np.arctan(g/beta)

    f_1 = np.power(np.cos(vphi), alpha+2.)
    if vphi < 1e-10:
        const = alpha**3./6. + alpha**2. /2. + alpha/3.
        f_2 = alpha + 1. - const * vphi**2.
    else:
        f_2 = np.sin((alpha+1.)*vphi)/np.sin(vphi)

    Xi = pre_factor * f_1 * f_2
    return Xi

@numba.njit
def sum_integrand_terms(r, rp, nu, nup, mu, a, alpha, beta, sigma2):
    g = gamma(r,rp,mu)
    A = sigma2/np.sum(a)

    Xi_nu3_nup3 = 0.
    for i in range(len(a)):
        Xi_nu3_nup3 += a[i] * Xi_term(g, alpha[i], beta[i])

    Xi_nu3_nup3 *= A * np.power(nu*nup, 3.)
    return Xi_nu3_nup3

@numba.njit
def barXi_gauss_inner(r_i_axis, r_j_axis, nu_i_axis, nu_j_axis, mu, a, alpha, beta, sigma2, w, x):
    # note that the integrand is only symmetric when nu_i_axis == nu_j_axis
    Np = len(x)
    barXi = 0.
    for i in range(Np):
        for j in range(Np):
            nu_i = nu_i_axis[i]
            nu_j = nu_j_axis[j]
            r_i = r_i_axis[i]
            r_j = r_j_axis[j]

            Xi_ij = sum_integrand_terms(r_i, r_j, nu_i, nu_j, mu, a, alpha, beta, sigma2)
            barXi += w[i]*w[j]*Xi_ij
    barXi /= 2.**2. # two factors of 2 for the two integrals
    return barXi

@numba.njit
def nb_linear_interp(x, xp, fp):
    # find index of the xp element nearest to x
    idx0 = np.argmin(np.abs(x - xp))

    # find the next-nearest index, which then provides an interval
    # containing x
    if np.abs(xp[idx0+1] - x) < np.abs(xp[idx0-1] - x):
        idx1 = idx0 + 1
    else:
        idx1 = idx0 - 1

    if idx0 > idx1:
        idx_a = idx1
        idx_b = idx0
    else:
        idx_a = idx0
        idx_b = idx1

    indices = np.array([idx_a, idx_b], dtype=numba.int64)
    f = fp[indices]

    a,b = xp[indices]

    # the 1st-order Lagrange interpolation of f on [a,b],
    # evaluated at x
    res = (f[0]*(b - x) + f[1]*(x - a))/(b-a)
    return res

@numba.njit
def barXi(mu, nu, nup,
          a, alpha, beta, sigma2,
          w, x, del_nu, nu_k, r_k):
    nu_a = nu - del_nu/2.
    nu_b = nu + del_nu/2.

    nup_a = nup - del_nu/2.
    nup_b = nup + del_nu/2.

    nu_i_axis = del_nu/2. * x + nu
    nu_j_axis = del_nu/2. * x + nup

    r_i_axis = np.zeros_like(nu_i_axis)
    r_j_axis = np.zeros_like(nu_j_axis)
    for n in range(nu_i_axis.size):
        r_i_axis[n] = nb_linear_interp(nu_i_axis[n], nu_k, r_k)
        r_j_axis[n] = nb_linear_interp(nu_j_axis[n], nu_k, r_k)

    barXi = barXi_gauss_inner(r_i_axis, r_j_axis, nu_i_axis, nu_j_axis, mu,
                              a, alpha, beta, sigma2, w, x)
    return barXi

def integrand_constructor(a, alpha, beta, sigma2, w, x, del_nu, nu_k, r_k):
    @numba.njit
    def integrand_func(theta, *args):
        mu = np.cos(theta)
        ell = args[0]
        nu = args[1]
        nup = args[2]

        res = njit_lpmv(0,ell, mu) * barXi(mu, nu, nup,
                                           a, alpha, beta, sigma2,
                                           w, x, del_nu, nu_k, r_k)
        return res

    @numba.cfunc("float64(intc, CPointer(float64))")
    def c_integrand(n, xx):
        return integrand_func(xx[0], xx[1], xx[2], xx[3])

    return LowLevelCallable(c_integrand.ctypes)

def CAPS_block(cosmo, ell_axis, nu_arr, nup_arr,
               del_nu, a, alpha, beta, sigma2, Np=5, eps=1e-8, verbose=False):
    k_b = 1.38064852e-23 # joules/kelvin
    c = 299792458. # meters/second
    A_Jy = 1e26 # Jy / (Watt/meter^2/Hz)
    nu_e = 1420.4057 # MHz

    nu_arr = np.array(nu_arr)
    nup_arr = np.array(nup_arr)
    ell_axis = np.array(ell_axis)

    Hz4_per_MHz4 = ((1e6)**2.)**2. # corrects the 4 factors of nu (MHz) in the barXi integrand function `sum_integrand_terms()`
    barC_pre_factor = (2.*k_b * A_Jy / c**2. / nu_e)**2. * Hz4_per_MHz4

    x, w = special.roots_legendre(Np)

    nu_start = np.amin(np.r_[nu_arr, nup_arr]) - 2.*del_nu
    nu_end = np.amax(np.r_[nu_arr, nup_arr]) + 2.*del_nu

    nu_k = np.linspace(nu_start, nu_end, 1000, endpoint=True)
    z_k = nu_e/nu_k - 1.
    r_k = cosmo.comoving_distance(z_k).value

    epsabs = 0.
    epsrel = eps

    integrand_func = integrand_constructor(a, alpha, beta, sigma2, w, x, del_nu, nu_k, r_k)

    barC_flat = np.zeros((ell_axis.size, nu_arr.size), dtype=np.float)
    barC_err_flat = np.zeros((ell_axis.size, nu_arr.size), dtype=np.float)

    counts = range(10,100,10)
    for l in range(ell_axis.size):
        ell_l = ell_axis[l]

        if ell_l > 10**3:
            limit = 10*ell_l
        else:
            limit = 10**3


        if verbose and int(100.*float(l)/ell_axis.size) in counts:
            print "About", counts.pop(0), "% done."

        for i in range(nu_arr.size):
            nu_i = nu_arr[i]
            nup_i = nup_arr[i]

            quad_args = (ell_l, nu_i, nup_i)

            res, err = integrate.quad(integrand_func, 0, np.pi,
                                      args=quad_args,
                                      epsabs=epsabs,
                                      epsrel=epsrel,
                                      limit=limit,
                                      weight='sin',
                                      wvar=1.)

            barC_flat[l, i] = 2*np.pi * res
            barC_err_flat[l, i] = 2*np.pi * err

    barC_flat *= barC_pre_factor
    barC_err_flat *= barC_pre_factor

    return barC_flat, barC_err_flat

def mappable_CAPS_block_constructor(cosmo, ell_axis, del_nu, a, alpha, beta, sigma2, Np, eps):
    def mappable_CAPS_block(nu_nup, ell_axis=ell_axis, cosmo=cosmo, del_nu=del_nu,
                            a=a, alpha=alpha, beta=beta, sigma2=sigma2,
                            Np=Np, eps=eps):

        nu_arr = np.array([nu_nup[0]])
        nup_arr = np.array([nu_nup[1]])
        return generator2.CAPS_block(cosmo, ell_axis, nu_arr, nup_arr,
                                     del_nu, a, alpha, beta, sigma2, Np=Np, eps=eps)

    return mappable_CAPS_block
