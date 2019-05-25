import numpy as np

import numba
from astropy import cosmology
from scipy import special, integrate, interpolate, optimize, linalg
from scipy.special import gamma as Gamma
import multiprocessing as mp
import h5py
import os, sys
import git
import time

import warnings

from redshifted_gaussian_fields import __path__ as rgf_path

def get_commit_hash():
    repo = git.Repo(rgf_path, search_parent_directories=True)
    commit_hash = np.string_(repo.head.object.hexsha)
    return commit_hash

class GammaPowerSpectrum(object):
    def __init__(self, a, alpha, beta, sigma2):
        self.a = a
        self.alpha = alpha
        self.beta = beta
        self.sigma2 = sigma2

        param_tuples = zip(self.a, self.alpha, self.beta)

        term_funcs = [lambda k, a_i=a_i,alpha_i=alpha_i,beta_i=beta_i: self.term(k, a_i, alpha_i, beta_i)
                      for (a_i, alpha_i, beta_i) in param_tuples]

        self.term_funcs = term_funcs
        self.pre_factor = 2.*np.pi**2. * self.sigma2/ np.sum(self.a)

    def __call__(self, k):
        return self.eval_P(k)

    def term(self, k, a_i, alpha_i, beta_i):
        Pi_k = a_i * np.power(beta_i, alpha_i+2.) / special.gamma(alpha_i + 2.) * k**(alpha_i-1.) * np.exp(-beta_i*k)
        return Pi_k

    def eval_P(self, k):

        terms = np.array([Pi(k) for Pi in self.term_funcs])

        P_k = self.pre_factor * np.sum(terms, axis=0)
        return P_k

class GammaPspecErrorEstimate(object):
    def __init__(self, a, alpha, beta, sigma2):
        self.a = a
        self.alpha = alpha
        self.beta = beta
        self.sigma2 = sigma2

        param_tuples = zip(self.a, self.alpha, self.beta)

        term_funcs = [lambda k, a_i=a_i,alpha_i=alpha_i,beta_i=beta_i: self.term(k, a_i, alpha_i, beta_i)
                      for (a_i, alpha_i, beta_i) in param_tuples]

        self.term_funcs = term_funcs

    def __call__(self, k, eps):
        terms = np.array([D_i(k) for D_i in self.term_funcs])
        D_k = eps * self.sigma2 * np.sum(terms, axis=0)
        return D_k

    def term(self, k, a, alpha, beta):
        f = 1./k * 4*np.pi * a * beta**2. / (alpha+1.) * np.sqrt(np.pi)/2. * Gamma((alpha-1.)/2.)/Gamma(alpha/2.)
        return f

class ComovingDistanceApproximation(object):
    """
    Creates an spline function that interpolates the comoving distance
    function of the input astropy.comology object over the frequency
    interval specified by [nu_start, nu_end].

    cosmology: an astropy.cosmology object
    nu_start: start of the frequency interval (MHz)
    nu_end: end of the frequency interval (MHz)
    nu_samples: number of nodes to use in the interpolation

    Usage:

    cosmo = astropy.cosmology.Planck15
    r_func = ComovingDistanceApproximation(100.,200., cosmo)
    nu_axis = np.linspace(130.,140.,101)
    r_nu = r_func(nu)

    """
    def __init__(self, nu_start, nu_end, cosmology, nu_samples=100):
        self.cosmo = cosmology # an astropy cosmology object
        self.nu1 = nu_start #MHz
        self.nu2 = nu_end #MHz
        self.nu_samples = nu_samples
        self.nu_nodes = np.linspace(nu_start, nu_end, nu_samples, endpoint=True)
        nu_21 = 1420.4057517667 # MHz
        self.z_nodes = (nu_21/self.nu_nodes) - 1.

        self.r_nodes = np.array(self.cosmo.comoving_distance(self.z_nodes), dtype=np.float)
        self.interpolant = interpolate.InterpolatedUnivariateSpline(self.nu_nodes, self.r_nodes)

    def __call__(self, nu):
        r_nu = self.interpolant(nu)
        return r_nu

class drdnuApproximation(object):
    def __init__(self, nu_start, nu_end, cosmology, nu_samples=100):
        self.cosmo = cosmology # an astropy cosmology object
        self.nu1 = nu_start #MHz
        self.nu2 = nu_end #MHz
        self.nu_samples = nu_samples
        self.nu_nodes = np.linspace(nu_start, nu_end, nu_samples, endpoint=True)
        nu_21 = 1420.4057517667 # MHz
        c = 299792458. # meters/second

        z_nodes = (nu_21/self.nu_nodes) - 1.
        Hz = self.cosmo.H(z_nodes) * 1e3 # meter / second / Mpc

        drdnu = c/Hz * (-nu_21/self.nu_nodes) * 1./(1e6*self.nu_nodes)

        self.drdnu_nodes = drdnu

        self.interpolant = interpolate.InterpolatedUnivariateSpline(self.nu_nodes, self.drdnu_nodes)

    def __call__(self, nu):
        drdnu = self.interpolant(nu)
        return drdnu

@numba.njit
def gamma(r1,r2,mu):
    return np.sqrt(r1**2. + r2**2. - 2.*r1*r2*mu)

@numba.njit
def Xi_term(g, alpha, beta):
#     pre_factor = beta/(2*np.pi**2. * alpha * (alpha+1.)) # the 2*np.pi**2 is canceled in the normalization
    pre_factor = 1./(alpha+1.)
    vphi = np.arctan(g/beta)

    f_1 = np.power(np.cos(vphi), alpha+2.)
    if np.all(vphi) < 1e-10:
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
def fixed_gauss_rule(mu, weights, P_ell_mu, nu1_axis, nu2_axis, r1_axis, r2_axis, a, alpha, beta, sigma2, w, x):
    N_mu = len(mu)
    res = 0.
    for n in range(N_mu):
        barXi_n = barXi_gauss_inner(r1_axis, r2_axis, nu1_axis, nu2_axis, mu[n], a, alpha, beta, sigma2, w, x)
        res += weights[n] * P_ell_mu[n] * barXi_n

    res *= 2.*np.pi
    return res


class MAPSBlock(object):
    def __init__(self, cosmology, nu_arr, nup_arr, del_nu, ell_axis, a_arr, alpha_arr, beta_arr, sigma2, Np=4, eps=1e-10):

        self.eps = eps
        self.Np = Np # number of points to use for the inner gaussian quadrature
        self.x, self.w = special.roots_legendre(self.Np)

        self.nu_arr = nu_arr
        self.nup_arr = nup_arr
        self.Nfreq_points = len(self.nu_arr)
        self.del_nu = del_nu

        self.ell_axis = ell_axis
        self.Nell = len(self.ell_axis)

        self.cosmo = cosmology
        nu_start = np.amin(np.r_[self.nu_arr, self.nup_arr]) - 2.*del_nu
        nu_end = np.amax(np.r_[self.nu_arr, self.nup_arr]) + 2.*del_nu

        self.r_func = ComovingDistanceApproximation(nu_start, nu_end, self.cosmo)

        self.a = a_arr
        self.alpha = alpha_arr
        self.beta = beta_arr
        self.sigma2 = sigma2

        self.mu_c = None

        self.barC = None

        k_b = 1.38064852e-23 # joules/kelvin
        c = 299792458. # meters/second
        A_Jy = 1e26 # Jy / (Watt/meter^2/Hz)
        nu_e = 1420.4057517667 # MHz

        Hz4_per_MHz4 = ((1e6)**2.)**2. # corrects the 4 factors of nu (MHz) in the barXi integrand function `sum_integrand_terms()`
        self.barC_pre_factor = (2.*k_b * A_Jy / c**2. / nu_e)**2. * Hz4_per_MHz4

    def set_integration_cutoffs(self):
        self.mu_c = np.zeros(self.Nfreq_points)

        with warnings.catch_warnings():
            # warnings.simplefilter("ignore")

            for i in range(self.Nfreq_points):
                nu_i = self.nu_arr[i]
                nup_i = self.nup_arr[i]

                f_mu_c = lambda mu: self.f_threshold(mu, nu_i, nup_i)

                if np.sign(f_mu_c(1.)*f_mu_c(-1.)) < 0.:
                    mu_c_i = optimize.brentq(f_mu_c, -1., 1.)
                else:
                    mu_c_i = 1.

                self.mu_c[i] = mu_c_i

    def barXi_gauss(self, mu, nu, nup): #, del_nu, a, alpha, beta, sigma2, w, x):
        nu_a = nu - self.del_nu/2.
        nu_b = nu + self.del_nu/2.
        nu_i_axis = (nu_b - nu_a)/2. * self.x + (nu_a + nu_b)/2.

        nup_a = nup - self.del_nu/2.
        nup_b = nup + self.del_nu/2.
        nu_j_axis = (nup_b - nup_a)/2. * self.x + (nup_a + nup_b)/2.

        r_i_axis = self.r_func(nu_i_axis)
        r_j_axis = self.r_func(nu_j_axis)

        barXi = barXi_gauss_inner(r_i_axis, r_j_axis, nu_i_axis, nu_j_axis, mu,
                                  self.a, self.alpha, self.beta, self.sigma2, self.w, self.x)
        return barXi

    def angular_integrand(self, mu, ell, nu, nup): #, del_nu, a, alpha, beta, sigma2, w, x, scale_factor):
        integrand = self.barXi_gauss(mu, nu, nup) * special.lpmv(0,ell,mu)
        return integrand

    def abs_angular_integrand(self, mu, nu, nup): #, del_nu, a, alpha, beta, sigma2, w, x, scale_factor):
        integrand = np.abs(self.barXi_gauss(mu, nu, nup))
        return integrand

    def integrate_abs_barXi(self, mu_c, nu, nup):
        # scale_factor = self.barXi_gauss(1., nu, nup)
        # scale_factor = np.abs(scale_factor)
        quad_args = (nu, nup)
        points = tuple(np.cos(np.linspace(np.arccos(mu_c), np.pi,20)))

        res, err = integrate.quad(self.abs_angular_integrand, -1., mu_c,
                                 args=quad_args,
                                 epsabs=0.,
                                 epsrel=self.eps,
                                 limit=10**3,
                                 points=points)

        return res

    def f_threshold(self, mu, nu, nup):
        norm = self.integrate_abs_barXi(1., nu, nup)
        f = self.integrate_abs_barXi(mu, nu, nup)/norm
        g = f - 50*np.finfo(np.float).eps
        return g

    def eval_barC(self, ell, nu, nup, mu_c):
        if mu_c == 1.: # the integral is effectively zero, so just return zero
            res = 0.
            err = 0.

        else:

            epsabs = self.eps
            epsrel = self.eps
            # limit = 10**3
            if ell > 10**3:
                limit = 10*ell
            else:
                limit = 10**3

            # theta_c = np.arccos(mu_c)
            # interval_frac = theta_c/np.pi
            #
            # zeros_est = int(ell*interval_frac) + 10
            #
            # points = tuple(np.flipud(np.cos(np.linspace(0.,theta_c, zeros_est, endpoint=True))))

            # points = (mu_c, np.cos((1.+np.arccos(mu_c))/2.) )
            # if ell == 50:
            #     print "Interval fraction is", interval_frac
            #     print len(points), np.arccos(points[0]),np.arccos(points[-1])

            if ell < 5:
                points = tuple(np.cos(np.linspace(0.,np.pi,20,)))
                # points = None
            else:
                # a very good approximation to the roots of the legendre polynomial
                aprx_roots = (1. - 1./(8.*ell**2.) + 1/(8.*ell**3.) ) * np.cos(np.pi * (4*np.arange(1,ell+1)-1.)/(4.*ell+2.))

                points = tuple(aprx_roots[aprx_roots >= mu_c])


            quad_args = (ell, nu, nup)

            res, err = integrate.quad(self.angular_integrand, -1., 1.,
                                    args=quad_args,
                                    epsabs=epsabs,
                                    epsrel=epsrel,
                                    limit=limit,
                                    points=points)

            res *= 2.*np.pi # /(2.*ell+1.)
            err *= 2.*np.pi # /(2.*ell+1.)

        return res, err

    def compute_barC(self):

        if self.mu_c is None:
            self.set_integration_cutoffs()
            # self.mu_c = -1.*np.ones_like(self.nu_arr)

        self.barC = np.zeros((self.Nell, self.Nfreq_points), dtype=np.float)
        self.barC_err = np.zeros((self.Nell, self.Nfreq_points), dtype=np.float)

        counts = range(10,100,10)
        for l in range(self.Nell):
            ell_l = self.ell_axis[l]
            # if int(100.*float(l)/self.Nell) in counts:
            #     print "About", counts.pop(0), "% done."

            for i in range(self.Nfreq_points):
                nu_i = self.nu_arr[i]
                nup_i = self.nup_arr[i]

                mu_c_i = self.mu_c[i]

                self.barC[l,i], self.barC_err[l,i] = self.eval_barC(ell_l, nu_i, nup_i, mu_c_i)

        self.barC *= self.barC_pre_factor
        self.barC_err *= self.barC_pre_factor

    def angular_integrand_ClC(self, theta, ell, nu, nup):
        mu = np.cos(theta)
        integrand = self.barXi_gauss(mu, nu, nup) * special.lpmv(0,ell,mu)
        return integrand

    def eval_barC_ClC(self, ell, nu, nup):

        # epsabs = self.eps
        epsabs = 0.
        epsrel = self.eps
        if ell > 10**3:
            limit = 10*ell
        else:
            limit = 10**3

        quad_args = (ell, nu, nup)

        # points = tuple(np.linspace(-np.pi,np.pi, ell+10, endpoint=True))
        # x,_ = roots_legendre(ell)
        # points = tuple(x)

        if ell == 0:
            points = None
        else:
            # a very good approximation to the roots of the legendre polynomial
            aprx_roots = (1. - 1./(8.*ell**2.) + 1/(8.*ell**3.) ) * np.cos(np.pi * (4*np.arange(1,ell+1)-1.)/(4.*ell+2.))

            points = tuple(np.arccos(aprx_roots))

        res, err = integrate.quad(self.angular_integrand_ClC, 0., np.pi,
                                    args=quad_args,
                                    epsabs=epsabs,
                                    epsrel=epsrel,
                                    limit=limit,
                                    points=points,
                                    weight='sin',
                                    wvar=1.,)
                                    # maxp1=limit)
        res *= 2.*np.pi
        err *= 2.*np.pi
        return res, err

    def compute_barC_ClC(self):

        self.barC = np.zeros((self.Nell, self.Nfreq_points), dtype=np.float)
        self.barC_err = np.zeros((self.Nell, self.Nfreq_points), dtype=np.float)

        counts = range(10,100,40)
        for l in range(self.Nell):
            ell_l = self.ell_axis[l]
            if int(100.*float(l)/self.Nell) in counts:
                print "About", counts.pop(0), "% done."

            for i in range(self.Nfreq_points):
                nu_i = self.nu_arr[i]
                nup_i = self.nup_arr[i]

                self.barC[l,i], self.barC_err[l,i] = self.eval_barC_ClC(ell_l, nu_i, nup_i)

        self.barC *= self.barC_pre_factor
        self.barC_err *= self.barC_pre_factor

    def compute_barC_fixed_gauss(self, ell_limit):
        self.barC = np.zeros((self.Nell, self.Nfreq_points), dtype=np.float)
        self.barC_err = np.zeros((self.Nell, self.Nfreq_points), dtype=np.float)

        nodes, weights = special.roots_legendre(ell_limit)
        counts = range(10,100,10)
        for l in range(self.Nell):
            if int(100.*float(l)/self.Nell) in counts:
                print "About", counts.pop(0), "% done."

            ell_l = self.ell_axis[l]
            P_ell_mu = special.lpmv(0,ell_l,nodes)

            for i in range(self.Nfreq_points):
                nu_i = self.nu_arr[i]
                nup_i = self.nup_arr[i]

                nu_a = nu_i - self.del_nu/2.
                nu_b = nu_i + self.del_nu/2.
                nu1_axis = (nu_b - nu_a)/2. * self.x + (nu_a + nu_b)/2.

                nup_a = nup_i - self.del_nu/2.
                nup_b = nup_i + self.del_nu/2.
                nu2_axis = (nup_b - nup_a)/2. * self.x + (nup_a + nup_b)/2.

                r1_axis = self.r_func(nu1_axis)
                r2_axis = self.r_func(nu2_axis)

                self.barC[l, i] = fixed_gauss_rule(nodes, weights, P_ell_mu,
                                                    nu1_axis, nu2_axis, r1_axis, r2_axis,
                                                    self.a, self.alpha, self.beta, self.sigma2, self.w, self.x)
        self.barC *= self.barC_pre_factor


def eval_barC_from_parameters(setup_parameters):
    common_parameters, nu_arr_block, nup_arr_block = setup_parameters
    cosmo, del_nu, ell_axis, a, alpha, beta, sigma2, Np, eps = common_parameters

    mapsb = MAPSBlock(cosmo, nu_arr_block, nup_arr_block, del_nu, ell_axis, a, alpha, beta, sigma2, Np=Np, eps=eps)
    mapsb.compute_barC_ClC()

    return mapsb.barC

class GaussianCosmologicalFieldGenerator(object):
    def __init__(self, cosmology, Pspec, nu_axis, del_nu, ell_axis, ell_down_sampling=5, Np=4, eps=1e-10):
        self.cosmo = cosmology

        # Pspec is a GammaPowerSpectrum instance
        self.a = Pspec.a
        self.alpha = Pspec.alpha
        self.beta = Pspec.beta
        self.sigma2 = Pspec.sigma2

        self.nu_axis = nu_axis
        self.Nfreq = len(nu_axis)
        self.del_nu = del_nu

        self.ell_axis = ell_axis
        self.Nell = len(self.ell_axis)

        self.ell_down_sampling = ell_down_sampling

        if self.ell_down_sampling > 1:
            self.ell_axis_d = np.append(self.ell_axis[::self.ell_down_sampling], self.ell_axis[-1]+1) # add the next-highest ell
        else:
            self.ell_axis_d = self.ell_axis

        self.Nell_d = len(self.ell_axis_d)

        self.Np = Np
        self.eps = eps

        self.barC = None
        self.eig_vals = None
        self.eig_vecs = None

        self.block_parameters = None

        self.overwrite = False

    def plan_MAPS_blocks(self, N_cpu='all'):
        # split frequency domain into blocks

        if N_cpu == 'all':
            N_cpu = mp.cpu_count()

        self.N_cpu = N_cpu
        nu_arr_inds, nup_arr_inds = np.triu_indices(self.Nfreq)
        self.nu_arr_inds = nu_arr_inds
        self.nup_arr_inds = nup_arr_inds
        nu_arr = self.nu_axis[nu_arr_inds]
        nup_arr = self.nu_axis[nup_arr_inds]

        self.nu_arr_blocks = np.array_split(nu_arr, nu_arr.size)
        self.nup_arr_blocks = np.array_split(nup_arr, nup_arr.size)

        common_parameters = (self.cosmo, self.del_nu, self.ell_axis_d, self.a, self.alpha, self.beta, self.sigma2, self.Np, self.eps)
        self.block_parameters = [(common_parameters, nu_arr_i, nup_arr_i) for nu_arr_i, nup_arr_i in zip(self.nu_arr_blocks, self.nup_arr_blocks)]

    def compute_MAPS(self):
        # use multiprocessing to compute blocks in parallel, then combine blocks

        start_time = time.time()

        if self.block_parameters == None:
            self.plan_MAPS_blocks()

        self.barC_init = np.zeros((self.Nell_d, self.Nfreq, self.Nfreq), dtype=np.float)

        pool = mp.Pool(processes=self.N_cpu)

        func = eval_barC_from_parameters

        blocks_list = pool.map(func, self.block_parameters)

        flat_barC = np.concatenate(blocks_list, axis=1)

        self.barC_init[:,self.nu_arr_inds, self.nup_arr_inds] = flat_barC
        self.barC_init[:,self.nup_arr_inds, self.nu_arr_inds] = flat_barC

        # interpolate in ell if needed
        if self.ell_down_sampling > 1:
            barC_init_interp = interpolate.interp1d(self.ell_axis_d, self.barC_init, kind='cubic', axis=0)
            self.barC = barC_init_interp(self.ell_axis)
        else:
            self.barC = self.barC_init

        end_time = time.time()
        print "Elapsed time:", end_time - start_time
    def decompose_barC(self):
        # compute eigenvalue decomposition
        self.eig_vals = np.zeros(self.barC.shape[:2])
        self.eig_vecs = np.zeros(self.barC.shape)
        for ll in range(self.barC.shape[0]):
            self.eig_vals[ll], self.eig_vecs[ll] = linalg.eigh(self.barC[ll])

    def save_data(self, full_file_name):

        if self.overwrite == False and os.path.exists(full_file_name):
            raise ValueError('A save file with that name already exists.')
        else:

            commit_hash = get_commit_hash()

            with h5py.File(full_file_name, 'w') as h5f:

                h5f.create_dataset('commit_hash', data=commit_hash)

                cosmology_name = np.string_(self.cosmo.name)
                iparams = h5f.create_group('input_parameters')

                iparams.create_dataset('full_file_name', data=np.string_(full_file_name))

                iparams.create_dataset('cosmology_name', data=cosmology_name)
                iparams.create_dataset('a', data=self.a)
                iparams.create_dataset('alpha', data=self.alpha)
                iparams.create_dataset('beta', data=self.beta)
                iparams.create_dataset('sigma2', data=self.sigma2)
                iparams.create_dataset('nu_axis', data=self.nu_axis)
                iparams.create_dataset('del_nu', data=self.del_nu)
                iparams.create_dataset('ell_axis', data=self.ell_axis)
                iparams.create_dataset('ell_down_sampling', data=self.ell_down_sampling)

                odata = h5f.create_group('output_data')
                odata.create_dataset('barC', data=self.barC)
                # odata.create_dataset('barC_err', data=self.barC_err)
                odata.create_dataset('eig_vals', data=self.eig_vals)
                odata.create_dataset('eig_vecs', data=self.eig_vecs)



            self.overwrite = False


#     def generate_sky_harmonics_spectrum(self):
        # return a realization of the a_lm(nu) for this gaussian field

#     def write_realization(self):
#         # save a realization with a unique id

def restore_from_file(full_file_name):
    with h5py.File(full_file_name, 'r') as h5f:
        iparams = h5f['input_parameters']
        cosmology_name = iparams['cosmology_name'].value
        cosmo = getattr(cosmology, cosmology_name)

        a = iparams['a'].value
        alpha = iparams['alpha'].value
        beta = iparams['beta'].value
        sigma2 = iparams['sigma2'].value
        nu_axis = iparams['nu_axis'].value
        del_nu = iparams['del_nu'].value
        ell_axis = iparams['ell_axis'].value
        ell_down_sampling = iparams['ell_down_sampling'].value

        Pspec = GammaPowerSpectrum(a, alpha, beta, sigma2)

        gcfg = GaussianCosmologicalFieldGenerator(cosmo, Pspec, nu_axis, del_nu, ell_axis, ell_down_sampling=ell_down_sampling)

        odata = h5f['output_data']
        gcfg.barC = odata.get('barC').value
        # gcfg.barC_err = odata.get('barC_err').value
        gcfg.eig_vals = odata.get('eig_vals').value
        gcfg.eig_vecs = odata.get('eig_vecs').value

    return gcfg

@numba.njit
def sh_realization_healpy_index(eig_vals, eig_vecs, seed):
    Nell, Nfreq = eig_vals.shape
    ell_max = Nell-1
    Nlm = ell_max*(ell_max+1)/2 + ell_max+1

    a_lm = np.zeros((Nlm, Nfreq), dtype=numba.complex128)

    np.random.seed(seed)

    for l in range(Nell):
        ev = eig_vals[l]
        V = eig_vecs[l]

        sqrt_half_ev = np.sqrt(ev/2.)

        for m in range(l+1):
            i = m * (2 * ell_max + 1 - m) // 2 + l

            r_re = np.random.randn(Nfreq)
            if m == 0:
                r_im = np.zeros(Nfreq, dtype=numba.float64)
                a_lm[i,:] += np.dot(V, np.sqrt(2.)*sqrt_half_ev*r_re)
                # a_lm[i,:] += 1j*np.dot(V, sqrt_half_ev*r_im)
            else:
                r_im = np.random.randn(Nfreq)
                a_lm[i,:] += np.dot(V, sqrt_half_ev*r_re)
                a_lm[i,:] += 1j*np.dot(V, sqrt_half_ev*r_im)

    a_lm[:,0]

    return a_lm

def pow_neg2_CAPS(cosmo, nu_axis, del_nu, A0, ell_max):
    nu_start = nu_axis[0] - 2*del_nu
    nu_end = nu_axis[-1] + 2*del_nu

    samples_per_MHz = 5./del_nu
    N_samples = int(samples_per_MHz * (nu_end - nu_start))

    r_func = ComovingDistanceApproximation(nu_start, nu_end, cosmo, nu_samples=N_samples)

    r_axis = r_func(nu_axis)

    ell_axis = np.arange(0,ell_max+1)

    Nell = ell_axis.size
    Nfreq = nu_axis.size

    @numba.njit
    def caps_inner(r_axis, ell_max):
        Nfreq = r_axis.size
        Nell = ell_max+1

        C_bare = np.zeros((Nell, Nfreq, Nfreq), dtype=numba.float64)

        for ell in range(Nell):
            for jj in range(Nfreq):
                for kk in range(Nfreq):
                    if jj >= kk:
                        r_j = r_axis[jj]
                        r_k = r_axis[kk]

                        # jj > kk is nu_j > nu_k, which means r_j < r_k
                        C_bare[ell, jj, kk] = 1./r_k * (r_j/r_k)**ell

                        if jj != kk:
                            C_bare[ell, kk, jj] = C_bare[ell, jj, kk]

            C_bare[ell] /= (2.*ell + 1.)

        return C_bare

    barC = A0 * caps_inner(r_axis, ell_max)

    k_b = 1.38064852e-23 # joules/kelvin
    c = 299792458. # meters/second
    A_Jy = 1e26 # Jy / (Watt/meter^2/Hz)
    nu_e = 1420.4057517667 # MHz

    Jyobs_per_Ksrc = A_Jy * 2 * k_b *(1e6*nu_axis/c)**2. * (nu_axis/nu_e)

    barC *= Jyobs_per_Ksrc[None,:,None] * Jyobs_per_Ksrc[None,None,:]

    return barC

def white_noise_variance(cosmo, nu_axis, del_nu, P0, eps='min'):
    """
    Observed variance in Jy^2/sr^2 of a flat-spectrum P(k) = P_0
    redshifted brightness field.

    """
    if eps == 'min':
        eps = 50.*np.finfo(np.float).eps

    nu_start = nu_axis[0] - 2*del_nu
    nu_end = nu_axis[-1] + 2*del_nu

    samples_per_MHz = 5./del_nu
    N_samples = int(samples_per_MHz * (nu_end - nu_start))

    drdnu_func = drdnuApproximation(nu_start, nu_end, cosmo, nu_samples=N_samples)

    r_func = ComovingDistanceApproximation(nu_start, nu_end, cosmo, nu_samples=N_samples)

    k_b = 1.38064852e-23 # joules/kelvin
    c = 299792458. # meters/second
    A_Jy = 1e26 # Jy / (Watt/meter^2/Hz)
    nu_e = 1420.4057517667 # MHz

    Hz7_per_MHz7 = (1e6)**7. # corrects the 7 factors of nu (MHz) in the integrand
    barC_pre_factor = (2.*k_b * A_Jy / c**2. / (1e6*nu_e))**2.
    barC_pre_factor /= (1e6*del_nu)**2.
    barC_pre_factor *= P0
    barC_pre_factor *= Hz7_per_MHz7

    barC_diag = np.zeros_like(nu_axis)

    # This integral is a bit pedantic, the relative difference
    # between the integral and the midpoint approximation is ~10^-4
    # in 100-200MHz for a channel width 0.1 MHz.
    # But it's a cheap calculation and by the gods we're gonna do it right.

    def integrand(nu):
        f = nu**6. / r_func(nu)**2. / np.abs(drdnu_func(nu))
        return f

    for i in range(nu_axis.size):

        nu_i = nu_axis[i]
        nu1 = nu_i - del_nu/2.
        nu2 = nu_i + del_nu/2.

        res, err = integrate.quad(integrand, nu1, nu2,
                                   epsabs=0.,
                                   epsrel=eps)
        # res = (nu2-nu1)*integrand(nu_i)
        barC_diag[i] = res

    barC_diag *= barC_pre_factor

    return barC_diag

@numba.njit
def white_noise_realization_hp_index(var_spectrum, ell_max, seed):
    L = ell_max+1
    Nfreq = var_spectrum.size
    Nlm = ell_max*(ell_max+1)/2 + ell_max+1

    a_lm = np.zeros((Nlm, Nfreq), dtype=numba.complex128)

    sqrt_half_var = np.sqrt(var_spectrum/2.)

    np.random.seed(seed)
    for l in range(L):
        for m in range(l+1):
            i = m * (2 * ell_max + 1 - m) // 2 + l

            r_re = np.random.randn(Nfreq)
            if m == 0:
                r_im = np.zeros(Nfreq, dtype=numba.float64)
                a_lm[i,:] = np.sqrt(2.)*sqrt_half_var * (r_re + 1j*r_im)
            else:
                r_im = np.random.randn(Nfreq)
                a_lm[i,:] = sqrt_half_var * (r_re + 1j*r_im)

    a_lm[0,:] = 0.
    return a_lm
