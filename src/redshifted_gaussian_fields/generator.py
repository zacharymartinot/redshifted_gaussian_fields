import numpy as np
import numba as nb
import ctypes

from scipy import interpolate, linalg, special
from astropy import cosmology
import healpy as hp

import h5py
import os
import git

from redshifted_gaussian_fields import __path__ as rgf_path

def get_commit_hash():
    try:
        repo = git.Repo(rgf_path, search_parent_directories=True)
        commit_hash = np.bytes_(repo.head.object.hexsha)

    except git.NoSuchPathError:
        commit_hash = np.bytes_('none')

    return commit_hash

class Constants:
    k_b = 1.38064852e-23 # joules/kelvin. Boltzman's Constants
    c = 299792458. # meters/second. Speed of light
    A_Jy = 1e26 # Jy / (Watt/meter^2/Hz)
    nu_e = 1420.4057517667 # MHz. Hydrogen line rest-frame emission frequency

    Hz4_per_MHz4 = ((1e6)**2.)**2. # corrects the 4 factors of nu (MHz) in the barC integrand function
    barC_pre_factor = (2.*k_b * A_Jy / c**2. / nu_e)**2. * Hz4_per_MHz4

con = Constants()

def get_comoving_distance_approximation(cosmo, nu1_mhz, nu2_mhz, Npts=101):
    nu_e = con.nu_e

    nu_mhz = np.linspace(nu1_mhz, nu2_mhz, Npts, endpoint=True)
    z_nu = nu_e/nu_mhz - 1

    r_nu = cosmo.comoving_distance(z_nu)

    r_spl = interpolate.CubicSpline(nu_mhz, r_nu)
    return r_spl

# Power Spectrum Definition
######################################

def gaussian_component(k, k0):
    return np.exp(-(k/k0)**2. / 2.)

def gaussian_bump_component(k, k0):
    return 0.5*(k/k0)**2. * np.exp(-(k/k0)**2. /2.)

def gaussian_spectrum(k, k0, a, bump=False):
    k = np.atleast_1d(k)
    a = np.atleast_1d(a)
    k0 = np.atleast_1d(k0)
    P = np.zeros(k.size)

    if bump:
        term_func = gaussian_bump_component
    else:
        term_func = gaussian_component

    for ii in range(a.size):
        P += a[ii]*term_func(k, k0[ii])

    return P

def gaussian_bump_spectrum(k, k0, a):
    return gaussian_spectrum(k, k0, a, bump=True)

class ParameterizedGaussianPowerSpectrum:

    def __init__(self, a, k0, renormalization=(1., 1.), term_type='flat_gauss'):
        self.a = a
        self.k0 = k0
        self.term_type = term_type
        self.renormalization = renormalization

        if term_type == 'flat_gauss':

            self.eval_P = gaussian_spectrum

        elif term_type == 'bump_gauss':

            self.eval_P = gaussian_bump_spectrum

        # renormalization[0] is k point
        # renormalization[1] is amplitude
        self.norm = renormalization[1] / self.eval_P(renormalization[0], k0, a)

    def __call__(self, k):
        return self.norm*self.eval_P(k, self.k0, self.a)

def top_hat_component(k, k0, delta):
    idx = np.where((k-k0) < delta)

    t = np.zeros(k.size)
    t[idx] = 1.
    return t

def top_hat_spectrum(k, a, k0, delta):
    k = np.atleast_1d(k)
    a = np.atleast_1d(a)
    k0 = np.atleast_1d(k0)
    delta = np.atleast_1d(delta)

    P = np.zeros(k.size)
    for ii in range(a.size):
        P += a[ii] * top_hat_component(k, k0[ii], delta[ii])

    return P

class TopHatPowerSpectrum:

    def __init__(self, a, k0, delta, renormalization=(1.,1.)):
        self.a = a
        self.k0 = k0
        self.delta = delta

        self.term_type = 'top_hat'
        self.renormalization = renormalization

        # renormalization[0] is k point
        # renormalization[1] is amplitude
        self.norm = renormalization[1] / top_hat_spectrum(renormalization[0], a, k0, delta)

    def __call__(self, k):
        return self.norm * top_hat_spectrum(k, self.a, self.k0, self.delta)

# Cross-frequency angular power spectrum evaluation
####################################################

# jit-able exponetialy scaled modified bessel function of the first kind
ive_func_addr = nb.extending.get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1ive')
ive_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
ive_cfunc = ive_functype(ive_func_addr)

@nb.njit
def asym_ive(n, x):
    """
    First 5 terms of the asymtotic large-x expansion of the exponentially
    scaled modified bessel function of the first kind.
    """
    a1 = (4*n**2. - 1.)/8.
    a2 = (4*n**2. - 9.)/16.
    a3 = (4*n**2. - 25.)/24.
    a4 = (4*n**2. - 49.)/32.

    y1 = a1 / x
    y2 = a2 / x
    y3 = a3 / x
    y4 = a4 / x

    res = 1 - y1 + y1*y2 - y1*y2*y3 + y1*y2*y3*y4
    res /= np.sqrt(2* np.pi * x)

    return res

@nb.vectorize("float64(float64, float64)")
def ive(v, x):
    a1 = (4.*v**2. - 1)/8.
    if x > 1e2 * a1:

        a2 = (4*v**2. - 9.)/16.
        a3 = (4*v**2. - 25.)/24.
        a4 = (4*v**2. - 49.)/32.

        y1 = a1 / x
        y2 = a2 / x
        y3 = a3 / x
        y4 = a4 / x

        res = 1 - y1 + y1*y2 - y1*y2*y3 + y1*y2*y3*y4
        res /= np.sqrt(2* np.pi * x)
    else:
        res = ive_cfunc(v, x)
    return res

@nb.vectorize("float64(int64, float64, float64, float64)")
def gaussian_C(ell, r, rp, k0):
    y = (r-rp)**2.
    x = r*rp

    k0_sqr = k0**2.

    C = k0_sqr / np.sqrt(x) * np.exp(-k0_sqr * y /2.) * ive(ell+0.5, k0_sqr*x)
    return C

@nb.vectorize("float64(int64, float64, float64, float64)")
def gaussian_bump_C(ell, r, rp, k0):
    y = (r-rp)**2.
    x = r*rp

    k0_sqr = k0**2.

    f1 = (2*ell + 3 - k0_sqr*(r**2. + rp**2.)) * ive(ell+0.5, k0_sqr*x) + 2*k0_sqr*x * ive(ell+1.5, k0_sqr*x)

    C = 0.5 * k0_sqr / np.sqrt(x) * np.exp(-k0_sqr * y /2.) * f1
    return C


jv_func_addr = nb.extending.get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1jv')
jv_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
jv_cfunc = jv_functype(jv_func_addr)

@nb.vectorize("float64(float64, float64)")
def jv(v, x):
    return jv_cfunc(v, x)

@nb.njit
def inner_top_hat_C_not_equal(ell, r, rp, q):
    x = r*q
    xp = rp*q

    A = q**2. * r*rp / ((2*ell + 1) * (rp**2. - r**2.))

    B1 = jv(ell-0.5, x) * jv(ell+1.5, xp)
    B2 = jv(ell+1.5, x) * jv(ell-0.5, xp)

    J = A * (B1 - B2)

    return J

@nb.njit
def inner_top_hat_C_equal(ell, r, rp, q):
    x = r*q
    J = 0.5 * q**2. *( jv(ell+0.5, x)**2. - jv(ell-0.5, x) * jv(ell+1.5, x) )
    return J

@nb.vectorize("float64(int64, float64, float64, float64, float64)")
def top_hat_C(ell, r, rp, k0, delta):

    if np.abs(r - rp) < 2e16*max(r, rp):
        C = inner_top_hat_C_equal(ell, r, rp, k0 + delta) - inner_top_hat_C_equal(ell, r, rp, k0 - delta)
    else:
        C = inner_top_hat_C_not_equal(ell, r, rp, k0 + delta) - inner_top_hat_C_not_equal(ell, r, rp, k0 - delta)

    return C

def generate_inner_eval_bare_barC(term_type='flat_gauss'):
    if term_type == 'flat_gauss':
        C_func = gaussian_C
    elif term_type == 'bump_gauss':
        C_func = gaussian_bump_C
    else:
        raise ValueError("Input term_type not recognized.")

    @nb.njit
    def inner_eval_bare_barC(ell, r_i_axis, rp_j_axis, nu_i_axis, nup_j_axis, a, k0, w, x):
        Np = len(x)

        bare_barC = 0.
        for ii in range(Np):
            for jj in range(Np):
                nu_i = nu_i_axis[ii]
                nu_j = nup_j_axis[jj]
                r_i = r_i_axis[ii]
                r_j = rp_j_axis[jj]

                C_nu3_nup3 = 0.
                for nn in range(len(a)):
                    C_nu3_nup3 += a[nn] * C_func(ell, r_i, r_j, k0[nn])

                C_nu3_nup3 *= np.power(nu_i * nu_j, 3.)

                bare_barC += w[ii] * w[jj] * C_nu3_nup3

        bare_barC /= 4.

        return bare_barC

    return inner_eval_bare_barC

@nb.njit
def differance_threshold(eps, k0):
    if eps == 0.:
        res = np.inf
    else:
        res = np.sqrt(-2.*np.log(eps)/k0**2.)
    return res

def generate_eval_bare_barC(term_type='flat_gauss'):
    if term_type == 'flat_gauss':
        C_func = gaussian_C
    elif term_type == 'bump_gauss':
        C_func = gaussian_bump_C
    else:
        raise ValueError("Input term_type not recognized.")

    inner_eval_bare_barC = generate_inner_eval_bare_barC(term_type=term_type)

    @nb.njit(parallel=True)
    def eval_bare_barC(ell_axis, nu_sub_intervals, r_sub_intervals, a, k0, w, x, eps):

        Nnu = nu_sub_intervals.shape[0]
        Nell = ell_axis.shape[0]

        d_max = differance_threshold(eps, np.amin(k0))

        bare_barC = np.zeros((Nell, Nnu, Nnu), dtype=nb.float64)
        for nn in nb.prange(Nell):

            ell = ell_axis[nn]

            for ii in range(Nnu):
                nu_i_axis = nu_sub_intervals[ii]
                r_i_axis = r_sub_intervals[ii]

                for jj in range(ii, Nnu):
                    nu_j_axis = nu_sub_intervals[jj]
                    r_j_axis = r_sub_intervals[jj]

                    if d_max < np.inf:
                        d_ij = np.abs(np.mean(r_i_axis - r_j_axis))

                        if d_ij <= d_max:
                            bare_barC[nn, ii, jj] = inner_eval_bare_barC(ell, r_i_axis, r_j_axis, nu_i_axis, nu_j_axis, a, k0, w, x)

                    else:
                        bare_barC[nn, ii, jj] = inner_eval_bare_barC(ell, r_i_axis, r_j_axis, nu_i_axis, nu_j_axis, a, k0, w, x)

                    if ii != jj:
                        bare_barC[nn, jj, ii] = bare_barC[nn, ii, jj]

        return bare_barC

    return eval_bare_barC

# Generator
######################

@nb.njit
def realization(ell, ev, V, a_lm):
    Nnu = ev.shape[0]

    for m in range(ell + 1):
        idx = ell*ell + ell + m
        # r = np.zeros(Nnu, dtype=nb.complex128)
        if m == 0:
            rr = np.real(a_lm[:, idx])
            a_lm[:, idx] = np.dot(V, np.sqrt(ev)*rr)

        else:

            rr = np.real(a_lm[:,idx])
            ri = np.imag(a_lm[:,idx])

            a_lm[:, idx] = np.dot(V, np.sqrt(ev/2.)* rr) + 1j*np.dot(V, np.sqrt(ev/2.)* ri)

    for m in range(-ell, 0):
        neg_idx = ell*ell + ell + m
        pos_idx = ell*ell + ell + abs(m)
        a_lm[:, neg_idx] = (-1.0)**m * np.conj(a_lm[:, pos_idx])

    return

class GaussianCosmologicalFieldGenerator:

    def __init__(self, cosmology, Pspec, nu_axis, del_nu, ell_axis, Np=5, eps=0.):
        self.a = Pspec.a
        self.k0 = Pspec.k0
        self.term_type = Pspec.term_type
        self.normalization = Pspec.norm
        self.renormalization = Pspec.renormalization
        self.Pspec = Pspec

        self.Np = Np
        self.x, self.w = special.roots_legendre(self.Np)

        self.eps = eps

        self.cosmo = cosmology

        nu_start = 0.9*np.amin(nu_axis)
        nu_end = 1.1*np.amax(nu_axis)
        self.r_func = get_comoving_distance_approximation(cosmology, nu_start, nu_end)
        self.rdiv_func = self.r_func.derivative()

        self.nu_axis = nu_axis
        self.del_nu = del_nu
        self.ell_axis = ell_axis

        self.Nnu = len(nu_axis)
        self.Nell = len(ell_axis)

        self.eval_bare_barC = generate_eval_bare_barC(term_type=self.term_type)

        self.seeds = []

    def compute_cross_frequency_angular_power_spectrum(self):

        nu_a = self.nu_axis - self.del_nu/2.
        nu_b = self.nu_axis + self.del_nu/2.

        nu_sub_intervals = 0.5*(nu_a + nu_b)[:,None] + 0.5*(nu_b - nu_a)[:,None] * self.x[None,:]
        r_sub_intervals = self.r_func(nu_sub_intervals)

        self.barC = self.eval_bare_barC(self.ell_axis,
                                           nu_sub_intervals,
                                           r_sub_intervals,
                                           self.a,
                                           self.k0,
                                           self.w,
                                           self.x,
                                           self.eps)

        self.barC *= self.normalization * con.barC_pre_factor

    def compute_eigen_decomposition(self):

        if not hasattr(self, 'barC'):
            print('Covariance not yet set. Starting computation...')
            self.compute_cross_frequency_angular_power_spectrum()
            print('Covariance calculation done.')

        self.eig_vals = np.zeros(self.barC.shape[:2])
        self.eig_vecs = np.zeros(self.barC.shape)

        for nn in range(self.barC.shape[0]):
            self.eig_vals[nn], self.eig_vecs[nn] = linalg.eigh(self.barC[nn])

    def generate_realization(self, seed):
        """
        Returns a random realization of the spherical harmonic coefficients
        of the specific intensity on the sky with statistics defined by
        self.barC. The output data have units of Jy/sr in the observers inertial
        frame.
        """
        L = np.amax(self.ell_axis) + 1
        # a_lm = np.zeros((self.Nnu, L**2), dtype=np.complex128)
        np.random.seed(seed)
        r = np.random.randn(2*self.Nnu, L**2)
        a_lm = r[:self.Nnu] + 1j*r[self.Nnu:]

        if not hasattr(self, 'eig_vals'):
            print('Eigen-decomposition not yet set. Starting computation...')
            self.compute_eigen_decomposition()
            print('Eigen-decomposition done.')

        for ll in range(self.Nell):

            ell = self.ell_axis[ll]

            ev = self.eig_vals[ll]
            V = self.eig_vecs[ll]

            realization(ell, ev, V, a_lm)

        return a_lm

    def generate_healpix_map_realization(self, seed, nside):
        """
        Returns a random realization the specific intensity on the sky with
        statistics defined by self.barC, evaluated at HEALPix map pixel locations.
        The output data have units of Jy/sr in the observers inertial
        frame.
        """
        a_lm = self.generate_realization(seed)

        I_map = np.zeros((self.Nnu, 12*nside**2), dtype=float)
        for ii in range(self.Nnu):
            a_lm_hp = reindex_ssht2hp(a_lm[ii])
            I_map[ii] = hp.alm2map(a_lm_hp, nside, pol=False)

        return I_map

    def generate_healpix_map_realization_low_memeory(self, seed, nside, full_file_path, overwrite=False):
        """
        Alterative to generate_realization which uses intermediate I/O to control
        the maximum memory usage.
        """

        if os.path.exists(full_file_path) and not overwrite:
            raise ValueError("A file with that name already exists, and 'overwrite' is not set.")

        L = np.amax(self.ell_axis) + 1

        if not hasattr(self, 'eig_vals'):
            print('Eigen-decomposition not yet set. Starting computation...')
            self.compute_eigen_decomposition()
            print('Eigen-decomposition done.')


        # open file for both reading and writing, create it if it doesn't exist
        with h5py.File(full_file_path, 'a') as h5f:

            for key in ['alms', 'base_noise']:
                if key in h5f.keys():
                    del h5f[key]

            h5f.create_dataset('alms', (self.Nnu, L**2), dtype='c16')
            h5f.create_dataset('base_noise', (2*self.Nnu, L**2), dtype='f8')

            np.random.seed(seed)

            # this generates the white-noise realization in an ordering that will match
            # the 'generate_realization' method, but without having to hold the full
            # array in memory
            for ii in range(2*self.Nnu):
                r = np.random.randn(L**2)
                h5f['base_noise'][ii, :] = r

            for ll in range(self.Nell):

                ell = self.ell_axis[ll]

                # a_lm = np.random.randn(self.Nnu, 2*ell + 1) + 1j*np.random.randn(self.Nnu, 2*ell+1)
                idx = ell*ell + ell + np.arange(-ell, ell+1)
                r_here = h5f['base_noise'][:,idx]
                a_lm = r_here[:self.Nnu] + 1j*r_here[self.Nnu:]

                # h5f['a2'][:,idx] = np.copy(a_lm)

                ev = self.eig_vals[ll]
                V  = self.eig_vecs[ll]

                for m in range(ell + 1):

                    idx = ell + m

                    if m == 0:
                        rr = np.real(a_lm[:, idx])
                        a_lm[:, idx] = np.dot(V, np.sqrt(ev) * rr)

                    else:

                        rr = np.real(a_lm[:,idx])
                        ri = np.imag(a_lm[:,idx])

                        a_lm[:,idx] = np.dot(V, np.sqrt(0.5*ev) * rr) + 1j*np.dot(V, np.sqrt(0.5*ev) * ri)

                for m in range(-ell, 0):
                    neg_idx = ell + m
                    pos_idx = ell + abs(m)
                    a_lm[:, neg_idx] = (-1.0)**m * np.conj(a_lm[:, pos_idx])

                idx = ell*ell + ell + np.arange(-ell, ell+1)
                h5f['alms'][:,idx] = a_lm

            if 'healpix_maps' in h5f.keys():
                del h5f['healpix_maps']

            h5f.create_dataset('healpix_maps', (self.Nnu, 12*nside**2), dtype='f8')

            for ii in range(self.Nnu):

                a_lm = h5f['alms'][ii,:]

                a_lm_hp = reindex_ssht2hp(a_lm)

                I_map = hp.alm2map(a_lm_hp, nside, pol=False)

                h5f['healpix_maps'][ii,:] = I_map

            del h5f['alms']
    
    def save_covariance_data(self, full_file_path, write_cov_data=True, write_eig_data=False, overwrite=False):

        if not overwrite and os.path.exists(full_file_path):
            raise ValueError('A save file with that name already exists.')

        self.full_file_path = full_file_path

        commit_hash = get_commit_hash()

        if write_cov_data and not hasattr(self, 'barC'):
            self.compute_cross_frequency_angular_power_spectrum()

        if write_eig_data and not hasattr(self, 'eig_vals'):
            self.compute_eigen_decomposition()

        with h5py.File(full_file_path, 'w') as h5f:

            h5f.create_dataset('commit_hash', data=commit_hash)
            h5f.create_dataset('numpy_version', data=np.string_(np.__version__))

            cosmology_name = np.string_(self.cosmo.name)
            iparams = h5f.create_group('input_parameters')

            iparams.create_dataset('full_file_path', data=np.string_(full_file_path))

            iparams.create_dataset('cosmology_name', data=cosmology_name)

            iparams.create_dataset('nu_axis', data=self.nu_axis)
            iparams.create_dataset('del_nu', data=self.del_nu)
            iparams.create_dataset('ell_axis', data=self.ell_axis)

            # data that specifies the power spectrum
            iparams.create_dataset('a', data=self.a)
            iparams.create_dataset('k0', data=self.k0)
            iparams.create_dataset('renormalization_point', data=self.Pspec.renormalization[0])
            iparams.create_dataset('renormalization_amplitude', data=self.Pspec.renormalization[1])
            iparams.create_dataset('term_type', data=np.string_(self.term_type))

            iparams.create_dataset('Np', data=self.Np)
            iparams.create_dataset('eps', data=self.eps)

            realizations = h5f.create_group('realization_seeds')

            odata = h5f.create_group('output_data')

            if write_cov_data:
                odata.create_dataset('barC', data=self.barC)

            if write_eig_data:
                odata.create_dataset('eig_vals', data=self.eig_vals)
                odata.create_dataset('eig_vecs', data=self.eig_vecs)

    def save_realization_seeds(self, N=1, seeds=None, full_file_path=None):

        # generate random seeds which can be saved if they are not provided
        if seeds is None:
            seeds = np.random.randint(2**32 - 1, size=N)

        # save covariance data if the file doesn't already exist
        if not hasattr(self, 'full_file_path'):
            if full_file_path is None:
                raise ValueError("Save file path not specified.")
            else:
                # this sets full
                self.save_covariance_data(full_file_path, write_cov_data=False)

        elif full_file_path is not None and self.full_file_path != full_file_path:
            print("Save file already exists, ignoring input file path.")

        with h5py.File(self.full_file_path, 'r') as h5f:
            if not h5f['realization_seeds'].keys():
                max_idx = 0
            else:
                # the key for each seed is e.g. `seed15`
                max_idx = max([int(k.replace('seed', '')) for k in h5f['realization_seeds'].keys()])

        seed_indices = max_idx + 1 + np.arange(len(seeds))

        with h5py.File(self.full_file_path) as h5f:

            for idx, seed in zip(seed_indices, seeds):

                h5f['realization_seeds'].create_dataset('seed' + str(idx), data=seed)

        self.seeds += list(seeds)

    def delete_saved_data(self, delete_cov_data=True, delete_eig_data=True):
        """
        Deletes saved data products from `compute_cros_frequency_angular_power_spectrum`
        and/or `compute_eigen_decomposition`, but does not delete the file or
        the input parameters, so the delete data products can always be recomputed.
        """

        if not hasattr(self, 'full_file_path'):
            raise ValueError("This object does not have a corresponding save file.")

        with h5py.File(self.full_file_path, 'a') as h5f:

            if delete_cov_data and 'output_data/barC' in h5f:
                del h5f['output_data']['barC']

            if delete_eig_data and '/output_data/eig_vals' in h5f:
                del h5f['output_data']['eig_vals']
                del h5f['output_data']['eig_vecs']

def restore_from_file(full_file_path):

    with h5py.File(full_file_path, 'r') as h5f:
        iparams = h5f['input_parameters']
        cosmology_name = iparams['cosmology_name'][()].decode()
        cosmo = getattr(cosmology, cosmology_name)

        nu_axis = iparams['nu_axis'][()]
        del_nu = iparams['del_nu'][()]
        ell_axis = iparams['ell_axis'][()]

        a = iparams['a'][()]
        k0 = iparams['k0'][()]
        renorm0 = iparams['renormalization_point'][()]
        renorm1 = iparams['renormalization_amplitude'][()]
        renormalization = (renorm0, renorm1)
        term_type = iparams['term_type'][()].decode()

        Pspec = ParameterizedGaussianPowerSpectrum(a, k0, renormalization, term_type)

        Np = iparams['Np'][()]
        eps = iparams['eps'][()]

        gcfg = GaussianCosmologicalFieldGenerator(cosmo, Pspec, nu_axis, del_nu, ell_axis, Np=Np, eps=eps)

        gcfg.full_file_path = full_file_path

        odata = h5f['output_data']

        if 'barC' in odata.keys():
            gcfg.barC = odata['barC'][()]

        if 'eig_vals' in odata.keys():
            gcfg.eig_vals = odata['eig_vals'][()]
            gcfg.eig_vecs = odata['eig_vecs'][()]

        if h5f['realization_seeds'].keys():
            rseeds = h5f['realization_seeds']
            gcfg.seeds = [rseeds[k][()] for k in rseeds.keys()]

    return gcfg

def reindex_hp2ssht(hp_flm, lmax=None):
    if lmax is None:
        lmax = hp.Alm.getlmax(hp_flm.size)

    L = lmax + 1
    ssht_flm = np.zeros(L**2, dtype=np.complex128)
    for el in range(L):
        for m in range(-el, el+1):
            hp_idx = hp.Alm.getidx(lmax, el, abs(m))
            ssht_idx = el*el + el + m
            if m >= 0:
                ssht_flm[ssht_idx] = hp_flm[hp_idx]
            else:
                ssht_flm[ssht_idx] = (-1.)**m * np.conj(hp_flm[hp_idx])
    return ssht_flm

def reindex_ssht2hp(flm):
    L = int(np.sqrt(flm.size))
    lmax = L - 1
    hp_flm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)

    for el in range(L):
        for m in range(el+1):
            hp_idx = hp.Alm.getidx(lmax, el, abs(m))
            ssht_idx = el*el + el + m
            hp_flm[hp_idx] = flm[ssht_idx]

    return hp_flm
