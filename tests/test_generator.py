import pytest

import numpy as np
import mpmath as mpm
import numba as nb
from redshifted_gaussian_fields import generator

def test_get_commit_hash():
    commit_hash = generator.get_commit_hash()

    assert isinstance(commit_hash, np.bytes_)

def test_get_comoving_distance_approximation():
    cosmo = generator.cosmology.Planck15

    r_func = generator.get_comoving_distance_approximation(cosmo, 10., 300.)

    nu_mhz = np.linspace(50.3, 250.8, 21)
    z_nu = generator.con.nu_e/nu_mhz - 1.

    r1 = cosmo.comoving_distance(z_nu).value
    r2 = r_func(nu_mhz)
    assert np.all(np.abs(r1 - r2)/r1 < 1e-8)

def test_gaussian_component():
    g = generator.gaussian_component(np.sqrt(2.), 1.)
    assert np.abs(g - 1/np.e) < 2e-16

def test_gaussian_bump_component():
    g = generator.gaussian_bump_component(np.sqrt(2.), 1.)
    assert np.abs(g - 1/np.e) < 2e-16

def test_gaussian_spectrum():
    a = np.array([1.,2.,3.])
    k0 = np.array([0.1,0.3,0.5])
    k = 0.4

    P1 = generator.gaussian_spectrum(k, k0, a)
    assert P1.size == 1

    P2 = 0
    for ii in range(3):
        P2 += a[ii]*generator.gaussian_component(k, k0[ii])

    assert P2 == P1[0]

    Pb1 = generator.gaussian_spectrum(k, k0, a, bump=True)
    assert Pb1.size == 1

    Pb2 = 0
    for ii in range(3):
        Pb2 += a[ii]*generator.gaussian_bump_component(k, k0[ii])

    assert Pb2 == Pb1[0]

    k = np.linspace(0.15,1.,6)

    P1 = generator.gaussian_spectrum(k, k0, a)
    assert P1.size == k.size

def test_ParameterizedGaussianPowerSpectrum():
    a = np.array([1.,2.])
    k0 = np.array([0.1,0.3])

    renorm_k = 0.2
    renorm_value = 2.

    Pspec = generator.ParameterizedGaussianPowerSpectrum(a, k0, renormalization=(renorm_k, renorm_value))

    assert np.abs(Pspec(0.2) - 2.) < 2e-16

    Pspec_b = generator.ParameterizedGaussianPowerSpectrum(a, k0, renormalization=(renorm_k, renorm_value), term_type='bump_gauss')

    assert np.abs(Pspec(0.31) - Pspec_b(0.31)) > 2e-16

def test_ive():

    def hp_ive(n,x):
        return mpm.exp(-x)*mpm.besseli(n, x)

    hp_ive = np.frompyfunc(mpm.autoprec(hp_ive), 2,1)

    ell_axis = np.linspace(0, 10**4, 1001)

    ive1 = generator.ive(ell_axis, 2e10)
    ive2 = np.array(list(map(np.float, hp_ive(ell_axis, 2e10))))

    assert np.all(np.abs(ive1 - ive2)/ive2 < 1e-15)

    x_ax = np.linspace(0, 1e3, 1001)
    x_ax += x_ax[0]

    ive1 = generator.ive(20.5, x_ax)
    ive2 = np.array(list(map(np.float, hp_ive(20.5, x_ax))))

    assert np.all(np.amax(np.abs(ive1 - ive2)) < 1e-15)

def test_gaussian_C():
    C = generator.gaussian_C(100, 9000., 9001., 0.1)
    assert type(C) is np.float64

def test_gaussian_bump_C():
    C = generator.gaussian_bump_C(100, 900., 9001., 0.1)
    assert type(C) is np.float64

def test_generate_inner_eval_bare_barC():
    ell = 20
    r_ax = np.linspace(9100.,9105,5)
    nu_ax = np.linspace(150,151,5)
    a = np.array([1.,2.])
    k0 = np.array([0.1,0.3])

    w = np.ones(nu_ax.size)
    x = np.linspace(-1., 1., 5)

    inner_eval_bare_barC = generator.generate_inner_eval_bare_barC()
    assert isinstance(inner_eval_bare_barC, nb.targets.registry.CPUDispatcher)

    barC = inner_eval_bare_barC(ell, r_ax, r_ax, nu_ax, nu_ax, a, k0, w, x)
    assert type(barC) is np.float64 or type(barC) is float

def test_generate_eval_bare_barC():

    ell_axis = np.arange(10)

    r_ax = np.linspace(9100.,9105,5)
    nu_ax = np.linspace(150,151,5)

    nu_sub_intervals = np.array([nu_ax, nu_ax])
    r_sub_intervals = np.array([r_ax, r_ax])

    a = np.array([1.,2.])
    k0 = np.array([0.1,0.3])

    w = np.ones(nu_ax.size)
    x = np.linspace(-1., 1., 5)

    eps = 0.

    eval_bare_barC =  generator.generate_eval_bare_barC(term_type='bump_gauss')
    assert isinstance(eval_bare_barC, nb.targets.registry.CPUDispatcher)

    barC = eval_bare_barC(ell_axis, nu_sub_intervals, r_sub_intervals, a, k0, w, x, eps)
    assert barC.shape == (10, 2, 2)
    assert type(barC[0,0,0]) is np.float64

def test_realization():
    L = 51
    Nnu = 5

    a_lm = np.zeros((Nnu, L**2), dtype=np.complex128)
    ell = 25

    np.random.seed(4300)
    ev = np.random.randn(Nnu)
    V = np.random.randn(Nnu, Nnu)

    generator.realization(ell, ev, V, a_lm)

    assert np.count_nonzero(a_lm) == Nnu*(2*ell + 1)

class TestGaussianCosmologicalFieldGenerator():

    @pytest.fixture(autouse=True)
    def _setup(self, fixed_test_parameters):
        cosmo, Pspec, nu_axis, del_nu, ell_axis, Np, eps = fixed_test_parameters

        self.gcfg = generator.GaussianCosmologicalFieldGenerator(cosmo, Pspec, nu_axis, del_nu, ell_axis, Np=Np, eps=eps)

    def test_compute_cross_frequency_angular_power_spectrum(self):
        gcfg = self.gcfg

        gcfg.compute_cross_frequency_angular_power_spectrum()
        Nnu = gcfg.nu_axis.size
        Nell = gcfg.ell_axis.size
        assert hasattr(gcfg, 'barC')
        assert gcfg.barC.shape == (Nell, Nnu, Nnu)

        # check the the (nu, nu') matrix is symmetric for a few ell
        for ll in range(0,Nell+1,10):
            assert np.all(np.abs(gcfg.barC[ll] - gcfg.barC[ll].T) == 0.0)

    def test_compute_eigen_decomposition(self):
        gcfg = self.gcfg

        gcfg.compute_eigen_decomposition()
        Nnu = gcfg.nu_axis.size
        Nell = gcfg.ell_axis.size

        assert hasattr(gcfg, 'eig_vals')
        assert gcfg.eig_vals.shape == (Nell, Nnu)
        assert hasattr(gcfg, 'eig_vecs')
        assert gcfg.eig_vecs.shape == (Nell, Nnu, Nnu)

        assert gcfg.eig_vals.size == np.count_nonzero(gcfg.eig_vals)

    def test_generate_realization(self):
        seed = 74839
        a_lm1 = self.gcfg.generate_realization(seed)
        a_lm2 = self.gcfg.generate_realization(seed)

        # these two arrays should be exactly (to-the-bit) equal
        assert np.all(a_lm1 == a_lm2)

    def test_save_covariance_data(self):
        pass

    def test_save_realization_seeds(self):
        pass

def test_run():
    pass
