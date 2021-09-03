import pytest

import numpy as np
import numba as nb

import h5py
import os

from astropy import cosmology

from redshifted_gaussian_fields import generator

@pytest.fixture(scope='session')
def fixed_test_parameters():
    k0 = np.logspace(-1.8,0.5,7)
    a = k0**(-2.5)

    Pspec = generator.ParameterizedGaussianPowerSpectrum(a, k0, renormalization=(0.5, 1.), term_type='flat_gauss')

    cosmo = cosmology.Planck15

    nu_axis = np.linspace(140., 150., 11)
    del_nu = 0.1

    ell_axis = np.arange(101)

    Np = 5

    eps = 0.

    return (cosmo, Pspec, nu_axis, del_nu, ell_axis, Np, eps)
