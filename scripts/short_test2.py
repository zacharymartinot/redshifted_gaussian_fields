import numpy as np
from astropy import cosmology

# import sys
# sys.path.append('/users/zmartino/zmartino/gaussian_field_generation/')
from gaussian_field_generation import generator

a = np.array([1., 10.])
alpha = np.array([7., 20.])
beta = np.array([100., 100./3.])
sigma2 = 1e-6

Pspec = generator.GammaPowerSpectrum(a, alpha, beta, sigma2)

cosmo = cosmology.Planck15
full_nu_axis = np.linspace(100.,200.,1024,endpoint=True)
del_nu = np.diff(full_nu_axis)[0]
nu_axis = full_nu_axis[512-192:512+192]

ell_max = 1000
ell_axis = np.arange(0,ell_max+1)

gcfg = generator.GaussianCosmologicalFieldGenerator(cosmo, Pspec, nu_axis, del_nu, ell_axis, ell_down_sampling=10)

gcfg.plan_MAPS_blocks(N_cpu=16)

gcfg.compute_MAPS()
gcfg.decompose_barC()

save_file_name = 'test_2.h5'
save_location = '/users/zmartino/zmartino/gaussian_field_generation/gaussian_field_generation/saved_data/'

full_file_name = save_location + save_file_name

gcfg.save_data(full_file_name)
print "Results saved to", save_file_name
