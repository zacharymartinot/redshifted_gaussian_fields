import numpy as np
from astropy import cosmology

from gaussian_field_generation import generator

# produces a flat P(k) over k in (0.05, 2)
ncomp = 8

alpha = 5.*np.ones(ncomp)
kp = np.geomspace(0.5e-1, 64e-1,ncomp)

beta = (alpha-1.)/kp

a = np.ones(ncomp)
a[0] = 1. + 0.5e-1
for i in range(1,ncomp):
    a[i] = (kp[i]/kp[0])**3.
sigma2 = 2e1

Pspec = generator.GammaPowerSpectrum(a, alpha, beta, sigma2)

cosmo = cosmology.Planck15
full_nu_axis = np.linspace(100.,200.,1024,endpoint=True)
del_nu = np.diff(full_nu_axis)[0]
nu_axis = full_nu_axis[512-128:512+128]

ell_max = 1000
ell_axis = np.arange(0,ell_max+1)

gcfg = generator.GaussianCosmologicalFieldGenerator(cosmo, Pspec, nu_axis, del_nu, ell_axis, ell_down_sampling=1, Np=1, eps=1e-6)

gcfg.plan_MAPS_blocks(N_cpu=16)

gcfg.compute_MAPS()
gcfg.decompose_barC()

save_file_name = 'model_4.h5'
save_location = '/users/zmartino/zmartino/gaussian_field_generation/gaussian_field_generation/saved_data/'

full_file_name = save_location + save_file_name

gcfg.overwrite = True
gcfg.save_data(full_file_name)
print "Results saved to", save_file_name
