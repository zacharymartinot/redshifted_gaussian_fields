import numpy as np
from astropy import cosmology

from gaussian_field_generation import generator

# produces a spectrum that is ~ k^-3 over k in [0.1,1]
ncomp = 6
a = np.ones(ncomp)
alpha = 5.*np.ones(ncomp)
kp = np.geomspace(0.25e-1, 8e-1,ncomp)

beta = (alpha-1.)/kp
sigma2 = 8e-2

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

save_file_name = 'model_3.h5'
save_location = '/users/zmartino/zmartino/gaussian_field_generation/gaussian_field_generation/saved_data/'

full_file_name = save_location + save_file_name

gcfg.overwrite = True
gcfg.save_data(full_file_name)
print "Results saved to", save_file_name
