import numpy as np
import h5py
from scipy import linalg
from astropy import cosmology
import time

from redshifted_gaussian_fields import generator

from RIMEz import sky_models

import inspect
this_file = inspect.getmodule(inspect.currentframe())
source_string = np.string_(inspect.getsource(this_file))

git_hash = generator.get_commit_hash()

nu_hz = 1e6*np.linspace(100.,200.,1024,endpoint=True)
del_nu = np.diff(nu_hz)[0]

cosmo = cosmology.Planck15

A0 = 1e1 * (1./0.2)**-2.

ell_max = 1000

start_time = time.time()

barC = generator.pow_neg2_CAPS(cosmo, 1e-6*nu_hz, 1e-6*del_nu, A0, ell_max)

eig_vals = np.zeros(barC.shape[:2])
eig_vecs = np.zeros_like(barC)

for ll in range(barC.shape[0]):
    eig_vals[ll], eig_vecs[ll] = linalg.eigh(barC[ll])

N_realizations = 20

save_location = '/users/zmartino/zmartino/projects/redshifted_gaussian_fields/saved_data/'

file_name = 'power_law_neg2_100_200MHz_realizations.h5'

random_seeds = np.random.randint(1,100000, 20)

with h5py.File(save_location + file_name, 'w') as h5f:
    h5f.create_dataset('git_hash', data=git_hash)
    h5f.create_dataset('script_source', data=source_string)

    cosmology_name = np.string_(cosmo.name)

    h5f.create_dataset('frequency_samples_hz', data=nu_hz)
    h5f.create_dataset('frequency_channel_width_hz', data=del_nu)
    h5f.create_dataset('cosmology_name', data=cosmology_name)
    h5f.create_dataset('power_spectrum_amplitude_parameter', data=A0)
    h5f.create_dataset('ell_max', data=ell_max)

    for ii in range(N_realizations):
        print "Generating realization {0}...".format(str(ii))

        seed_i = random_seeds[ii]

        alm = generator.sh_realization_healpy_index(eig_vals, eig_vecs, seed_i)

        Ilm_s = sky_models.hp2ssht_index(alm.T)

        Ilm_s = Ilm_s.reshape(Ilm_s.shape + (1,))

        data_ii = h5f.create_group(str(ii))
        data_ii.create_dataset('Ilm', data=Ilm_s)
        data_ii.create_dataset('numpy_random_seed', data=seed_i)

end_time = time.time()

print "Elapsed time:", (end_time - start_time)/60., "minutes"
print "Done."
