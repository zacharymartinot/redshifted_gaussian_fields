"""CLI interface for redshifted_gaussian_fields."""
import typer
import logging
import numpy as np
from astropy import cosmology
from redshifted_gaussian_fields import generator
import time
from rich.console import Console
from pathlib import Path
from hera_cli_utils.logging import setup_logger
from enum import Enum
import h5py

class LogLevel(str, Enum):
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"

class TraceMemBackend(str, Enum):
    tracemalloc = "tracemalloc"
    psutil = "psutil"
    
cns = Console()
logger = logging.getLogger(__name__)
app = typer.Typer()

@app.command()
def covariance(
    freq_min: float,
    freq_max: float,
    delta_nu: float,
    output: Path = typer.Option(
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
    ),
    ell_max: int = 1250,
    log_level: LogLevel = LogLevel.info,
    log_time_as_diff: bool = True,
    log_show_mem: bool = True,
    log_mem_backend: TraceMemBackend = TraceMemBackend.tracemalloc,
    overwrite: bool = False,
):
    setup_logger(
        level = log_level.value,
        width = 160,
        show_time_as_diff = log_time_as_diff,
        rich_tracebacks = True,
        show_mem = log_show_mem,
        mem_backend = log_mem_backend.value,
        show_path = False,
    )
    
    freqs = np.arange(freq_min, freq_max, delta_nu)
    
    k0 = np.logspace(-2.,1.,11)
    a = k0**(-2.7)

    normalization_point = 0.2
    normalization_amplitude = 1.

    Pspec = generator.ParameterizedGaussianPowerSpectrum(a, k0, renormalization=(normalization_point, normalization_amplitude), term_type='flat_gauss')

    nu_axis = 1e-6*freqs # MHz
    del_nu = np.diff(nu_axis)[0]

    ell_axis = np.arange(0,ell_max+1)

    cosmo = cosmology.Planck15
    Np = 15
    eps = 1e-15
    gcfg = generator.GaussianCosmologicalFieldGenerator(cosmo, Pspec, nu_axis, del_nu, ell_axis, Np=Np, eps=eps)

    logger.info("Computing covariance...")
    t1 = time.time()
    gcfg.compute_cross_frequency_angular_power_spectrum()
    t2 = time.time()
    logger.info(f"Elapsed time: {(t2 - t1)/60.:.2f} minutes.")

    gcfg.save_covariance_data(output, overwrite=overwrite)
    logger.info(f"Saved covariance data to {output}.")



@app.command()
def realization(
    nside: int = 32,
    covpath: Path = typer.Option(exists=True, file_okay = True, dir_okay = False),
    outpath: Path = typer.Option(file_okay = True, dir_okay = False),
    seed: int = 2038, 
    low_memory: bool = True,
    log_level: LogLevel = LogLevel.info,
    log_time_as_diff: bool = True,
    log_show_mem: bool = True,
    log_mem_backend: TraceMemBackend = TraceMemBackend.tracemalloc,
    overwrite: bool = False,
):
    gcfg = generator.restore_from_file(covpath)
    setup_logger(
        level = log_level.value,
        width = 160,
        show_time_as_diff = log_time_as_diff,
        rich_tracebacks = True,
        show_mem = log_show_mem,
        mem_backend = log_mem_backend.value,
        show_path = False,
    )
    if low_memory:
        gcfg.generate_healpix_map_realization_low_memory(seed, nside, outpath, overwrite=overwrite)
    else:
        hmap = gcfg.generate_healpix_map_realization(seed, nside)

        with h5py.File(outpath, 'w') as h5f:
            h5f.create_dataset('frequencies_mhz', data=gcfg.nu_axis)
            h5f.create_dataset('healpix_maps_Jy_per_sr', data=hmap)
            h5f.create_dataset('nside', data=nside)
            h5f.create_dataset('realization_random_seed', data=seed)


if __name__ == '__main__':
    app()