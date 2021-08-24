import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd
from agn_utils.bbh_population_generators.spin_conversions import make_spin_vector, \
    calculate_relative_spins_from_component_spins
from agn_utils.pe_postprocessing.jsons_to_numpy import load_posteriors_and_trues, save_posteriors_and_trues
from tqdm.auto import tqdm
from agn_utils.data_formetter import ld_to_dl

from agn_utils.plotting.posterior_violin_plotter import simple_violin_plotter

def read_pe_table(dir):
    f = f"{dir}/PEtable.txt"
    cols = open(f).readlines()[0].split()
    print(cols)
    df = pd.read_csv(f, ",", skiprows=1, names=cols)
    print(df)


def read_hdf(f):
    """
    srcmass1: The source-frame mass of the larger object, in solar masses.
    srcmass2: The source-frame mass of the smaller object, in solar masses.
    srcmchirp: The source-frame chirp mass, in solar masses.
    q: The mass ratio, larger object mass to smaller object mass.
    chi_eff: The effective spin of the binary.
    chi_p: The precessing-spin parameter of the binary.
    spin1_a: The dimensionless spin-magnitude of the larger object.
    spin2_a: The dimensionless spin-magnitude of the smaller object.
    spin1_azimuthal: The azimuthal angle of the spin of the larger object.
    spin2_azimuthal: The azimuthal angle of the spin of the smaller object.
    spin1_polar: The polar angle of the spin of the spin of the larger object.
    spin2_polar: The polar angle of the spin of the spin of the smaller object.
    tc: The geocentric GPS time of the signal merger.
    ra: The right ascension of the signal (in radians).
    dec: The declination of the signal (in radians).
    distance: The lumionsity distance to the signal (in Mpc).
    redshift: The cosmological redshift of the signal.
    comoving_volume: The comoving volume at the redshift of the signal.
    inclination: The inclination of the binary’s orbital angular momentum with respect to the line of sight, in radians. An inclination of 0 (pi) corresponds to a face-on (face-away) orientation.
    coa_phase: The coalescence phase of the binary system.
    loglikelihood: The natural log of the likelihood of each sample.
    """
    samples = h5py.File(f, 'r')
    keys = list(samples['samples'].keys())
    posterior = pd.DataFrame({k: samples['samples'][k][:] for k in keys})
    s1x, s1y, s1z = make_spin_vector(tilt=posterior.spin1_polar, phi=posterior.spin1_azimuthal)
    s2x, s2y, s2z = make_spin_vector(tilt=posterior.spin2_polar, phi=posterior.spin2_azimuthal)
    a_1, phi_1, tilt_1, a_2, phi_2, tilt_2, phi_12, theta_12, phi_z_s12 = calculate_relative_spins_from_component_spins(
        s1x, s1y, s1z, s2x, s2y, s2z)
    return dict(cos_tilt_1=np.cos(tilt_1), cos_theta_12=np.cos(theta_12))


def read_hdfs(pkl_fname):
    hdf_dir = os.path.dirname(pkl_fname)
    hdf_regex = os.path.join(hdf_dir, "*.hdf")
    files = glob.glob(hdf_regex)[0:2]
    posteriors, labels, truths = [], [], []
    for f in tqdm(files, desc="HDFs", total=len(files)):
        labels.append(f.replace("_", " "))
        posteriors.append(read_hdf(f))

    posteriors = ld_to_dl(posteriors)

    return dict(posteriors=posteriors, labels=labels)


def load_gwtc_posteriors(pkl_fname):
    if (os.path.isfile(pkl_fname)):
        dat = load_posteriors_and_trues(pkl_fname)
    else:
        dat = read_hdfs(pkl_fname)
        save_posteriors_and_trues(dat, pkl_fname)
    return dat




if __name__ == '__main__':
    gwtc1_dir = sys.argv[1]
    print(f"Getting GWTC events from {gwtc1_dir}")
    read_pe_table(gwtc1_dir)
    load_gwtc_posteriors(pkl_fname=f"{gwtc1_dir}/gwtc.pkl")
    simple_violin_plotter(dat, fname="")
