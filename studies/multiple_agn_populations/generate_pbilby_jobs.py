import os

from bilby.gw.prior import BBHPriorDict

from agn_utils.pe_setup.setup_multiple_pbilby_injections import pbilby_jobs_generator

PRI_PATH = "data/bbh.prior"
PRIOR = BBHPriorDict(filename=PRI_PATH)

POPS = ['data/pop_a_highsnr.dat', 'data/pop_b_highsnr.dat']


def main_job_gen():
    for p in POPS:
        pbilby_jobs_generator(
            injection_file=p,
            label=os.path.basename(p).replace(".dat", ""),
            prior_file="data/bbh.prior",
            psd_file="data/aLIGO_late_psd.txt",
            waveform="IMRPhenomXPHM",
        )


if __name__ == "__main__":
    main_job_gen()
