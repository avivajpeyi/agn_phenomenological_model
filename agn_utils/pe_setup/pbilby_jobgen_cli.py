import argparse
import os

from agn_utils.pe_setup.setup_multiple_pbilby_injections import pbilby_jobs_generator


def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-files", nargs='+', default=[])
    parser.add_argument("--prior-file", type=str)
    parser.add_argument("--psd-file", type=str)
    parser.add_argument("--cluster", type=str)
    parser.add_argument("--fref", type=float, default=20)
    args = parser.parse_args()
    return args


def main_job_gen(prior_file, psd_file, pop_files, cluster, fref=20):
    for p in pop_files:
        pbilby_jobs_generator(
            injection_file=p,
            label=os.path.basename(p).replace(".dat", ""),
            prior_file=prior_file,
            psd_file=psd_file,
            waveform="IMRPhenomXPHM",
            cluster=cluster,
            fref=fref
        )


def main():
    args = create_parser_and_read_args()
    main_job_gen(prior_file=args.prior_file, psd_file=args.psd_file, pop_files=args.pop_files, cluster=args.cluster, fref=args.fref)


if __name__ == "__main__":
    main()