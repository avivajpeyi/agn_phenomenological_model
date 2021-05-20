import argparse

from agn_utils.batch_processing import create_python_script_jobs
from agn_utils.chi_regressor import (
    AvailibleRegressors,
    chi_regressor_trainer,
    load_model,
    prediction_for_different_sigma,
)

TRAINING_DATA = "training_data.h5"


def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-dag", help="Make dag", action="store_true")
    parser.add_argument(
        "--outdir", help="model's outdir'", type=str, default="my.model"
    )
    parser.add_argument(
        "--model-type", help="model type", type=str, default="SCIKIT"
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="num training samples to use (use all sample by default)",
    )
    args = parser.parse_args()
    return args


def main():
    args = create_parser_and_read_args()
    if args.make_dag:
        model_type = AvailibleRegressors[args.model_type.upper()]
        outdir = f"{model_type.name.lower()}_regressor_files"
        extra_lines = []
        if model_type == AvailibleRegressors.TF:
            extra_lines = ["request_gpus = 1"]
        create_python_script_jobs(
            main_job_name="training_chi_regressor",
            python_script=__file__,
            job_args_list=[{"outdir": outdir, "model-type": model_type.name}],
            job_names_list=["trainer"],
            request_memory="16 GB",
            extra_lines=extra_lines,
        )
    else:
        model_type = AvailibleRegressors[args.model_type.upper()]
        chi_regressor_trainer(
            outdir=args.outdir,
            model_type=model_type,
            training_fname=TRAINING_DATA,
            n_samples=args.nsamples,
        )
        regressor = load_model(outdir=args.outdir, model_type=model_type)
        for s in [[0.1, 0.8], [3, 0.3], [1, 1]]:
            prediction_for_different_sigma(
                sigma_1=s[0],
                sigma_12=s[1],
                n=5000,
                regressor=regressor,
                save=True,
            )


if __name__ == "__main__":
    main()
