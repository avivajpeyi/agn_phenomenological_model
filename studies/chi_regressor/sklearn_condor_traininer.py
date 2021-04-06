import argparse

from agn_utils.batch_processing import create_python_script_jobs
from agn_utils.chi_regressor import chi_regressor_trainer, load_model, \
    prediction_for_different_sigma

MODEL_FNAME = "sklearn_chi_regressor.model"
TRAINING_DATA = "training_data.h5"


def create_parser_and_read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-dag", help="Make dag", action="store_true")
    parser.add_argument("--model-fname", help="model fname", type=str,
                        default='my.model')
    parser.add_argument("--model-type", help="model type", type=str,
                        default='Scikit')
    args = parser.parse_args()
    return args


def main():
    args = create_parser_and_read_args()
    if args.make_dag:
        create_python_script_jobs(
            main_job_name="training_chi_regressor",
            python_script=__file__,
            job_args_list=[{'model-fname': MODEL_FNAME}],
            job_names_list=["trainer"],
            request_memory="16 GB"
        )
    else:
        chi_regressor_trainer(
            model_fname=args.model_fname, model_type=args.model_type,
            training_fname=TRAINING_DATA, n_samples=None)
        regressor = load_model(MODEL_FNAME)
        for s in [[0.1, 0.8], [3, 0.3], [1, 1]]:
            prediction_for_different_sigma(
                sigma_1=s[0], sigma_12=s[1], n=5000, regressor=regressor
            )


if __name__ == '__main__':
    main()
