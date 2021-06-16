from agn_utils.bbh_population_generators import get_bbh_population_from_agn_params

if __name__ == "__main__":
    df = get_bbh_population_from_agn_params(
        num_samples=100,
        sigma_1=0.5,
        sigma_12=2
    )
    df.to_csv("samples.dat", sep=' ')
