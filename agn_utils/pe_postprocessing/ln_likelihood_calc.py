import pickle

import bilby


def setup_likelihood(res_file, data_dump_file):
    res = bilby.gw.result.CBCResult.from_json(filename=res_file)

    with open(data_dump_file, "rb") as file:
        data_dump = pickle.load(file)

    ifo_list = data_dump["ifo_list"]
    waveform_generator = data_dump["waveform_generator"]
    injection_parameters = data_dump.get("injection_parameters", None)

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifo_list, waveform_generator=waveform_generator)

    likelihood.parameters.update(injection_parameters)
    # print(injection_parameters)
    # print(likelihood.parameters)
    true_logL = likelihood.log_likelihood()
    max_likelihood_posterior = max(res.posterior.log_likelihood)
    max_lnl_param = res.posterior.to_dict('records')[-1]
    likelihood.parameters.update(max_lnl_param)
    max_likelihood_posterior = likelihood.log_likelihood()

    print(
        f"max(posterior LnL): {max(res.posterior.log_likelihood) + res.log_noise_evidence}\n"
        # f"LnL(posterior max LnL param): {max_likelihood_posterior}\n"
        f"LnL(true val): {true_logL}")
    h1_dat_fig = res.plot_interferometer_waveform_posterior(interferometer=ifo_list[0], save=False)
    h1_dat_fig.savefig("h1.png")
    l1_dat_fig = res.plot_interferometer_waveform_posterior(interferometer=ifo_list[0], save=False)
    l1_dat_fig.savefig("h1.png")
    pass


setup_likelihood(
    res_file="../../studies/snr50_100_fixedphase/outdir_pop_a_highsnr/out_pop_a_highsnr_01/result/pop_a_highsnr_01_0_result.json",
    data_dump_file="../../studies/snr50_100_fixedphase/outdir_pop_a_highsnr/out_pop_a_highsnr_01/data/pop_a_highsnr_01_data_dump.pickle",
)
