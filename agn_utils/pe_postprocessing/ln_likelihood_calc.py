import bilby
import numpy as np
import pickle

def setup_likelihood(res_file, data_dump_file, reference_frequency, waveform, duration=4,sampling_frequency=2048):

    res  = bilby.gw.result.CBCResult.from_json(filename=res_file)


    with open(data_dump_file, "rb") as file:
        data_dump = pickle.load(file)

    ifo_list = data_dump["ifo_list"]
    waveform_generator = data_dump["waveform_generator"]
    # waveform_generator.start_time = ifo_list[0].time_array[0]
    injection_parameters = data_dump.get("injection_parameters", None)

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifo_list, waveform_generator=waveform_generator)

    likelihood.parameters.update(injection_parameters)
    # print(injection_parameters)
    # print(likelihood.parameters)
    true_logL = likelihood.log_likelihood()
    max_likelihood_posterior = max(res.posterior.log_likelihood)


    print(f"MaxPosterior LnL: {max_likelihood_posterior}\n"
          f"LnL(true val): {true_logL}")



setup_likelihood(
    res_file="../../studies/snr50_100_fixedphase/outdir_pop_a_highsnr/out_pop_a_highsnr_01/result/pop_a_highsnr_01_0_result.json",
    data_dump_file="../../studies/snr50_100_fixedphase/outdir_pop_a_highsnr/out_pop_a_highsnr_01/data/pop_a_highsnr_01_data_dump.pickle",
    reference_frequency=20,
    waveform="IMRPheonomPXHM",
    duration=4,sampling_frequency=2048
)