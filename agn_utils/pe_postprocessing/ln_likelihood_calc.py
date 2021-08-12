import pickle

import bilby


def get_lnL(params, res_file, data_dump_file):
    lnl = setup_likelihood(res_file, data_dump_file)
    vals = []
    for p in params:
        lnl.parameter.update(p)
        vals.append(lnl.log_likelihood())
    return vals


def setup_likelihood(res_file, data_dump_file):
    res = bilby.gw.result.CBCResult.from_json(filename=res_file)

    with open(data_dump_file, "rb") as file:
        data_dump = pickle.load(file)

    ifo_list = data_dump["ifo_list"]
    waveform_generator = data_dump["waveform_generator"]
    injection_parameters = data_dump.get("injection_parameters", None)

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifo_list, waveform_generator=waveform_generator)

    return likelihood


def main():
    res_file = "../../studies/snr50_100_fixedphase/outdir_pop_a_highsnr/out_pop_a_highsnr_01/result/pop_a_highsnr_01_0_result.json",
    data_dump_file = "../../studies/snr50_100_fixedphase/outdir_pop_a_highsnr/out_pop_a_highsnr_01/data/pop_a_highsnr_01_data_dump.pickle",
    res = bilby.gw.result.CBCResult.from_json(filename=res_file)
    params = [res.injection_parameters]
    lnl = get_lnL(params, res_file, data_dump_file)

    print(
        f"max(posterior LnL): {max(res.posterior.log_likelihood) + res.log_noise_evidence}\n"
        f"LnL(true val): {lnl[0]}"
    )
