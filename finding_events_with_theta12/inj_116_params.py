import logging

import bilby
import pandas as pd
import numpy as np

logging.getLogger("bilby").setLevel(logging.ERROR)

INJ_116_PARAMS = {
    'a_1': 0.389904504601928,
    'a_2': 0.1418128960775321,
    'cos_tilt_1': 0.9328944117496468,
    'cos_tilt_2': 0.8029568249927979,
    'dec': 0.9646705051434916,
    'incl': 5.804161943126117,
    'iota': 0.48066097,
    'luminosity_distance': 445.31366907470334,
    'mass_1': 35.96686320346735,
    'mass_2': 34.13510878564803,
    'phase': 5.085901425161205,
    'phi_1': 4.797048641494505,
    'phi_12': 1.368807147538346,
    'phi_2': 3.024263135443057,
    'phi_jl': 0.0,
    'psi': 1.0615357991139194,
    'ra': 1.6706990245656723,
    'theta_jn': 0.4790233640325695,
    'tilt_1': 0.36842866945870006,
    'tilt_2': -0.6385567492319952,
    'geocent_time': 0.0
}


@np.vectorize
def get_injection_snr(
        a_1,
        a_2,
        dec,
        ra,
        psi,
        phi_12,
        phase,
        incl,
        geocent_time,
        mass_1,
        mass_2,
        luminosity_distance,
        tilt_1,
        tilt_2,
        theta_jn,
        phi_jl, **kwargs):
    """
    :returns H1 snr, L1 snr, network SNR
    """
    injection_parameters = dict(
        a_1=a_1,
        a_2=a_2,
        dec=dec,
        ra=ra,
        psi=psi,
        phi_12=phi_12,
        phase=phase,
        incl=incl,
        geocent_time=geocent_time,
        mass_1=mass_1,
        mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        theta_jn=theta_jn,
        phi_jl=phi_jl,
    )

    duration = 4
    sampling_frequency = 2048.

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant='IMRPhenomPv2',
            reference_frequency=50., minimum_frequency=20.
        )
    )

    # Set up interferometers.
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters['geocent_time'] - 2
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator,
        parameters=injection_parameters
    )

    snrs = [ifo.meta_data["optimal_SNR"] for ifo in ifos]
    network_snr = np.sqrt(np.sum([i ** 2 for i in snrs]))
    return snrs[0], snrs[1], network_snr


def create_injections(num=10):
    distances = np.logspace(np.log10(200), np.log10(5000), base=10, num=num)
    masses = np.logspace(np.log10(15), np.log10(200), base=10,  num=num)
    inj_params = []
    df = []
    for d, m in zip(distances, masses):
        params = INJ_116_PARAMS.copy()
        params.update({
            'luminosity_distance': d,
            'mass_1': m,
            'mass_2': m,
        })
        inj_params.append(params)
        _, _, snr = get_injection_snr(**params)
        df.append(dict(dist=d, m1=m, snr=snr))
    snr_info = pd.DataFrame(df)
    print(snr_info)
    snr_info.to_csv("SNR_info.csv", index=False)

    pd.DataFrame(inj_params).to_csv("similar_orientation_injections.dat", sep=" ", index=False)


def main():
    create_injections()


if __name__ == '__main__':
    main()
