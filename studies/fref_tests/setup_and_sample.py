import bilby
import sys
import csv
import numpy as np


def load_injection_parameters():
    with open('fref_test_injection.csv',newline='') as pscfile:
        reader = csv.reader(pscfile)
        next(reader)
        results = dict(reader)
    for k,v in results.items():
        results[k] = float(v)
    return results

def load_prior():
    return bilby.gw.prior.CBCPriorDict(filename="fref_test.prior")

def main(label, pe_fref, injection_fref):
    np.random.seed(0)
    duration = 4.
    sampling_frequency = 2048.

    outdir = f'outdir_{label}'
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    priors = load_prior()
    injection_parameters = load_injection_parameters()

    # Setup IFO and inject signal
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection_parameters['geocent_time'] - 3)

    injection_waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant='IMRPhenomPv2',
                                reference_frequency=injection_fref, minimum_frequency=20.))
    ifos.inject_signal(waveform_generator=injection_waveform_generator,
                       parameters=injection_parameters)

    # Setup LnL
    analysis_waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant='IMRPhenomPv2',
                                reference_frequency=pe_fref, minimum_frequency=20.))
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=analysis_waveform_generator)

    # Sample
    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
        injection_parameters=injection_parameters, outdir=outdir, label=label)

    # Make a corner plot.
    result.plot_corner()



if __name__ == '__main__':
    if sys.argv[1]=='run_a':
        main(label="run_a", injection_fref=20, pe_fref=20)
    elif sys.argv[1]=='run_b':
        main(label="run_b", injection_fref=0.001, pe_fref=20)