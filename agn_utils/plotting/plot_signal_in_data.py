import pickle

import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import (
    infft, logger, latex_plot_format
)
from bilby.gw.utils import asd_from_freq_series
from matplotlib import rcParams
from tqdm.auto import tqdm
from agn_utils.pe_postprocessing.ln_likelihood_calc import get_lnL


@latex_plot_format
def plot_interferometer_waveform_posterior(res, interferometer, level=0.9, n_samples=None, start_time=None,
                                           end_time=None, outdir='.', signals_to_plot={}):
    """
    Plot the posterior for the waveform in the frequency domain and
    whitened time domain.

    If the strain data is passed that will be plotted.

    If injection parameters can be found, the injection will be plotted.

    Parameters
    ==========
    interferometer: (str, bilby.gw.detector.interferometer.Interferometer)
        detector to use, if an Interferometer object is passed the data
        will be overlaid on the posterior
    level: float, optional
        symmetric confidence interval to show, default is 90%
    n_samples: int, optional
        number of samples to use to calculate the median/interval
        default is all
    start_time: float, optional
        the amount of time before merger to begin the time domain plot.
        the merger time is defined as the mean of the geocenter time
        posterior. Default is - 0.4
    end_time: float, optional
        the amount of time before merger to end the time domain plot.
        the merger time is defined as the mean of the geocenter time
        posterior. Default is 0.2

    Returns
    =======
    fig: figure-handle, only is save=False

    Notes
    -----
    To reduce the memory footprint we decimate the frequency domain
    waveforms to have ~4000 entries. This should be sufficient for decent
    resolution.
    """

    DATA_COLOR = "#ff7f0e"
    WAVEFORM_COLOR = "#1f77b4"
    INJECTION_COLOR = "#000000"

    if not isinstance(interferometer, bilby.gw.detector.Interferometer):
        raise TypeError('interferometer type must be Interferometer')

    logger.info("Generating waveform figure for {}".format(interferometer.name))

    if n_samples is None:
        samples = res.posterior
    else:
        samples = res.posterior.sample(n_samples, replace=False)

    if start_time is None:
        start_time = - 0.4
    start_time = np.mean(samples.geocent_time) + start_time
    if end_time is None:
        end_time = 0.2
    end_time = np.mean(samples.geocent_time) + end_time

    time_idxs = (
            (interferometer.time_array >= start_time) &
            (interferometer.time_array <= end_time)
    )
    frequency_idxs = np.where(interferometer.frequency_mask)[0]
    logger.debug("Frequency mask contains {} values".format(
        len(frequency_idxs))
    )
    frequency_idxs = frequency_idxs[::max(1, len(frequency_idxs) // 4000)]
    logger.debug("Downsampling frequency mask to {} values".format(
        len(frequency_idxs))
    )
    plot_times = interferometer.time_array[time_idxs]
    plot_times -= interferometer.strain_data.start_time
    start_time -= interferometer.strain_data.start_time
    end_time -= interferometer.strain_data.start_time
    plot_frequencies = interferometer.frequency_array[frequency_idxs]

    waveform_arguments = res.waveform_arguments
    waveform_arguments['waveform_approximant'] = "IMRPhenomPv2"

    waveform_generator = res.waveform_generator_class(
        duration=res.duration, sampling_frequency=res.sampling_frequency,
        start_time=res.start_time,
        frequency_domain_source_model=res.frequency_domain_source_model,
        parameter_conversion=res.parameter_conversion,
        waveform_arguments=waveform_arguments)

    old_font_size = rcParams["font.size"]
    rcParams["font.size"] = 20
    fig, axs = plt.subplots(
        2, 1,
        gridspec_kw=dict(height_ratios=[1.5, 1]),
        figsize=(16, 12.5)
    )

    axs[0].loglog(
        plot_frequencies,
        asd_from_freq_series(
            interferometer.frequency_domain_strain[frequency_idxs],
            1 / interferometer.strain_data.duration),
        color=DATA_COLOR, label='Data', alpha=0.3)
    axs[0].loglog(
        plot_frequencies,
        interferometer.amplitude_spectral_density_array[frequency_idxs],
        color=DATA_COLOR, label='ASD')
    axs[1].plot(
        plot_times, infft(
            interferometer.whitened_frequency_domain_strain *
            np.sqrt(2. / interferometer.sampling_frequency),
            sampling_frequency=interferometer.strain_data.sampling_frequency)[time_idxs],
        color=DATA_COLOR, alpha=0.3)
    logger.debug('Plotted interferometer data.')

    fd_waveforms = list()
    td_waveforms = list()
    for _, params in tqdm(samples.iterrows(), desc="Processing Samples", total=len(samples)):
        try:
            params = dict(params)
            wf_pols = waveform_generator.frequency_domain_strain(params)
            fd_waveform = interferometer.get_detector_response(wf_pols, params)
            fd_waveforms.append(fd_waveform[frequency_idxs])
            td_waveform = infft(
                fd_waveform * np.sqrt(2. / interferometer.sampling_frequency) /
                interferometer.amplitude_spectral_density_array,
                res.sampling_frequency)[time_idxs]
        except Exception as e:
            logger.debug(f"ERROR: {e}\nparams: {params}")
            pass
        td_waveforms.append(td_waveform)
    fd_waveforms = asd_from_freq_series(
        fd_waveforms,
        1 / interferometer.strain_data.duration)
    td_waveforms = np.array(td_waveforms)

    delta = (1 + level) / 2
    upper_percentile = delta * 100
    lower_percentile = (1 - delta) * 100
    logger.debug(
        'Plotting posterior between the {} and {} percentiles'.format(
            lower_percentile, upper_percentile
        )
    )

    lower_limit = np.mean(fd_waveforms, axis=0)[0] / 1e3
    axs[0].loglog(
        plot_frequencies,
        np.mean(fd_waveforms, axis=0), color=WAVEFORM_COLOR, label='Mean reconstructed')
    axs[0].fill_between(
        plot_frequencies,
        np.percentile(fd_waveforms, lower_percentile, axis=0),
        np.percentile(fd_waveforms, upper_percentile, axis=0),
        color=WAVEFORM_COLOR, label='{}\% credible interval'.format(
            int(upper_percentile - lower_percentile)),
        alpha=0.3)
    axs[1].plot(
        plot_times, np.mean(td_waveforms, axis=0),
        color=WAVEFORM_COLOR)
    axs[1].fill_between(
        plot_times, np.percentile(
            td_waveforms, lower_percentile, axis=0),
        np.percentile(td_waveforms, upper_percentile, axis=0),
        color=WAVEFORM_COLOR,
        alpha=0.3)

    if len(signals_to_plot) > 0:
        for d in signals_to_plot:
            params = d['params']
            label = d['label']
            col = d['color']
            try:
                hf_inj = waveform_generator.frequency_domain_strain(params)
                hf_inj_det = interferometer.get_detector_response(hf_inj, params)
                ht_inj_det = infft(
                    hf_inj_det * np.sqrt(2. / interferometer.sampling_frequency) /
                    interferometer.amplitude_spectral_density_array,
                    res.sampling_frequency)[time_idxs]

                axs[0].loglog(
                    plot_frequencies,
                    asd_from_freq_series(
                        hf_inj_det[frequency_idxs],
                        1 / interferometer.strain_data.duration),
                    label=label, linestyle=':', color=col)
                axs[1].plot(plot_times, ht_inj_det, linestyle=':', color=col)
                logger.debug('Plotted injection.')
            except IndexError as e:
                logger.info('Failed to plot injection with message {}.'.format(e))

    f_domain_x_label = "$f [\\mathrm{Hz}]$"
    f_domain_y_label = "$\\mathrm{ASD} \\left[\\mathrm{Hz}^{-1/2}\\right]$"
    t_domain_x_label = "$t - {} [s]$".format(interferometer.strain_data.start_time)
    t_domain_y_label = "Whitened Strain"

    axs[0].set_xlim(interferometer.minimum_frequency,
                    interferometer.maximum_frequency)
    axs[1].set_xlim(start_time, end_time)
    axs[0].set_ylim(lower_limit)
    axs[0].set_xlabel(f_domain_x_label)
    axs[0].set_ylabel(f_domain_y_label)
    axs[1].set_xlabel(t_domain_x_label)
    axs[1].set_ylabel(t_domain_y_label)
    axs[0].legend(loc='lower left', ncol=2)

    filename = f"{outdir}/{res.label}_{interferometer.name}_waveform.png"

    plt.tight_layout()
    fig.savefig(fname=filename, dpi=600)
    plt.close()
    logger.info("Waveform figure saved to {}".format(filename))
    rcParams["font.size"] = old_font_size


def plot_res_signal_and_max_l(res_file, data_dump_file):
    res = bilby.gw.result.CBCResult.from_json(filename=res_file)

    with open(data_dump_file, "rb") as file:
        data_dump = pickle.load(file)
    ifo_list = data_dump["ifo_list"]

    true = res.injection_parameters
    max_post_param = res.posterior.to_dict('records')[-1]
    lnL = get_lnL(params=[true, max_post_param], res_file=res_file, data_dump_file=data_dump_file)

    plot_interferometer_waveform_posterior(
        res=res, interferometer=ifo_list[0],
        signals_to_plot=[
            dict(params=true, label=f"True (LnL={lnL[0]:.2f})", color="black"),
            dict(params=max_post_param, label=f"MaxLnL (LnL={lnL[1]:.2f}]", color="red"),
        ])

if __name__ == '__main__':
    res_file = "../../studies/snr50_100_fixedphase/outdir_pop_a_highsnr/out_pop_a_highsnr_01/result/pop_a_highsnr_01_0_result.json"
    data_dump_file = "../../studies/snr50_100_fixedphase/outdir_pop_a_highsnr/out_pop_a_highsnr_01/data/pop_a_highsnr_01_data_dump.pickle"
    plot_res_signal_and_max_l(res_file, data_dump_file)
