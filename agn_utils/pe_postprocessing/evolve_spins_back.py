from lalsimulation.tilts_at_infinity import prec_avg_tilt_comp
import bilby
import numpy as np
import pandas as pd
from bilby.gw.conversion import solar_mass
from tqdm.auto import tqdm


def get_tilts_at_inf(posterior: pd.DataFrame, fref: float):
    param_list = posterior.to_dict('records')

    for i, p in tqdm(enumerate(param_list), desc="Back-evolving Params", total=len(param_list)):
        p, _ = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(p)
        # m1, m2: Detector frame masses of the binary, in kg
        # chi1, chi2: Dimensionless spin magnitudes of the binary
        # tilt1, tilt2: Tilt angles of the binary's spins (w.r.t. the Newtonian orbital angular momentum) at fref
        # phi12: Angle between the in-plane components of the spins at fref
        # fref: Reference frequency, in Hz
        kwargs = dict(
            m1=p["mass_1"] * solar_mass,
            m2=p["mass_2"] * solar_mass,
            chi1=p["a_1"],
            chi2=p["a_2"],
            tilt1=p["tilt_1"],
            tilt2=p["tilt_2"],
            phi12=p["phi_12"],
            fref=fref
        )
        converted = prec_avg_tilt_comp(**kwargs)
        # Output: dictionary with entries 'tilt1_inf', 'tilt2_inf' for evolution to infinity and entries 'tilt1_sep_min',
        # 'tilt1_sep_max', 'tilt1_sep_avg', 'tilt2_sep_min', 'tilt2_sep_max', 'tilt2_sep_avg' for evolution to a
        # finite separation (i.e., a finite orbital angular momentum)
        for k in converted.keys():
            new_k = k.replace("tilt", "tilt_")
            param_list[i][new_k] = converted[k]

    return pd.DataFrame(param_list)


def test_converter():
    from agn_utils.plotting.overlaid_corner_plotter import CORNER_KWARGS
    import corner
    prior = bilby.gw.prior.BBHPriorDict()
    samples = pd.DataFrame(prior.sample(1000))
    samples = get_tilts_at_inf(samples, fref=20)
    for c in samples.columns:
        if 'tilt' in c[0:5]:
            samples[f'cos_{c}'] = np.cos(samples[c])
    params = ['cos_tilt_1', 'cos_tilt_2', 'cos_tilt_1_inf', 'cos_tilt_2_inf']
    labels = [r'$\cos\theta_1$', r'$\cos\theta_2$', r'$\cos\theta_1^{\infty}$', r'$\cos\theta_2^{\infty}$']

    fig = corner.corner(samples[params], labels=labels, **CORNER_KWARGS, color="tab:blue")
    fig.savefig('spins_at_different_freq.png')
    print("Complete")


if __name__ == '__main__':
    test_converter()
