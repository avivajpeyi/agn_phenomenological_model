from lalsimulation.tilts_at_infinity import prec_avg_tilt_comp
from lalsimulation.tilts_at_infinity import calc_tilts_at_infty_hybrid_evolve
import bilby
import numpy as np
import pandas as pd
from bilby.gw.conversion import solar_mass
from tqdm.auto import tqdm
import corner
import time


def get_tilts_at_inf(posterior: pd.DataFrame, fref: float, only_precession=True):
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
        if only_precession:
            converted_dict = back_evolve_only_precession(kwargs)
        else:
            converted_dict = back_evolve_hybrid(kwargs)

        for k, v in converted_dict.items():
            param_list[i][k] = v

    return pd.DataFrame(param_list)


def back_evolve_only_precession(kwargs):
    final_dict = {}
    try:
        converted = prec_avg_tilt_comp(**kwargs)
    except Exception as e:
        converted = {}
    converted_keys = ['tilt1_inf', 'tilt2_inf', 'tilt1_sep_min', 'tilt1_sep_max', 'tilt1_sep_avg', 'tilt2_sep_min', 'tilt2_sep_max', 'tilt2_sep_avg']
    for k in converted_keys:
        new_k = k.replace("tilt", "tilt_")
        final_dict[new_k] = converted.get(k, np.nan)
    return final_dict

def back_evolve_hybrid(kwargs):
    final_dict = {}
    try:
        converted = calc_tilts_at_infty_hybrid_evolve(**kwargs)
    except Exception as e:
        converted = {}
    converted_keys = ['tilt1_inf', 'tilt2_inf', 'tilt1_transition', 'tilt2_transition', 'phi12_transition']
    for k in converted_keys:
        if 'tilt' in k:
            new_k = k.replace("tilt", "tilt_")
        if 'phi' in k:
            new_k = k.replace("phi", "phi_")
        if 'trans' in k:
            new_k = k.replace("_transition", "_ft")
        final_dict[new_k] = converted.get(k, np.nan)
    return final_dict

def test_converter():
    prior = bilby.gw.prior.BBHPriorDict()
    samples = pd.DataFrame(prior.sample(10))
    t0 = time.time()
    samples = get_tilts_at_inf(samples, fref=20, only_precession=True)
    t1 = time.time()
    samples = get_tilts_at_inf(samples, fref=20, only_precession=False)
    t2 = time.time()
    print(f"only precession: {(t1-t0)/len(samples)}")
    print(f"hybrid: {(t2-t1)/len(samples)}")


if __name__ == '__main__':
    test_converter()
