import glob
import os
import pickle
import numpy as np
import pandas as pd
from bilby.gw.result import CBCResult
from tqdm import tqdm
from typing import Dict, List
from ..bbh_population_generators.calculate_extra_bbh_parameters import add_cos_theta_12_from_component_spins, result_post_processing
from bilby.gw.conversion import _generate_all_cbc_parameters, convert_to_lal_binary_black_hole_parameters


def load_bilby_results(regex):
    res = []
    files = glob.glob(regex)
    files = sorted(files)
    for f in tqdm(files, desc=f"Loading {regex}"):
        try:
            r = CBCResult.from_json(filename=f)
            r = result_post_processing(r)

            res.append(r)
        except Exception as e:
            print(f"Error: {f}\n{e}")

    return res
#
# def convert_params(p):
#     for i in [1, 2]:
#         if isinstance(p, pd.DataFrame):
#             p = p.astype({f'spin_{i}x': 'float64', f'spin_{i}y': 'float64', f'spin_{i}z': 'float64'})
#         else:
#             waveform_defaults = {
#                 'reference_frequency': 20.0, 'waveform_approximant': 'IMRPhenomPv2',
#                 'minimum_frequency': 20.0}
#             p = _generate_all_cbc_parameters(
#                 p, defaults=waveform_defaults,
#                 base_conversion=convert_to_lal_binary_black_hole_parameters,
#                 likelihood=None, priors=None, npool=None)
#
#     p = add_cos_theta_12_from_component_spins(p)
#     p['cos_theta_1'] = np.cos(p['tilt_1'])
#     p['cos_theta_1'] = p['cos_tilt_1']
#     return p
#
# def process_samples(s, rf):
#     s['reference_frequency'] = rf
#     s, _ = convert_to_lal_binary_black_hole_parameters(s)
#     s = generate_mass_parameters(s)
#     s = generate_spin_parameters(s)
#     s = add_cos_theta_12_from_component_spins(s)
#     try:
#         s = add_snr(s)
#         s['snr'] = s['network_snr']
#     except Exception as e:
#         pass
#     return s


def convert_results_to_dict_of_posteriors(res, params):
    posteriors = {p: [] for p in params}
    trues = {p: [] for p in params}
    labels = []

    for r in res:
        for p in params:
            posteriors[p].append(r.posterior[p].ravel())
            if r.injection_parameters != None:
                trues[p].append(r.injection_parameters[p])
        labels.append(r.label.replace("_"," "))

    return dict(posteriors=posteriors, trues=trues, labels=labels)


def save_posteriors_and_trues(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def load_posteriors_and_trues(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        return data


def get_bilby_results(regex, picklefname, params, clean=False)->Dict[str,List]:
    """
    :return: dict(posteriors=posteriors, trues=trues, labels=labels)
    """

    if not clean and os.path.isfile(picklefname):
        dat = load_posteriors_and_trues(picklefname)
    else:
        print(f"Cannot find {picklefname}. Recreating.")
        res = load_bilby_results(regex)
        dat = convert_results_to_dict_of_posteriors(res, params)
        save_posteriors_and_trues(dat, picklefname)

    print(f"Loaded {len(dat['labels'])} posteriors with {list(dat['posteriors'].keys())} params")

    return dat


