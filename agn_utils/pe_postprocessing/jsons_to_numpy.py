import glob
import os
import pickle
import numpy as np
import pandas as pd
from bilby.gw.result import CBCResult
from tqdm import tqdm

from ..bbh_population_generators.calculate_extra_bbh_parameters import add_cos_theta_12_from_component_spins
from bilby.gw.conversion import generate_all_bns_parameters

def load_bilby_results(regex):
    res = []
    files = glob.glob(regex)
    for f in tqdm(files):
        r = CBCResult.from_json(filename=f)
        pos = r.posterior.sample(1000)
        r.posterior = convert_params(pos)
        r.injection_parameters = convert_params(r.injection_parameters)
        res.append(r)

    return res

def convert_params(p):
    for i in [1, 2]:
        if isinstance(p, pd.DataFrame):
            p = p.astype({f'spin_{i}x': 'float64', f'spin_{i}y': 'float64', f'spin_{i}z': 'float64'})
        else:
            p = generate_all_bns_parameters(p)
    p = add_cos_theta_12_from_component_spins(p)
    p['cos_theta_1'] = p['cos_tilt_1']
    return p


def convert_results_to_dict_of_posteriors(res, params):
    posteriors = {p: [] for p in params}
    trues = {p: [] for p in params}

    for r in res:
        for p in params:
            posteriors[p].append(r.posterior[p].ravel())
            if r.injection_parameters != None:
                trues[p].append(r.injection_parameters[p])

    return dict(posteriors=posteriors, trues=trues)


def save_posteriors_and_trues(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def load_posteriors_and_trues(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        return data


def get_bilby_results(regex, picklefname, params, clean=False):


    if not clean and os.path.isfile(picklefname):
        dat = load_posteriors_and_trues(picklefname)
    else:
        res = load_bilby_results(regex)
        dat = convert_results_to_dict_of_posteriors(res, params)
        save_posteriors_and_trues(dat, picklefname)


    return dat


