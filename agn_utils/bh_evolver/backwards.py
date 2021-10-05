import time

import bilby
import numpy as np
import pandas as pd
from bilby.gw.conversion import solar_mass
from lalsimulation.tilts_at_infinity import calc_tilts_at_infty_hybrid_evolve
from lalsimulation.tilts_at_infinity import prec_avg_tilt_comp
from tqdm.auto import tqdm

from . import BaseEvolver

from typing import Dict

class PrecessionBackwardEvolver(BaseEvolver):

    def convert_to_evol_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Converts to dict of params for the evolution"""
        return dict(
            m1=params["mass_1"] * solar_mass,
            m2=params["mass_2"] * solar_mass,
            chi1=params["a_1"],
            chi2=params["a_2"],
            tilt1=params["tilt_1"],
            tilt2=params["tilt_2"],
            phi12=params["phi_12"],
            fref=self.fref
        )

    def convert_results_to_ligo_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Converts to evolution output to LIGO param dict"""
        final_params = self.params.copy()
        for k in params.keys():
            if 'tilt' in k:
                new_k = k.replace("tilt", "tilt_")
            if 'phi' in k:
                new_k = k.replace("phi", "phi_")
            if 'trans' in k:
                new_k = k.replace("_transition", "_ft")
            else:
                new_k = k
            final_params[new_k] = params.get(k, np.nan)

        return final_params

    def _evolve(self, params):
        """Warpper funct to call the evolution code from the external library
        ['tilt1_inf', 'tilt2_inf', 'tilt1_sep_min', 'tilt1_sep_max', 'tilt1_sep_avg', 'tilt2_sep_min', 'tilt2_sep_max', 'tilt2_sep_avg'
        """
        return prec_avg_tilt_comp(**params)


class HybridBackwardEvolver(PrecessionBackwardEvolver):
    def _evolve(self, params):
        """Warpper funct to call the evolution code from the external library
        ['tilt1_inf', 'tilt2_inf', 'tilt1_transition', 'tilt2_transition', 'phi12_transition']
        """
        return calc_tilts_at_infty_hybrid_evolve(**params)

