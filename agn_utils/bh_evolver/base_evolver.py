import time
from abc import ABC, abstractmethod
from typing import Dict

from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters as convert_to_lal_bbh


class BaseEvolver():
    def __init__(self, params: Dict[str, float], fref: float):
        self.params, _ = convert_to_lal_bbh(params)
        self.fref = fref

    @abstractmethod
    def convert_to_evol_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Converts to dict of params for the evolution"""
        return {}

    @abstractmethod
    def convert_results_to_ligo_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Converts to evolution output to LIGO param dict"""
        return {}

    @abstractmethod
    def _evolve(self, params) -> Dict[str, float]:
        """Warpper funct to call the evolution code from the external library"""
        pass

    def run_evolver(self):

        evolver_input = self.convert_to_evol_params(self.params.copy())

        start_time = time.time()
        try:
            evolver_output = self._evolve(evolver_input)
            evolver_output['evolution_successful'] = True
        except Exception as e:
            evolver_output = {
                'evolution_successful': False,
                'error': str(e)
            }

        end_time = time.time()
        evolver_output['runtime'] = end_time - start_time

        processed_output = self.convert_results_to_ligo_params(evolver_output)
        return processed_output
