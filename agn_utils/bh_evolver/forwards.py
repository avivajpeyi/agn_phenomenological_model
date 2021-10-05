# from . import BaseEvolver
import precession
import time
import numpy

from precession import rtof, ftor
from typing import Dict

from base_evolver import BaseEvolver

class PrecessionForwardEvolver(BaseEvolver):

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




def test():
    t0=time.time()
    q=0.75 # Mass ratio
    chi1=0.5 # Primary’s spin magnitude
    chi2=0.95 # Secondary’s spin magnitude
    print("Take a BH binary with q=%.2f, chi1=%.2f and ,→ chi2=%.2f" %(q,chi1,chi2))
    sep=numpy.logspace(10,1,10) # Output separations


    t1= numpy.pi/3. # Spin orientations at r_vals[0]
    t2= 2.*numpy.pi/3.
    dp= numpy.pi/4.
    M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2)
    freq = [rtof(s, M) for s in sep]
    print(ftor(20, M))
    t1v,t2v,dpv=precession.evolve_angles(t1,t2,dp,sep,q,S1,S2)
    print("Perform BH binary inspiral")
    print("log10(r/M) \t freq(Hz) \t theta1 \t theta2 \t deltaphi")
    for r, f, t1,t2,dp in zip(numpy.log10(sep), freq, t1v,t2v,dpv):
        print("%.0f \t\t %.3f \t\t %.3f \t\t %.3f \t\t %.3f" %(r,f, t1,t2,dp))
    t=time.time()-t0
    print("Executed in %.3fs" %t)

if __name__ == '__main__':
    print(ftor(0.001, 40))