import unittest
import pandas as pd
import numpy as np
import unittest
from deepdiff import DeepDiff
import numpy as np
import pandas as pd

pd.reset_option('all')
pd.set_option("precision", 4)

MSUN = 1.9884099021470415e+30  # Kg
REFERENCE_FREQ = 20.0
PI = np.pi
PI2 = np.pi * 2.0
ANGLE_PARMS = ["phi_1", "phi_2", "phi_2", "tilt_1", "tilt_2", "theta_12"]

from agn_utils.bbh_population_generators.spin_conversions import (agn_to_cartesian_coords,
                                                                  cartesian_to_spherical_coords,
                                                                  calculate_relative_spins_from_component_spins,
                                                                  spherical_to_cartesian_coords)


class MyTestCase(unittest.TestCase):

    def test_start_with_agn_params(self):
        init_params = dict(
            incl=0.1, phi_1=0.25 * np.pi, tilt_1=0.5 * np.pi, theta_12=np.pi / 3, phi_z_s12=np.pi / 4, a_1=1.0,
            a_2=1.0, mass_1=20.0, mass_2=10.0, reference_frequency=REFERENCE_FREQ, phase=0.1
        )

        cc = {}
        (
            cc['incl'],
            cc['spin_1x'], cc['spin_1y'], cc['spin_1z'],
            cc['spin_2x'], cc['spin_2y'], cc['spin_2z'],
        ) = agn_to_cartesian_coords(**init_params)
        cc['mass_1'], cc['mass_2'], cc['reference_frequency'], cc['phase'] = init_params['mass_1'], init_params[
            'mass_2'], \
                                                                             init_params['reference_frequency'], \
                                                                             init_params['phase']

        sp = {}
        (sp['theta_jn'], sp['phi_jl'], sp['tilt_1'], sp['tilt_2'], sp['phi_1'], sp['phi_2'], sp['a_1'], sp['a_2'],
         sp['phi_12'], sp['theta_12'], sp['phi_z_s12'],
         ) = cartesian_to_spherical_coords(**cc)
        sp['mass_1'], sp['mass_2'], sp['reference_frequency'], sp['phase'] = init_params['mass_1'], init_params[
            'mass_2'], \
                                                                             init_params['reference_frequency'], \
                                                                             init_params['phase']
        _, _, _, _, _, _, cc['phi_12'], cc['theta_12'], cc['phi_z_s12'] = calculate_relative_spins_from_component_spins(
            cc['spin_1x'], cc['spin_1y'],
            cc['spin_1z'], cc['spin_2x'],
            cc['spin_2y'], cc['spin_2z'])
        fi = {}
        (fi['incl'], fi['spin_1x'], fi['spin_1y'], fi['spin_1z'], fi['spin_2x'], fi['spin_2y'],
         fi['spin_2z']) = spherical_to_cartesian_coords(**sp)
        _, _, _, _, _, _, fi['phi_12'], fi['theta_12'], cc['phi_z_s12'] = calculate_relative_spins_from_component_spins(
            fi['spin_1x'], fi['spin_1y'],
            fi['spin_1z'], fi['spin_2x'],
            fi['spin_2y'], fi['spin_2z'])

        df = pd.DataFrame(
            [init_params, cc, sp, fi],
            index=["agn", "cartesian", "spherical", 'final']
        )
        for p in ANGLE_PARMS:
            df[f"{p} [pi]"] = df[p] / np.pi
        df = df.round(3)
        df = df.T

        print(df.to_markdown())

    def test_start_list_of_params(self):
        dat = {'incl': np.array([3.94921017, 1.32995487, 4.68609126, 1.19239323, 2.60931897]),
               'phi_1': np.array([5.15985128, 0.5264748, 3.33111948, 5.23079906, 4.63027178]),
               'tilt_1': np.array([0.2258557, 0.40816371, 0.49343825, 0.5051792, 1.15176435]),
               'theta_12': np.array([1.1060856, 2.27114761, 0.43123208, 1.05749932, 1.31113746]),
               'phi_z_s12': np.array([4.58981089, 3.8536873, 2.42671716, 3.05066716, 3.84617618]),
               'a_1': np.array([0.6, 0.6, 0.6, 0.6, 0.6]),
               'a_2': np.array([0.6, 0.6, 0.6, 0.6, 0.6]),
               'mass_1': np.array([58.13670017, 55.69592184, 28.73117918, 43.26630506, 17.86542829]),
               'mass_2': np.array([58.13670017, 55.69592184, 28.73117918, 43.26630506, 17.86542829]),
               'reference_frequency': [20.0, 20.0, 20.0, 20.0, 20.0],
               'phase': np.array([1.40732206, 0.31020436, 3.55943835, 6.1410302, 2.4002834])}
        incl, s1x, s1y, s1z, s2x, s2y, s2z = agn_to_cartesian_coords(**dat)
        initial_spin_vector = dict(s1x=s1x, s1y=s1y, s1z=s1z, s2x=s2x, s2y=s2y, s2z=s2z, i=incl)
        mass_1, mass_2 = dat['mass_1'], dat['mass_2']
        theta_jn, phi_jl, tilt_1, tilt_2, phi_1, phi_2, a_1, a_2, phi_12, theta_12, phi_z_s12 = cartesian_to_spherical_coords(
            incl,
            s1x, s1y, s1z, s2x, s2y, s2z,
            mass_1, mass_2, dat["phase"], dat["reference_frequency"]
        )
        initial_sphereical = dict(theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,  a_1=a_1, a_2=a_2, phi_12=phi_12, theta_12=theta_12, phi_z_s12=phi_z_s12)

        # roundtrip check
        incl_2, s1x_2, s1y_2, s1z_2, s2x_2, s2y_2, s2z_2 = spherical_to_cartesian_coords(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, dat["reference_frequency"], dat["phase"])
        final_spin_vector = dict(s1x=s1x_2, s1y=s1y_2, s1z=s1z_2, s2x=s2x_2, s2y=s2y_2, s2z=s2z_2, i=incl_2)
        theta_jn_2, phi_jl_2, tilt_1_2, tilt_2_2, phi_1_2, phi_2_2, a_1_2, a_2_2, phi_12_2, theta_12_2, phi_z_s12_2 = cartesian_to_spherical_coords(incl_2,
                                      s1x_2, s1y_2, s1z_2, s2x_2, s2y_2, s2z_2,
                                      mass_1, mass_2, dat["phase"], dat["reference_frequency"])
        final_sphereical = dict(theta_jn=theta_jn_2, phi_jl=phi_jl_2, tilt_1=tilt_1_2, tilt_2=tilt_2_2, a_1=a_1_2, a_2=a_2_2, phi_12=phi_12_2, theta_12=theta_12_2, phi_z_s12=phi_z_s12_2)

        diff = DeepDiff(initial_spin_vector, final_spin_vector, math_epsilon=0.001)
        if len(diff) != 0:
            print(f"ERROR: roundrip conversion with cartesian: {diff}")

        diff = DeepDiff(initial_sphereical, final_sphereical, math_epsilon=0.001)
        if len(diff) != 0:
            self.fail(f"ERROR: roundrip conversion with spin: {diff}")



if __name__ == '__main__':
    unittest.main()
