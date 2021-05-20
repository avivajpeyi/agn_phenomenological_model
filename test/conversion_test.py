import unittest

import numpy as np

from agn_utils import conversion


class ConversionTestCase(unittest.TestCase):

    def test_agn_to_norm(self):
        orig_params = dict(
            chieff=0.1956248582096523, chip=0.282842712474619,
            theta1=np.pi / 4, theta2=np.pi / 6,
            deltaphi=np.pi * 3 / 2, q=0.8
        )
        expected_params = dict(chi1=0.4, chi2=0.1)
        new_params = conversion.convert_agn_to_normal(**orig_params)
        new_param_dict = {k: v for k, v in zip(expected_params.keys(), new_params)}
        self.assertEqual(new_param_dict, expected_params)

    def test_norm_to_agn(self):
        orig_params = dict(
            chi1=0.4, chi2=0.1,
            theta1=np.pi / 4, theta2=np.pi / 6,
            deltaphi=np.pi * 3 / 2, q=0.8
        )
        expected_params = dict(chi_eff=0.1956248582096523, chi_p=0.282842712474619)
        new_params = conversion.convert_normal_to_agn(**orig_params)
        new_param_dict = {k: v for k, v in zip(expected_params.keys(), new_params)}
        self.assertEqual(new_param_dict, expected_params)

    def test_roundtrip_chi1_less_than_chi2(self):
        orig_params = dict(
            chi1=0.4, chi2=0.1,
            theta1=np.pi / 4, theta2=np.pi / 6,
            deltaphi=np.pi * 3 / 2, q=0.8
        )

        chieff, chip, chi1, chi2, theta1, theta2, q, phi1, phi2 = conversion.convert_normal_to_agn(**orig_params)
        a1, a2, q, theta1, theta2, phi1, phi2 = conversion.convert_agn_to_normal(chieff, chip, q, theta1, theta2, phi1, phi2)
        final_params = dict(
            theta1=theta1, theta2=theta2,  q=q,  chi1=a1, chi2=a2, phi1=phi1, phi2=phi2
        )

        self.assertEqual(final_params, orig_params)


if __name__ == '__main__':
    unittest.main()
