import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import precession

import agn_utils.conversion.precession_avg_chi_p as chip

LABELS = dict(
    heuristic=r"Original",
    thomas=r"$|\vec{\chi_\perp}|$",
    averaged=r"$\langle\chi_p\rangle$"
)


class ConversionTestCase(unittest.TestCase):

    def compute_different_chi_p(self, q, chi1, chi2, theta1, theta2, deltaphi, M_msun,
                                num_pts=1000):
        params = get_bbh_evolution(q, chi1, chi2, theta1, theta2, deltaphi, num_pts)
        r = params['r']
        q = params['q']
        chi1 = params['chi1']
        chi2 = params['chi2']
        theta1 = params['theta1']
        theta2 = params['theta2']
        deltaphi = params['deltaphi']
        M_msun_array = np.array([M_msun] * num_pts)  # Msun
        freq = precession.rtof(r, M_msun)

        return dict(
            heuristic=chip.chip_heuristic(q, chi1, chi2, theta1, theta2),
            thomas=chip.chip_thomas(q, chi1, chi2, theta1, theta2, deltaphi),
            averaged=chip.chip_averaged(q, chi1, chi2, theta1, theta2, deltaphi,
                                        fref=freq, M_msun=M_msun_array),
            freq=freq,
            r=r

        )

    def test_chip_methods(self):
        kwargs = dict(
            q=0.8,
            chi1=1,
            chi2=1,
            theta1=np.pi / 5,
            theta2=np.pi / 3,
            deltaphi=np.pi / 2,
            M_msun=60
        )
        res = self.compute_different_chi_p(**kwargs)
        self.plot_res(res)

    def plot_res(self, res):
        freq = res.pop('freq')
        r = res.pop('r')
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        for name, chip_data in res.items():
            ax1.plot(freq, chip_data, '-o', label=LABELS[name])
        ax1.set_ylabel(r"$\chi_p$")
        ax1.set_xlabel(r"$f\mathrm{[Hz]}$")
        ax1.set_xscale('log')
        ax1.set_xlim(5, 20)

        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig('chi_p_comparisions.png')


def get_bbh_evolution(q, chi1, chi2, theta1, theta2, deltaphi, num_pts):
    """
    Params defined at r=100/M
    """
    t0 = time.time()
    print("Take a BH binary with q={q:.2f}, chi1={chi1:.2f} and chi2= {chi2:.2f}")
    rs = np.geomspace(100, 10, num_pts)  # separations
    M, m1, m2, S1, S2 = precession.get_fixed(q, chi1, chi2)
    t1v, t2v, dpv = precession.evolve_angles(theta1, theta2, deltaphi, rs, q, S1, S2)
    print("Perform BH binary inspiral")
    print("log10(r/M) \t theta1 \t theta2 \t deltaphi")
    for r, t1, t2, dp in zip(np.log10(rs), t1v, t2v, dpv):
        print("%.0f \t\t %.3f \t\t %.3f \t\t %.3f" % (r, t1, t2, dp))
    t = time.time() - t0
    print("Executed in %.3fs" % t)
    return dict(
        q=np.array([q] * num_pts),
        chi1=np.array([chi1] * num_pts),
        chi2=np.array([chi2] * num_pts),
        theta1=t1v,
        theta2=t2v,
        deltaphi=dpv,
        r=rs
    )


if __name__ == '__main__':
    unittest.main()
