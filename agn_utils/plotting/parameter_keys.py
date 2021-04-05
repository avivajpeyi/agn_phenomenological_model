import numpy as np

PARAMS = dict(
    chi_eff=dict(latex_label=r"$\chi_{eff}$", range=(-1, 1)),
    chi_p=dict(latex_label=r"$\chi_{p}$", range=(0, 1)),
    cos_tilt_1=dict(latex_label=r"$\cos(t1)$", range=(-1, 1)),
    cos_tilt_2=dict(latex_label=r"$\cos(t2)$", range=(-1, 1)),
    cos_theta_12=dict(latex_label=r"$\cos \theta_{12}$", range=(-1, 1)),
    tilt_1=dict(latex_label=r"$tilt_{1}$", range=(0, np.pi)),
    remnant_kick_mag=dict(latex_label=r'$|\vec{v}_k|\ $km/s', range=(0, 3000)),
    chirp_mass=dict(latex_label="$M_{c}$", range=(5, 200)),
    mass_1_source=dict(latex_label='$m_1^{\\mathrm{source}}$', range=(0, 200)),
    mass_2_source=dict(latex_label='$m_2^{\\mathrm{source}}$', range=(0, 200)),
    luminosity_distance=dict(latex_label='$d_L$', range=(50, 20000)),
    log_snr=dict(latex_label='$\\rm{log}_{10}\ \\rho$)', range=(-1, 3)),
)
