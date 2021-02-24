
import warnings

import bilby
import deepdish as dd
import matplotlib.pyplot as plt
import pandas as pd
from bilby.core.prior import Cosine, Interped, Sine, Uniform
from bilby.hyper.model import Model
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.cupy_utils import to_numpy
from gwpopulation.cupy_utils import trapz
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import iid_spin_magnitude_beta, truncnorm
from matplotlib import rcParams

warnings.filterwarnings("ignore")

rcParams["font.size"] = 20
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 20
rcParams["font.size"] = 30
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 10
rcParams['axes.linewidth'] = 2.5
rcParams['axes.edgecolor'] = 'black'
rcParams['xtick.labelsize'] = 25
rcParams['xtick.major.size'] = 10.0
rcParams['xtick.minor.size'] = 5.0
rcParams['ytick.labelsize'] = 25
rcParams['ytick.major.size'] = 10.0
rcParams['ytick.minor.size'] = 5.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

from tqdm import tqdm
import numpy as np

MAX_SAMPLES = 500

CORNER_KWARGS = dict(
    smooth=0.99,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color='tab:orange',
    quantiles=(0.16, 0.84),
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)

O3A_RES = '../data/Population_Samples/default/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json'


def load_O3a_results():
    return bilby.core.result.read_in_result(O3A_RES)


def get_posterior_sample_with_highest_lnl(res: bilby.core.result.Result):
    posterior = res.posterior
    samples = posterior.to_dict('records')
    post_prob = [res.posterior_probability(s)[0] for s in tqdm(samples)]
    posterior['post_prob'] = post_prob
    # max_lnl = posterior[posterior.log_likelihood == posterior.log_likelihood.max()]
    max_lnl = posterior[posterior.post_prob == posterior.post_prob.max()]
    max_lnl_dict = max_lnl.to_dict('records')[-1]
    print(max_lnl_dict)
    return max_lnl_dict


def generate_prior(p, make_plot=False):
    p, _ = convert_to_beta_parameters(p)
    NUM = 200
    mass = np.linspace(5, 100, num=NUM)
    q = np.linspace(0, 1, num=NUM)
    cos_vals = np.linspace(-1, 1, num=NUM)
    a = np.linspace(0, 1, num=NUM)
    z = np.linspace(0, 2.3, num=NUM)
    margs = dict(alpha=p['alpha'], mmin=p['mmin'],
                 mmax=p['mmax'], lam=p['lam'],
                 mpp=p['mpp'], sigpp=p['sigpp'], delta_m=p['delta_m'])
    mass_model = SinglePeakSmoothedMassDistribution()
    p_mass = mass_model.p_m1(dataset=pd.DataFrame(dict(mass_1=mass)), **margs)
    p_q = mass_model.p_q(dataset=pd.DataFrame(dict(mass_ratio=q, mass_1=mass)),
                         beta=p["beta"], mmin=p["mmin"], delta_m=p["delta_m"])
    p_costheta12 = truncnorm(xx=cos_vals, mu=1, sigma=p['sigma_12'], high=1, low=-1)
    p_costilt1 = truncnorm(xx=cos_vals, mu=1, sigma=p['sigma_1'], high=1, low=-1)
    p_a = iid_spin_magnitude_beta(dataset=pd.DataFrame(dict(a_1=a, a_2=a)),
                                  amax=p['amax'], alpha_chi=p['alpha_chi'],
                                  beta_chi=p['beta_chi'])
    p_z = PowerLawRedshift(z_max=2.3).probability(
        dataset=pd.DataFrame(dict(redshift=z)), lamb=p['lamb'])



    # after generating prior, generate samples, then convert the samples to BBH params

    priors = bilby.prior.PriorDict(dict(
        a_1=Interped(a, p_a, minimum=0, maximum=1, name='a_1', latex_label="$a_1$"),
        a_2=Interped(a, p_a, minimum=0, maximum=1, name='a_2', latex_label="$a_2$"),
        redshift=Interped(z, p_z, minimum=0, maximum=2.3, name='redshift',
                          latex_label="$z$"),
        cos_tilt_1=Interped(cos_vals, p_costilt1, minimum=-1, maximum=1,
                            name='cos_tilt_1', latex_label="$\\cos\ \\mathrm{tilt}_1$"),
        cos_theta_12=Interped(cos_vals, p_costheta12, minimum=-1, maximum=1,
                              name='cos_theta_12', latex_label="$\\cos\ \\theta_{12}$"),
        mass_1_source=Interped(mass, p_mass, minimum=5, maximum=100,
                               name='mass_1_source', latex_label="$m_{1}$"),
        mass_ratio=Interped(q, p_q, minimum=0, maximum=1, name='mass_ratio',
                            latex_label="$q$"),
        # Detection
        dec=Cosine(name='dec'),
        ra=Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
        theta_jn=Sine(name='theta_jn'),
        psi=Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
        phase=Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic'),
        phi_12=Uniform(name='phi_12', minimum=0, maximum=2 * np.pi,
                       boundary='periodic'),
        phi_jl=Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')
    ))

    if make_plot:
        fig, axs = plt.subplots(6, figsize=(5, 18))
        i = 0
        mass_ax = axs[i]
        mass_ax.plot(mass, p_mass)
        mass_ax.set_ylabel("$p(m\ \\mathrm{source})$")
        mass_ax.set_xlabel("$m\ \\mathrm{source}$")
        mass_ax.set_yscale('log')
        mass_ax.set_xlim(p['mmin'], p['mmax'])
        mass_ax.set_ylim(1e-5, 5e-1)
        i += 1
        q_ax = axs[i]
        q_ax.plot(q, p_q)
        q_ax.set_ylabel("$p(q)$")
        q_ax.set_xlabel("$q$")
        q_ax.set_xlim(0, 1)
        i += 1
        spin_mag_ax = axs[i]
        spin_mag_ax.plot(a, p_a)
        spin_mag_ax.set_ylabel("$p(|a|)$")
        spin_mag_ax.set_xlabel("$|a|$")
        spin_mag_ax.set_xlim(0, 1)
        i += 1
        costheta12_ax = axs[i]
        costheta12_ax.plot(cos_vals, p_costheta12)
        costheta12_ax.set_ylabel("$p(\\cos\ \\theta_{12})$")
        costheta12_ax.set_xlabel("$\\cos\ \\theta_{12}$")
        costheta12_ax.set_xlim(-1, 1)
        i += 1
        costilt1_ax = axs[i]
        costilt1_ax.plot(cos_vals, p_costilt1)
        costilt1_ax.set_ylabel("$p(\\cos \\mathrm{tilt}_1)$")
        costilt1_ax.set_xlabel("$\\cos \\mathrm{tilt}_1$")
        costheta12_ax.set_xlim(-1, 1)
        i += 1
        redshift_ax = axs[i]
        redshift_ax.plot(z, p_z)
        redshift_ax.set_ylabel("$p(z)$")
        redshift_ax.set_xlabel("$z$")
        redshift_ax.set_xlim(0, 2.3)

        axs[0].set_title("Population Distributions")
        plt.tight_layout()
        plt.savefig("prior.png")

    return priors


def get_isotropic_vectors(num_points=100):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0, 2*np.pi, size=num_points)
    costheta = np.random.uniform(-1, 1, size=num_points)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def mass_spectrum_plot(result, hyper_param_vals):
    limits = [5, 95]
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    mass_1_grid, mass_ratio_grid = np.meshgrid(mass_1, mass_ratio)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    peak_1 = 0
    peak_2 = 0

    data = dict(mass_1=mass_1_grid, mass_ratio=mass_ratio_grid)
    lines = dict(mass_1=list(), mass_ratio=list())
    ppd = np.zeros_like(data["mass_1"])

    if len(result.posterior) > MAX_SAMPLES:
        samples = result.posterior.sample(MAX_SAMPLES)
    else:
        samples = result.posterior

    model = Model([SinglePeakSmoothedMassDistribution()])

    for ii in tqdm(range(len(samples))):
        parameters = dict(samples.iloc[ii])
        model.parameters.update(parameters)
        prob = model.prob(data)
        ppd += prob
        mass_1_prob = trapz(prob, mass_ratio, axis=0)
        mass_ratio_prob = trapz(prob, mass_1, axis=-1)

        lines["mass_1"].append(mass_1_prob)
        lines["mass_ratio"].append(mass_ratio_prob)
    for key in lines:
        lines[key] = np.vstack([to_numpy(line) for line in lines[key]])

    ppd /= len(samples)
    ppd = to_numpy(ppd)

    mass_1 = to_numpy(mass_1)
    mass_ratio = to_numpy(mass_ratio)

    mass_1_ppd = np.trapz(ppd, mass_ratio, axis=0)
    mass_ratio_ppd = np.trapz(ppd, mass_1, axis=-1)

    label = " ".join(result.label.split("_")).title()

    model.parameters.update(hyper_param_vals)
    my_p = model.prob(data)
    my_mass_1_prob = trapz(my_p, mass_ratio, axis=0)
    my_mass_ratio_prob = trapz(my_p, mass_1, axis=-1)

    axs[0].semilogy(mass_1, mass_1_ppd, label="O3a")
    axs[0].semilogy(mass_1, my_mass_1_prob, label="Selected Pop-Model", color='r')
    axs[0].fill_between(
        mass_1,
        np.percentile(lines["mass_1"], limits[0], axis=0),
        np.percentile(lines["mass_1"], limits[1], axis=0),
        alpha=0.5,
    )
    _peak_1 = max(np.percentile(lines["mass_1"], limits[1], axis=0))
    peak_1 = max(peak_1, _peak_1)
    axs[1].semilogy(mass_ratio, mass_ratio_ppd)
    axs[1].semilogy(mass_ratio, my_mass_ratio_prob, color='r')
    axs[1].fill_between(
        mass_ratio,
        np.percentile(lines["mass_ratio"], limits[0], axis=0),
        np.percentile(lines["mass_ratio"], limits[1], axis=0),
        alpha=0.5,
    )
    _peak_2 = max(np.percentile(lines["mass_ratio"], limits[1], axis=0))
    peak_2 = max(peak_2, _peak_2)
    filename = f"{result.outdir}/{result.label}_mass_data.h5"
    dd.io.save(filename, data=dict(lines=lines, ppd=ppd))

    axs[0].set_xlim(2, 100)
    axs[0].set_ylim(peak_1 / 1000, peak_1 * 1.1)
    axs[0].set_xlabel("$m_{1}$ [$M_{\\odot}$]")
    axs[0].legend(loc="best")
    ylabel = "$p(m_{1})$ [$M_{\\odot}^{-1}$]"
    axs[0].set_ylabel(ylabel)

    axs[1].set_xlim(0.1, 1)
    axs[1].set_ylim(peak_2 / 10000, peak_2 * 1.1)
    axs[1].set_xlabel("$q$")
    axs[1].ylabel = "$p(q)$"
    axs[1].set_ylabel(ylabel)

    file_name = f"{result.outdir}/{result.label}_mass_spectrum.pdf"
    plt.tight_layout()

    plt.savefig(file_name, format="pdf", dpi=600)
    plt.close()


PARAMS = ['alpha', 'beta', 'mmax', 'mmin', 'lam', 'mpp', 'sigpp', 'delta_m',
          'mu_chi', 'sigma_chi', 'xi_spin', 'sigma_spin']

HYPER_PARAM_VALS = {
    "alpha": 2.62,
    "beta": 1.26,
    "mmax": 86.73,
    "mmin": 4.5,
    "lam": 0.12,
    "mpp": 33.5,
    "sigpp": 5.09,
    "delta_m": 4.88,
    "mu_chi": 0.25,
    "sigma_chi": 0.03,
    "sigma_1": 0.5,
    "sigma_12": 2,
    "amax": 1.0,
    "lamb": 0.0,
}


def pretty_plots():
    r = load_O3a_results()
    r.outdir = '.'
    truths = [v for k, v in HYPER_PARAM_VALS.items() if k in PARAMS]
    r.plot_corner(parameters=PARAMS, truths=truths)
    mass_spectrum_plot(r, HYPER_PARAM_VALS)

def convert_to_bbh_params(samples):
    print(samples)
    s1x, s1y, s1z = get_isotropic_vectors(num_points=1000)
    s2x, s2y, s2z = get_isotropic_vectors(num_points=1000)
    s1_dot_s2

    df['s1x'], df['s1y'], df['s1z'] = df['spin_1x'], df['spin_1y'], df['spin_1z']
    df['s2x'], df['s2y'], df['s2z'] = df['spin_2x'], df['spin_2y'], df['spin_2z']
    df['s1_dot_s2'] = (df['s1x'] * df['s2x']) + (df['s1y'] * df['s2y']) + (
            df['s1z'] * df['s2z'])
    df['s1_mag'] = np.sqrt(
        (df['s1x'] * df['s1x']) + (df['s1y'] * df['s1y']) + (df['s1z'] * df['s1z']))
    df['s2_mag'] = np.sqrt(
        (df['s2x'] * df['s2x']) + (df['s2y'] * df['s2y']) + (df['s2z'] * df['s2z']))
    df['cos_theta_12'] = df['s1_dot_s2'] / (df['s1_mag'] * df['s2_mag'])
    # Lhat = [0, 0, 1]
    df['cos_theta_1L'] = df['s1z'] / (df['s1_mag'])
    df['tilt1'], df['theta_1L'] = np.arccos(df['cos_tilt_1']), np.arccos(
        df['cos_theta_1L'])
    df['diff'] = df['tilt1'] - df['theta_1L']
    df = calculate_weight(df, sigma=0.5)
    return df






if __name__ == "__main__":
    main()
