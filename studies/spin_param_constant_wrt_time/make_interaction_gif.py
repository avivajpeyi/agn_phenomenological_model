import glob
import logging
import multiprocessing
import os
import shutil
import time

from matplotlib.pyplot import imshow
import bilby
import corner
import lalsimulation
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from bilby.core.prior import Cosine, Interped, Uniform
from bilby_report.tools import image_utils
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import iid_spin_magnitude_beta, truncnorm
from matplotlib import rcParams
from numpy import cos, sin
from tqdm import tqdm


def join_images(*rows, bg_color=(0, 0, 0, 0), alignment=(0.5, 0.5)):
    rows = [
        [image.convert('RGBA') for image in row]
        for row
        in rows
    ]

    heights = [
        max(image.height for image in row)
        for row
        in rows
    ]

    widths = [
        max(image.width for image in column)
        for column
        in zip(*rows)
    ]

    tmp = Image.new(
        'RGBA',
        size=(sum(widths), sum(heights)),
        color=bg_color
    )

    for i, row in enumerate(rows):
        for j, image in enumerate(row):
            y = sum(heights[:i]) + int((heights[i] - image.height) * alignment[1])
            x = sum(widths[:j]) + int((widths[j] - image.width) * alignment[0])
            tmp.paste(image, (x, y))

    return tmp


def join_images_horizontally(*row, bg_color=(0, 0, 0), alignment=(0.5, 0.5)):
    return join_images(
        row,
        bg_color=bg_color,
        alignment=alignment
    )


def join_images_vertically(*column, bg_color=(0, 0, 0), alignment=(0.5, 0.5)):
    return join_images(
        *[[image] for image in column],
        bg_color=bg_color,
        alignment=alignment
    )


rcParams["font.size"] = 20
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams['axes.labelsize'] = 30
rcParams['axes.titlesize'] = 30
rcParams['axes.labelpad'] = 20
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

BILBY_BLUE_COLOR = '#0072C1'
VIOLET_COLOR = "#8E44AD"

PARAMS = dict(
    chi_eff=dict(l=r"$\chi_{\mathrm{eff}}$", r=(-1, 1)),
    chi_p=dict(l=r"$\chi_{\mathrm{p}}$", r=(0, 1)),
    cos_tilt_1=dict(l=r"$\cos \mathrm{tilt}_1$", r=(-1, 1)),
    cos_tilt_2=dict(l=r"$\cos \mathrm{tilt}_2$", r=(-1, 1)),
    cos_theta_12=dict(l=r"$\cos \theta_{12}$", r=(-1, 1)),
    # phi_1=dict(l=r"$\phi_{1}$", r=(0, 2 * np.pi)),
    # phi_2=dict(l=r"$\phi_{2}$", r=(0, 2 * np.pi)),
    # a_1=dict(l=r"$a_1$", r=(0, 1)),
    # a_2=dict(l=r"$a_2$", r=(0, 1)),
    mass_ratio=dict(l=r"q", r=(0, 1)),
)

CORNER_KWARGS = dict(
    smooth=0.85,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color='tab:orange',
    # quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)

logging.getLogger("bilby").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.INFO)

REFERENCE_FREQ = 10 ** -3
NUM = 10000

POP_MODEL_VALS = {
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
    "amax": 1.0,
    "lamb": 0.0,
}


def timing(function):
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) / 60.0
        duration_sec = (end_time - start_time) % 60.0
        f_name = function.__name__
        logging.debug(f"{f_name} took {int(duration)}min {int(duration_sec)}s")

        return result

    return wrap


def get_samples_from_prior(num_samples, prior_fname):
    prior = bilby.gw.prior.PriorDict(filename=prior_fname)
    samples = prior.sample(num_samples)
    samples = bilby.gw.conversion.generate_all_bbh_parameters(samples)
    samples = add_agn_samples_to_df(samples)
    return pd.DataFrame(samples)


def add_agn_samples_to_df(df):
    df['s1x'], df['s1y'], df['s1z'] = df['spin_1x'], df['spin_1y'], df['spin_1z']
    df['s2x'], df['s2y'], df['s2z'] = df['spin_2x'], df['spin_2y'], df['spin_2z']
    df['s1_dot_s2'] = (df['s1x'] * df['s2x']) + (df['s1y'] * df['s2y']) + (
            df['s1z'] * df['s2z'])
    df['s1_mag'] = np.sqrt(
        (df['s1x'] * df['s1x']) + (df['s1y'] * df['s1y']) + (df['s1z'] * df['s1z']))
    df['s2_mag'] = np.sqrt(
        (df['s2x'] * df['s2x']) + (df['s2y'] * df['s2y']) + (df['s2z'] * df['s2z']))
    df['cos_theta_12'] = df['s1_dot_s2'] / (df['s1_mag'] * df['s2_mag'])
    return df



def generate_population_prior(p):
    p, _ = convert_to_beta_parameters(p)
    # make grid_x-vals
    mass = np.linspace(5, 100, num=NUM)
    q = np.linspace(0, 1, num=NUM)
    cos_vals = np.linspace(-1, 1, num=NUM)
    a = np.linspace(0, 1, num=NUM)
    z = np.linspace(0, 2.3, num=NUM)

    # calcualte probabilites
    mass_model = SinglePeakSmoothedMassDistribution()
    p_mass = mass_model.p_m1(
        dataset=pd.DataFrame(dict(mass_1=mass)),
        alpha=p['alpha'], mmin=p['mmin'], mmax=p['mmax'], lam=p['lam'], mpp=p['mpp'],
        sigpp=p['sigpp'], delta_m=p['delta_m'])
    p_q = mass_model.p_q(
        dataset=pd.DataFrame(dict(mass_ratio=q, mass_1=mass)), beta=p["beta"],
        mmin=p["mmin"], delta_m=p["delta_m"])
    p_costheta12 = truncnorm(xx=cos_vals, mu=1, sigma=p['sigma_12'], high=1,
                             low=-1)
    p_costilt1 = truncnorm(xx=cos_vals, mu=1, sigma=p['sigma_1'], high=1, low=-1)
    p_a = iid_spin_magnitude_beta(
        dataset=pd.DataFrame(dict(a_1=a, a_2=a)),
        amax=p['amax'], alpha_chi=p['alpha_chi'],
        beta_chi=p['beta_chi'])
    p_z = PowerLawRedshift(z_max=2.3).probability(
        dataset=pd.DataFrame(dict(redshift=z)), lamb=p['lamb'])

    # after generating prior, generate samples, then convert the samples to BBH params
    priors = bilby.prior.PriorDict(dict(
        # a_1=Interped(a, p_a, minimum=0, maximum=1, name='a_1', latex_label="$a_1$"),
        # a_2=Interped(a, p_a, minimum=0, maximum=1, name='a_2', latex_label="$a_2$"),
        a_1=Uniform(minimum=0, maximum=0.9, name='a_1', latex_label="$a_1$"),
        a_2=Uniform(minimum=0, maximum=0.9, name='a_2', latex_label="$a_2$"),
        redshift=Interped(z, p_z, minimum=0, maximum=2.3, name='redshift',
                          latex_label="$pred_z$"),
        cos_tilt_1=Interped(cos_vals, p_costilt1, minimum=-1, maximum=1,
                            name='cos_tilt_1', latex_label="$\\cos\ \\mathrm{tilt}_1$"),
        cos_theta_12=Interped(cos_vals, p_costheta12, minimum=-1, maximum=1,
                              name='cos_theta_12', latex_label="$\\cos\ \\theta_{12}$"),
        mass_1_source=Interped(mass, p_mass, minimum=5, maximum=100,
                               name='mass_1_source', latex_label="$m_{1}$"),
        # mass_ratio=Interped(q, p_q, minimum=0, maximum=1, name='mass_ratio', latex_label="$q$"),
        mass_ratio=Uniform(minimum=0, maximum=0.9, name='q', latex_label="$q$"),
        dec=Cosine(name='dec'),
        ra=Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'),
        psi=Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'),
        phi_1=Uniform(name="phi_1", minimum=0, maximum=2 * np.pi, boundary='periodic',
                      latex_label="$\\phi_1$"),
        phi_12=Uniform(name="phi_12", minimum=0, maximum=2 * np.pi, boundary='periodic',
                       latex_label="$\\phi_{12}$"),
        phase=Uniform(name="phase", minimum=0, maximum=2 * np.pi, boundary='periodic'),
        incl=Uniform(name="incl", minimum=0, maximum=2 * np.pi, boundary='periodic'),
        geocent_time=Uniform(minimum=-0.1, maximum=0.1, name="geocent_time",
                             latex_label="$t_c$", unit="$s$")
    ))
    return priors


def generate_s1_to_z_rotation_matrix(theta, phi):
    return np.array((
        (cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)),
        (-sin(phi), cos(phi), 0),
        (cos(phi) * sin(theta), sin(theta) * sin(phi), cos(theta))
    ))


@np.vectorize
def generate_agn_spins(θ12, ϕ12, θ1, ϕ1):
    """https://www.wolframcloud.com/obj/d954c5d7-c296-40c5-b776-740441099c40"""
    s2 = [
        -(cos(θ12) * sin(θ1)) + cos(θ1) * cos(ϕ1 - ϕ12) * sin(θ12),
        sin(θ12) * sin(ϕ1 - ϕ12),
        cos(θ1) * cos(θ12) + cos(ϕ1 - ϕ12) * sin(θ1) * sin(θ12)
    ]
    tilt_2 = np.arccos(s2[2])
    phi_2 = np.arctan2(s2[1], s2[0])
    if phi_2 < 0:
        phi_2 += 2.0 * np.pi

    return tilt_2, phi_2


def get_mass_samples(m1_source, q, z):
    m1 = m1_source * (1 + np.array(z))
    m2 = m1 * q
    return m1, m2


@np.vectorize
def transform_component_spins(incl=2, S1x=0, S1y=0, S1z=1, S2x=0, S2y=0, S2z=1, m1=20,
                              m2=20, phase=0):
    """https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html#ga6920c640f473e7125f9ddabc4398d60a"""
    thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2 = (
        lalsimulation.SimInspiralTransformPrecessingWvf2PE(
            incl=incl, S1x=S1x, S1y=S1y, S1z=S1z, S2x=S2x, S2y=S2y, S2z=S2z, m1=m1,
            m2=m2, fRef=REFERENCE_FREQ, phiRef=phase
        ))
    return thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2


@np.vectorize
def make_spin_vector(theta, phi):
    return (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))


def mult_magn_to_vect(mags, vecs):
    return np.array([m * v for m, v in zip(mags, vecs)])


@timing
def get_samples(num_samples=10000, population_params={}):
    poolname = multiprocessing.current_process().name
    logging.debug(f"{poolname}: Drawing samples from Population Prior")
    params = POP_MODEL_VALS.copy()
    params.update(population_params)
    pop_model = generate_population_prior(params)
    s = pd.DataFrame(pop_model.sample(num_samples)).to_dict('list')
    phi_1, z = s["phi_1"], s["redshift"]
    m1, m2 = get_mass_samples(s["mass_1_source"], s["mass_ratio"], z)
    tilt_1, theta_12 = np.arccos(s["cos_tilt_1"]), np.arccos(s["cos_theta_12"])
    tilt_2, phi_2 = generate_agn_spins(θ12=theta_12, ϕ12=s["phi_12"], θ1=tilt_1,
                                       ϕ1=phi_1)
    lumin_dist = bilby.gw.conversion.redshift_to_luminosity_distance(z)
    s1 = mult_magn_to_vect(np.array(s["a_1"]),
                           np.array(make_spin_vector(tilt_1, phi_1)).transpose())
    s2 = mult_magn_to_vect(np.array(s["a_2"]),
                           np.array(make_spin_vector(tilt_2, phi_2)).transpose())
    s1x, s1y, s1z = s1[:, 0], s1[:, 1], s1[:, 2]
    s2x, s2y, s2z = s2[:, 0], s2[:, 1], s2[:, 2]
    theta_jn, phi_jl, _, _, _, _, _ = (
        transform_component_spins(
            incl=s['incl'], S1x=s1x, S1y=s1y, S1z=s1z, S2x=s2x, S2y=s2y, S2z=s2z,
            m1=m1 * bilby.utils.solar_mass, m2=m2 * bilby.utils.solar_mass,
            phase=s['phase']
        ))
    chi_eff = get_chi_eff(s1z=s1z, s2z=s2z, q=s["mass_ratio"])
    chi_p = get_chi_p(s1x=s1x, s1y=s1y, s2x=s2x, s2y=s2y, q=s["mass_ratio"])
    s['chi_eff'] = chi_eff
    s['chi_p'] = chi_p
    s['phi_2'] = phi_2
    s['mass_1'] = m1
    s['mass_2'] = m2
    s['luminosity_distance'] = lumin_dist
    s['tilt_1'] = tilt_1
    s['tilt_2'] = tilt_2
    s['cos_tilt_2'] = np.cos(tilt_2)
    s['theta_jn'] = theta_jn
    s['phi_jl'] = phi_jl
    return pd.DataFrame(s)


@np.vectorize
def get_chi_eff(s1z, s2z, q):
    # return (s1z * s2z) * (q / (1 + q))
    return (s1z + s2z * q) / (1 + q)


@np.vectorize
def get_chi_p(s1x, s1y, s2x, s2y, q):
    chi1p = np.sqrt(s1x ** 2 + s1y ** 2)
    chi2p = np.sqrt(s2x ** 2 + s2y ** 2)
    qfactor = q * ((4 * q) + 3) / (4 + (3 * q))
    return np.maximum(chi1p, chi2p * qfactor)


def sample_from_agn_population(sigma_1, sigma_12, n=NUM):
    samples = get_samples(
        num_samples=n,
        population_params=dict(sigma_1=sigma_1, sigma_12=sigma_12)
    )
    return samples[[k for k in PARAMS.keys()]]


def overlaid_corner(samples_list, sample_labels,
                    samples_colors, params, fname, title, show_legend=True):
    """Plots multiple corners on top of each other"""
    # print(f"plotting {fname}")
    # sort the sample columns
    samples_list = [s[params] for s in samples_list]

    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = min([len(s) for s in samples_list])

    CORNER_KWARGS.update(
        range=[PARAMS[p]['r'] for p in params],
        labels=[PARAMS[p]['l'] for p in params],
    )

    fig = corner.corner(
        samples_list[0],
        color=samples_colors[0],
        **CORNER_KWARGS,
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), min_len),
            color=samples_colors[idx],
            **CORNER_KWARGS
        )
    if show_legend:
        plt.legend(
            handles=[
                mlines.Line2D([], [], color=samples_colors[i], label=sample_labels[i])
                for i in range(n)
            ],
            fontsize=20, frameon=False,
            bbox_to_anchor=(1, ndim), loc="upper right"
        )
    plt.suptitle(title)
    fig.savefig(fname)
    plt.close(fig)


def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)


def plot_overlaid_corners(sigma_1_list, sigma_12_list, pltdir, n=NUM,
                          show_aligned=False, show_isotropic=False, show_agn=True):
    isotropic_samples = get_samples_from_prior(num_samples=n,
                                               prior_fname="isotropic.prior")
    aligned_samples = get_samples_from_prior(num_samples=n, prior_fname="aligned.prior")

    if os.path.isdir(pltdir):
        assert pltdir != '.'
        shutil.rmtree(pltdir)
    os.makedirs(pltdir, exist_ok=False)

    i = 0
    for i_sig1, i_sig12 in tqdm(
            zip(sigma_1_list, sigma_12_list),
            total=len(sigma_1_list), desc="Hyper-Param settings"
    ):
        f_theta = f"{pltdir}/{i:02}_sig12-{i_sig12:.1f}_sig1-{i_sig1:.1f}_theta.png"
        f_chi = f"{pltdir}/{i:02}_sig12-{i_sig12:.1f}_sig1-{i_sig1:.1f}_chi.png"
        f_combo = f"{pltdir}/{i:02}_sig12-{i_sig12:.1f}_sig1-{i_sig1:.1f}_combo.png"
        agn_label = r"$\sigma_{1}=" + f"{i_sig1:.1f}" + r", \sigma_{12}=" + f"{i_sig12:.1f}$"

        samples_list, samples_colors, samples_labels = [], [], []
        if show_aligned:
            samples_list.append(aligned_samples)
            samples_colors.append("tab:green")
            samples_labels.append("Field")
        if show_isotropic:
            samples_list.append(isotropic_samples)
            samples_colors.append("tab:orange")
            samples_labels.append("Dynamic")
        if show_agn:
            agn_samples = sample_from_agn_population(
                sigma_1=i_sig1,
                sigma_12=i_sig12,
                n=n
            )
            samples_list.append(agn_samples)
            samples_colors.append("tab:blue")
            samples_labels.append("AGN " + agn_label)

        overlaid_corner(
            samples_list, samples_labels, samples_colors,
            params=["chi_eff", "chi_p"],
            fname=f_chi, title="", show_legend=False
        )
        overlaid_corner(
            samples_list, samples_labels, samples_colors,
            params=["cos_tilt_1", "cos_tilt_2", "cos_theta_12"],
            fname=f_theta, title=""
        )

        images = [
            [Image.open(f_chi), Image.open(f_theta)]
        ]

        combined_image = join_images(
            *images,
            bg_color='white',
            alignment=(0.5, 0.5)
        )
        combined_image.save(f_combo)
        i += 1


ISO_SAMPLES = get_samples_from_prior(num_samples=1000, prior_fname="isotropic.prior")
ALIGNED_SAMPLES = get_samples_from_prior(num_samples=1000, prior_fname="aligned.prior")


def plot_one_corner(show_aligned, show_isotropic, show_agn, n=NUM, sig1=0, sig12=0):
    plt.close('all')
    f_theta = f"tmp_theta.png"
    f_chi = f"tmp_chi.png"
    agn_label = r"$\sigma_{1}=" + f"{sig1:.1f}" + r", \sigma_{12}=" + f"{sig12:.1f}$"
    samples_list, samples_colors, samples_labels = [], [], []
    if show_aligned:
        samples_list.append(ALIGNED_SAMPLES)
        samples_colors.append("tab:green")
        samples_labels.append("Field")
    if show_isotropic:
        samples_list.append(ISO_SAMPLES)
        samples_colors.append("tab:orange")
        samples_labels.append("Dynamic")
    if show_agn:
        agn_samples = sample_from_agn_population(
            sigma_1=sig1,
            sigma_12=sig12,
            n=n
        )
        samples_list.append(agn_samples)
        samples_colors.append("tab:blue")
        samples_labels.append("AGN " + agn_label)

    overlaid_corner(
        samples_list, samples_labels, samples_colors,
        params=["chi_eff", "chi_p"],
        fname=f_chi, title="", show_legend=False
    )
    overlaid_corner(
        samples_list, samples_labels, samples_colors,
        params=["cos_tilt_1", "cos_tilt_2", "cos_theta_12"],
        fname=f_theta, title=""
    )

    images = [
        [Image.open(f_chi), Image.open(f_theta)]
    ]

    combined_image = join_images(
        *images,
        bg_color='white',
        alignment=(0.5, 0.5)
    )
    plt.axis('off')
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.asarray(combined_image),  aspect='equal')


def save_gif(gifname, regex="*.png", outdir="gif", loop=False, ):
    image_paths = glob.glob(f"{outdir}/{regex}")
    gif_filename = os.path.join(outdir, gifname)
    orig_len = len(image_paths)
    image_paths.sort()
    if loop:
        image_paths += image_paths[::-1]
    assert orig_len <= len(image_paths)
    image_utils.make_gif(
        image_paths=image_paths,
        duration=200,
        gif_save_path=gif_filename
    )
    print(f"Saved gif {gif_filename}")


# def plot_one_corner(sigma_1=0.01, sigma_12=0.01):
#     n = 10000
#     plot_overlaid_corners([0],[0], pltdir="aligned_spin", n=n, show_aligned=True, show_agn=False)
#     plot_overlaid_corners([100], [100], pltdir="iso_spin", n=n, show_isotropic=True, show_agn=False)


def gif_main():
    sigma12_vals = list(np.arange(0.3, 3, 0.3))
    sigma1_vals = [1 for _ in range(len(sigma12_vals))]

    outdir = "vary_sigma12"
    plot_overlaid_corners(sigma_1_list=sigma1_vals,
                          sigma_12_list=sigma12_vals, pltdir=outdir)
    save_gif("vary_12.gif", outdir=outdir, loop=True, regex="*_combo.png")

    sigma1_vals = list(np.arange(0.3, 3, 0.3))
    sigma12_vals = [0.1 for _ in range(len(sigma1_vals))]

    outdir = "vary_sigma1"
    plot_overlaid_corners(sigma_1_list=sigma1_vals,
                          sigma_12_list=sigma12_vals, pltdir=outdir)
    save_gif("vary_1.gif", outdir=outdir, loop=True, regex="*_combo.png")


def test_main():
    # plot_one_corner(sigma_1=0.01, sigma_12=0.01)
    gif_main()


if __name__ == '__main__':
    test_main()
