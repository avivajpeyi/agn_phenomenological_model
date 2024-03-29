"""
3D Agn spin visualisation
"""
import logging
import os
import shutil

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import scipy.stats
from bbh_simulator.calculate_kick_vel_from_samples import Samples
from matplotlib import rc
from tqdm import tqdm

logging.getLogger("bbh_simulator").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

rc("text", usetex=True)

N_VEC = "Num BBH"
COS_theta_12 = "cos(theta_12)"
COS_theta_1L = "cos(theta_1L)"

BILBY_BLUE_COLOR = "#0072C1"
VIOLET_COLOR = "#8E44AD"

PARAMS = dict(
    chi_eff=dict(l=r"$\chi_{eff}$", r=(-1, 1)),
    chi_p=dict(l=r"$\chi_{p}$", r=(0, 1)),
    cos_tilt_1=dict(l=r"$\cos(t1)$", r=(-1, 1)),
    cos_tilt_2=dict(l=r"$\cos(t2)$", r=(-1, 1)),
    cos_theta_12=dict(l=r"$\cos \theta_{12}$", r=(-1, 1)),
    cos_theta_1L=dict(l=r"$\cos \theta_{1L}$", r=(-1, 1)),
)


def rotate_vector_along_z(v1, theta):
    """
    |cos tilt   −sin tilt   0| |x|   |x cos tilt − y sin tilt|   |x'|
    |sin tilt    cos tilt   0| |y| = |x sin tilt + y cos tilt| = |y'|
    |  0       0      1| |z|   |        z        |   |z'|
    """
    x, y, z = v1[0], v1[1], v1[2]
    return [
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta),
        z,
    ]


def rotate_vector_along_y(v1, theta):
    """
    | cos tilt    0   sin tilt| |grid_x|   | grid_x cos tilt + pred_z sin tilt|   |grid_x'|
    |   0      1       0| |grid_y| = |         grid_y        | = |grid_y'|
    |−sin tilt    0   cos tilt| |pred_z|   |−grid_x sin tilt + pred_z cos tilt|   |pred_z'|
    """
    x, y, z = v1[0], v1[1], v1[2]
    return [
        x * np.cos(theta) + z * np.sin(theta),
        y,
        -x * np.sin(theta) + z * np.cos(theta),
    ]


def get_isotropic_vector(std=1):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    theta = np.random.uniform(0, np.pi * 2)
    # truncated normal distribution --> peaks at costheta = 1
    # hyperparam --> sigma
    # costheta = np.random.uniform(std, 1)

    mean = 1
    clip_a, clip_b = -1, 1

    if std == 0:
        std = 0.00001
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    costheta = scipy.stats.truncnorm.rvs(
        a=a, b=b, loc=mean, scale=std, size=1
    )[0]
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(theta)
    y = np.sin(theta) * np.sin(theta)
    z = np.cos(theta)
    return [x, y, z]


def rotate_v2_to_v1(v1, v2):
    azimuth = get_azimuth_angle(v1[0], v1[1])
    zenith = get_zenith_angle(v1[2])
    v2 = rotate_vector_along_y(v2, zenith)
    v2 = rotate_vector_along_z(v2, azimuth)
    return v2


def compute_vectors(mesh):
    origin = 0
    vectors = mesh.points - origin
    vectors = normalise_vectors(vectors)
    return vectors


def normalise_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


class SphereAngleAnimation:
    def __init__(self):
        # default parameters
        self.kwargs = {
            "radius": 1,
            N_VEC: 100,
            COS_theta_1L: 1,
            COS_theta_12: 1,
        }
        self.s1_color = "lightblue"
        self.s2_color = "lightgreen"
        self.plotter = self.init_plotter()
        self.add_sliders()
        self.plotter.show("AGN BBH spins")
        self.add_vectors()

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def add_sliders(self):
        LEFT = dict(
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
        )
        MIDDLE = dict(pointa=(0.35, 0.1), pointb=(0.64, 0.1))
        RIGHT = dict(
            pointa=(0.67, 0.1),
            pointb=(0.98, 0.1),
        )

        self.plotter.add_slider_widget(
            callback=lambda value: self(COS_theta_1L, value),
            rng=[0, 1],
            value=1,
            title=f"min {COS_theta_1L}",
            style="modern",
            **LEFT,
        )
        self.plotter.add_slider_widget(
            callback=lambda value: self(COS_theta_12, value),
            rng=[0, 1],
            value=1,
            title=f"min {COS_theta_12}",
            style="modern",
            **MIDDLE,
        )
        self.plotter.add_slider_widget(
            callback=lambda value: self(N_VEC, int(value)),
            rng=[1, 1000],
            value=100,
            title=N_VEC,
            style="modern",
            **RIGHT,
        )

    def init_plotter(self):
        p = pv.Plotter()
        p.add_mesh(pv.Sphere(radius=self.kwargs["radius"]))
        ar_kwgs = dict(
            scale=self.kwargs["radius"] * 2,
            shaft_radius=0.01,
            tip_radius=0.05,
            tip_length=0.1,
        )
        p.add_mesh(pv.Arrow(direction=[1, 0, 0], **ar_kwgs), color="blue")  # x
        p.add_mesh(pv.Arrow(direction=[0, 1, 0], **ar_kwgs), color="red")  # y
        p.add_mesh(
            pv.Arrow(direction=[0, 0, 1], **ar_kwgs), color="green"
        )  # Z
        p.add_legend(
            labels=[
                ["L", "green"],
                ["S1", self.s1_color],
                ["S2", self.s2_color],
            ]
        )
        return p

    def add_vectors(self):
        s1_vectors = [
            get_isotropic_vector(self.kwargs[COS_theta_1L])
            for _ in range(self.kwargs[N_VEC])
        ]
        s2_vectors = [
            get_isotropic_vector(self.kwargs[COS_theta_12])
            for _ in range(self.kwargs[N_VEC])
        ]
        s2_vectors = [
            rotate_v2_to_v1(s1, s2) for s1, s2 in zip(s1_vectors, s2_vectors)
        ]

        self.add_vector_list(s1_vectors, name="s1", color=self.s1_color)
        self.add_vector_list(s2_vectors, name="s2", color=self.s2_color)

    def add_vector_list(self, vectors, name, color):
        self.plotter.remove_actor(f"{name}_pts")
        self.plotter.remove_actor(f"{name}_arrows")
        pt_cloud = pv.PolyData(vectors)
        vectors = compute_vectors(pt_cloud)
        pt_cloud["vectors"] = vectors
        arrows = pt_cloud.glyph(
            orient="vectors",
            scale=False,
            factor=0.3,
        )
        self.plotter.add_mesh(
            pt_cloud,
            color=color,
            point_size=10,
            render_points_as_spheres=True,
            name=f"{name}_pts",
        )
        self.plotter.add_mesh(arrows, color=color, name=f"{name}_arrows")

    def update(self):
        self.add_vectors()


def get_zenith_angle(z):
    """Angle from z to vector [0, pi)"""
    return np.arccos(z)


def get_azimuth_angle(x, y):
    """angle bw north vector and projected vector on the horizontal plane [0, 2pi]"""
    azimuth = np.arctan2(y, x)  # [-pi, pi)
    if azimuth < 0.0:
        azimuth += 2 * np.pi
    return azimuth


def get_chi_eff(s1, s2, q=1):
    s1z, s2z = s1[2], s2[2]
    return (s1z * s2z) * (q / (1 + q))


def get_chi_p(s1, s2, q=1):
    chi1p = np.sqrt(s1[0] ** 2 + s1[1] ** 2)
    chi2p = np.sqrt(s2[0] ** 2 + s2[1] ** 2)
    qfactor = q * ((4 * q) + 3) / (4 + (3 * q))
    return np.maximum(chi1p, chi2p * qfactor)


N = 1000


def convert_vectors_to_bbh_param(cos_theta1L_std, cos_theta12_std):
    """Generate BBH spin vectors and convert to LIGO BBH params
    cos_tilt_i:
        Cosine of the zenith angle between the s and j [-1,1]
    theta_12:
        diff bw azimuthal angles of the s1hat+s2 projections on orbital plane [0, 2pi]
    theta_jl:
        diff bw L and J azimuthal angles [0, 2pi]
    """
    n = N
    lhat = normalise_vectors([[0, 0, 1] for _ in range(n)])
    s1hat = normalise_vectors(
        [get_isotropic_vector(cos_theta1L_std) for _ in range(n)]
    )
    s2hat = normalise_vectors(
        [get_isotropic_vector(cos_theta12_std) for _ in range(n)]
    )
    s2hat = normalise_vectors(
        [rotate_v2_to_v1(s1v, s2v) for s1v, s2v in zip(s1hat, s2hat)]
    )

    df = pd.DataFrame(
        dict(
            spin_1x=s1hat[:, 0],
            spin_1y=s1hat[:, 1],
            spin_1z=s1hat[:, 2],
            spin_2x=s2hat[:, 0],
            spin_2y=s2hat[:, 1],
            spin_2z=s2hat[:, 2],
            cos_tilt_1=np.cos([get_zenith_angle(v[2]) for v in s1hat]),
            cos_tilt_2=np.cos([get_zenith_angle(v[2]) for v in s2hat]),
            chi_eff=[get_chi_eff(s1, s2) for s1, s2 in zip(s1hat, s2hat)],
            chi_p=[get_chi_p(s1, s2) for s1, s2 in zip(s1hat, s2hat)],
            cos_theta_12=[
                np.cos(get_angle_bw_vectors(s1, s2))
                for s1, s2 in zip(s1hat, s2hat)
            ],
            cos_theta_1L=[
                np.cos(get_angle_bw_vectors(s1, l))
                for s1, l in zip(s1hat, lhat)
            ],
            mass_1_source=[25 for _ in s1hat],
            mass_2_source=[25 for _ in s1hat],
        )
    )
    s = Samples(posterior=df)
    # s.calculate_remnant_kick_velocity()

    return s.posterior


def get_angle_bw_vectors(v1, v2):
    unit_vector1 = v1 / np.linalg.norm(v1)
    unit_vector2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    return np.arccos(dot_product)


def plot_corner_of_spins(cos_theta1L_std, cos_theta12_std, save=True):
    bbh_vectors = convert_vectors_to_bbh_param(
        cos_theta1L_std=cos_theta1L_std, cos_theta12_std=cos_theta12_std
    )
    params = [p for p in PARAMS.keys()]
    bbh_vectors = bbh_vectors[params]
    labels = [PARAMS[p]["l"] for p in params]
    range = [PARAMS[p]["r"] for p in params]
    corner.corner(bbh_vectors, **CORNER_KWARGS, labels=labels, range=range)
    if save:
        plt.savefig(
            f"spins_theta1L{cos_theta1L_std:.2f}_theta12{cos_theta12_std:.2f}.png"
        )


def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (
        len_of_longest_samples / len_current_samples
    )


def plot_overlaid_corners(cos_theta1L_std_vals, cos_theta12_std_vals, pltdir):
    params = dict(
        chi_eff=dict(l=r"$\chi_{eff}$", r=(-1, 1)),
        chi_p=dict(l=r"$\chi_{p}$", r=(-1, 1)),
        cos_tilt_1=dict(l=r"$\cos(t1)$", r=(-1, 1)),
        cos_theta_12=dict(l=r"$\cos \theta_{12}$", r=(-1, 1)),
        remnant_kick_mag=dict(l=r"$|\vec{v}_k|\ $km/s", r=(0, 3000)),
    )

    base = convert_vectors_to_bbh_param(cos_theta1L_std=1, cos_theta12_std=1)
    labels = [params[p]["l"] for p in params]
    range = [params[p]["r"] for p in params]
    kwargs = dict(**CORNER_KWARGS, labels=labels, range=range)

    if os.path.isdir(pltdir):
        shutil.rmtree(pltdir)
    os.makedirs(pltdir, exist_ok=False)

    i = 0
    for min_cos_theta1L, min_cos_theta12 in tqdm(
        zip(cos_theta1L_std_vals, cos_theta12_std_vals),
        total=len(cos_theta1L_std_vals),
        desc="Hyper-Param settings",
    ):
        f = f"{pltdir}/{i:02}_p12{min_cos_theta12:.1f}_p1L{min_cos_theta1L:.1f}.png"
        compare = convert_vectors_to_bbh_param(
            cos_theta1L_std=min_cos_theta1L, cos_theta12_std=min_cos_theta12
        )
        compare.to_csv(f.replace(".png", ".csv"))

        fig = corner.corner(base[params], **kwargs, color=BILBY_BLUE_COLOR)
        normalising_weights = get_normalisation_weight(
            len(compare), max(len(compare), len(base))
        )
        corner.corner(
            compare[params],
            fig=fig,
            weights=normalising_weights,
            **kwargs,
            color=VIOLET_COLOR,
        )

        orig_line = mlines.Line2D(
            [], [], color=BILBY_BLUE_COLOR, label="Isotropic Spins"
        )
        weighted_line = mlines.Line2D(
            [],
            [],
            color=VIOLET_COLOR,
            label=f"Adjusted spins $\sigma \cos(12)={min_cos_theta12:.1f}, \sigma \cos(1L)={min_cos_theta1L:.1f}$",
        )
        plt.legend(
            handles=[orig_line, weighted_line],
            fontsize=25,
            frameon=False,
            bbox_to_anchor=(1, len(labels)),
            loc="upper right",
        )
        plt.savefig(f)
        plt.close()
        i += 1


import glob
from bilby_report.tools import image_utils


def save_gif(gifname, outdir="gif", loop=False):
    image_paths = glob.glob(f"{outdir}/*.png")
    gif_filename = os.path.join(outdir, gifname)
    orig_len = len(image_paths)
    image_paths.sort()
    if loop:
        image_paths += image_paths[::-1]
    assert orig_len <= len(image_paths)
    image_utils.make_gif(
        image_paths=image_paths, duration=50, gif_save_path=gif_filename
    )
    print(f"Saved gif {gif_filename}")


if __name__ == "__main__":
    r = SphereAngleAnimation()

    # varying = list(np.arange(0, 2.1, 0.5))
    # constant = [1 for i in range(len(varying))]
    #
    # outdir = "../output/vary_12"
    # plot_overlaid_corners(cos_theta1L_std_vals=constant,
    #                       cos_theta12_std_vals=varying, pltdir=outdir)
    # save_gif("vary_12.gif", outdir=outdir, loop=True)
    #
    # outdir = "../output/vary_1L"
    # plot_overlaid_corners(cos_theta1L_std_vals=varying,
    #                       cos_theta12_std_vals=constant, pltdir=outdir)
    # save_gif("vary_1L.gif", outdir=outdir, loop=True)
