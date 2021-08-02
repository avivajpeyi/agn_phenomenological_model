import math
from math import sin, cos

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import interactive
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
""" Plot polygon (face colour must be set afterwards, otherwise it over-rides the transparency)
    https://stackoverflow.com/questions/18897786/transparency-for-poly3dcollection-plot-in-matplotlib """

interactive(True)
matplotlib.use('macosx')
matplotlib.pyplot.ion()


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


setattr(Axes3D, 'annotate3D', _annotate3D)
setattr(Axes3D, 'arrow3D', _arrow3D)


def plot_arc3d(vector1, vector2, radius=0.2, fig=None, colour='C0', vector_colors=[], vector_labels=[], arc_label="",
               arrow_kwargs={}):
    """ Plot arc between two given vectors in 3D space. """

    """ Confirm correct input arguments """
    assert len(vector1) == 3
    assert len(vector2) == 3

    """ Calculate vector between two vector end points, and the resulting spherical angles for various points along 
        this vector. From this, derive points that lie along the arc between vector1 and vector2 """
    v = [i - j for i, j in zip(vector1, vector2)]
    v_points_direct = [(vector2[0] + v[0] * l, vector2[1] + v[1] * l, vector2[2] + v[2] * l) for l in np.linspace(0, 1)]
    v_phis = [math.atan2(v_point[1], v_point[0]) for v_point in v_points_direct]
    v_thetas = [math.acos(v_point[2] / np.linalg.norm(v_point)) for v_point in v_points_direct]

    v_points_arc = [(
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta)
    )
        for theta, phi in zip(v_thetas, v_phis)]
    v_points_arc.append((0, 0, 0))

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', frame_on=True)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.set_zlim(0, 0.8)
    else:
        ax = fig.gca()

    points_collection = Poly3DCollection([v_points_arc], alpha=0.4)
    points_collection.set_facecolor(colour)
    ax.add_collection3d(points_collection)

    if len(vector_labels) == len(vector_colors) == 2:
        for v, label, col, kwg in zip([vector1, vector2], vector_labels, vector_colors, arrow_kwargs):
            ax.annotate3D(label, v, xytext=(3, 3), textcoords='offset points', color=col)
            if kwg.get("mutation_scale", None) is None:
                kwg['mutation_scale'] = 10
            ax.arrow3D(0, 0, 0, v[0], v[1], v[2], color=col, **kwg)

    point_id = int(len(v_points_arc) / 2)
    ax.annotate3D(arc_label, v_points_arc[point_id], xytext=(0, 0), textcoords='offset points', color=colour)

    if arc_label:
        legend_label = f"{arc_label} = {np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1)* np.linalg.norm(vector2))) / np.pi : 0.2f} $\pi$"
        print(legend_label)
        ax.plot([],[], color=colour, label=legend_label)
    return fig


def plot_bbh_spins(s1, s2):
    plt.ion()
    l = [0, 0, 0.6]
    lx = [0.6, 0, 0]
    dashed = {"linestyle": "dashed", "arrowstyle": "-|>"}
    fig = plot_arc3d(
        vector1=l,
        vector2=lx,
        radius=0.0,
        vector_labels=[r'$\hat{L}$', r'$\hat{L}_x$'],
        vector_colors=['k', 'k'],
        arc_label=r"",
        arrow_kwargs=[{}, dashed]
    )
    fig = plot_arc3d(
        vector1=s1,
        vector2=s2,
        radius=0.6,
        vector_labels=[r'$\hat{s}_1$', r"$\hat{s}_{2}$"],
        vector_colors=['b', 'r'],
        arc_label=r"$\theta_{12}$",
        colour="green",
        fig=fig,
        arrow_kwargs=[{}, {}]
    )

    fig = plot_arc3d(
        vector1=[s1[0], s1[1], 0],
        vector2=[s2[0], s2[1], 0],
        radius=0.6,
        vector_labels=[r'$\hat{s}_{1p}$', r"$\hat{s}_{2p}$"],
        vector_colors=['b', 'r'],
        arc_label=r"$\phi_{12}$",
        colour="orange",
        fig=fig,
        arrow_kwargs=[dashed, dashed]
    )

    fig = plot_arc3d(
        vector1=lx, vector2=[s2[0], s2[1], 0],
        radius=0.4, arc_label=r"$\phi_{2}$", colour="red", fig=fig,
    )
    fig = plot_arc3d(
        vector1=lx, vector2=[s1[0], s1[1], 0],
        radius=0.2, arc_label=r"$\phi_{1}$", colour="blue", fig=fig,
    )

    fig = plot_arc3d(
        vector1=l, vector2=s2,
        radius=0.3, arc_label=r"$\theta_{2}$", colour="red", fig=fig,
    )
    fig = plot_arc3d(
        vector1=l, vector2=s1,
        radius=0.3, arc_label=r"$\theta_{1}$", colour="blue", fig=fig,
    )
    fig.gca().legend(bbox_to_anchor=(1,1), loc="upper left")

if __name__ == '__main__':
    s1  = [0.7, 0.7, 0.3]
    s1 /= np.linalg.norm(s1)
    s2 = [0, 0.7, 0.7]
    s2 /= np.linalg.norm(s2)
    print(s1)
    print(s2)
    plot_bbh_spins(s1, s2)


    plt.show(block=True)
