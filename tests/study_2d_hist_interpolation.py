import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from fast_histogram import histogram2d
x = np.random.normal(0, 1, 10000)
y = np.random.normal(0, 1, 10000)

X = np.linspace(-2, 2, 10000)
Y = np.linspace(-2, 2, 10000)
X, Y = np.meshgrid(X, Y)


def convert_bin_edges_to_points(arr):
    return np.array([(i + j) / 2.0 for i, j in zip(arr[:-1], arr[1:])])


def convert_meshgrid_to_arrays(X, Y, Z):
    """Converts (n,m) arrays to array of (n*m) enteries"""
    return X.flatten(), Y.flatten(), Z.flatten()

# if density:
#     # calculate the probability density function
#     s = hist.sum()
#     for i in _range(D):
#         shape = np.ones(D, int)
#         shape[i] = nbin[i] - 2
#         hist = hist / dedges[i].reshape(shape)
#     hist /= s

H, xedges, yedges = histogram2d(x, y, normed=True)
xpos, ypos = np.meshgrid(convert_bin_edges_to_points(xedges),
                         convert_bin_edges_to_points(yedges))

fig, ax = plt.subplots(nrows=2, ncols=2)
# Plot the model function and the randomly selected sample points
ax[0, 0].contourf(xpos, ypos, H)
ax[0, 0].scatter(x, y, alpha=0.2, marker='.')
ax[0, 0].set_title('Sample points on f(X,Y)')

# Interpolate using three different methods and plot_posterior_predictive_check
# for i, method in enumerate(('nearest', 'linear', 'cubic')):
#     train_x, train_y, train_z = convert_meshgrid_to_arrays(xpos, ypos, H)
#     Ti = griddata((train_x, train_y), train_z, (X, Y), method=method)
#     r, c = (i + 1) // 2, (i + 1) % 2
#     ax[r, c].contourf(X, Y, Ti)
#     ax[r, c].set_title("method = '{}'".format(method))

plt.tight_layout()
plt.show()
