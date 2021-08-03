# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(
    "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle")


def plot():
    """Plots a scatter plot."""
    sample_x = np.random.normal(4, 0.1, 500)
    sample_y = np.random.normal(4, 0.1, 500)
    fig, ax = plt.subplots()
    ax.plot(sample_x, sample_y, '.')
    fig.show()


def main():
    plot()


if __name__ == "__main__":
    main()
