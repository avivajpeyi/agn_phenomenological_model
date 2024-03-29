#!/usr/bin/env python

# Inspired by:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/

import codecs
import os
import re

from setuptools import find_packages, setup

# PROJECT SPECIFIC

NAME = "agn_utils"
PACKAGES = find_packages(where=".")
META_PATH = os.path.join("agn_utils", "__init__.py")
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
]
INSTALL_REQUIRES = [
    "gwpopulation",
    "surfinbh",
    "deepdiff",
    "sklearn",
    "joblib",
    "fpdf",
    "gputil",
    "psutil",
    "tabulate",
    "lalsuite>=7.0",
    "precession",
    "pycondor",
    "bilby_pipe",
    "parmap"
]

EXTRA_REQUIRE = {"test": ["pytest>=3.6", "testbook>=0.2.3"]}
EXTRA_REQUIRE["dev"] = EXTRA_REQUIRE["test"] + [
    "pre-commit",
    "flake8",
    "black",
    "isort",
]

# END PROJECT SPECIFIC


HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        version=find_meta("version"),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRE,
        classifiers=CLASSIFIERS,
        zip_safe=True,
        entry_points={
            "console_scripts": [
                "cdf_check=agn_utils.plotting.cdf_plotter.sigma_cdf_difference_check:main",
                "pe_corner=agn_utils.plotting.pe_corner_plotter.make_pe_corner:main",
                "plot_corners=agn_utils.plot_corners:main",
                "pbilby_pe_jobgen=agn_utils.pe_setup.pbilby_jobgen_cli:main",
                "draw_population_samples=agn_utils.pe_setup.make_population_files:main",
                "clean_dat_file=agn_utils.pe_setup.clean_inj_file:main",
                "reweight_res=agn_utils.pe_postprocessing.rejection_sample_res:main"
            ]
        }
    )
