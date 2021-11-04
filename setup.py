""""Distribution setup"""

import os

from setuptools import setup, find_packages

import versioneer

ROOT = os.path.abspath(os.path.dirname(__file__))

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="FORCE",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Forecasting Offshore wind Reductions in Cost of Energy",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "pyyaml",
        "orbit-nrel"
    ],
)
