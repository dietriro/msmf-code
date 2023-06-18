#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="msmfcode",
    version="0.1",
    description="Package containing code for the simulation, evaluation and optimization of multi-scale, multi-field place codes.",
    url="https://github.com/dietriro/msmf-code",
    author="Robin Dietrich",
    packages=find_packages(exclude=[]),
    install_requires=[
        "PyYAML",
        "numpy",
        "colorlog",
        "matplotlib",
        "scipy",
        "Pympler",
        "plotly",
        "pandas",
        "setuptools"
    ],
)
