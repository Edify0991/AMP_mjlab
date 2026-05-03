"""Installation script for the 'wbc_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
]

# Installation operation
setup(
    name="wbc_mjlab",
    packages=find_packages(include=["src", "src.*", "rsl_rl", "rsl_rl.*"]),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
