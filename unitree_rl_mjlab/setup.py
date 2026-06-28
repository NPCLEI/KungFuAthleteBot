"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "loguru",
    "numpy>=1.23.5,<2",
    "scipy",
    "tensordict",
    "tqdm",
    "tyro>=1.0.0",
]

setup(
    name="unitree_rl_mjlab",
    packages=find_packages(),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
