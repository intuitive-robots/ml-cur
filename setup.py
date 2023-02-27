from setuptools import find_packages, setup

setup(
    name="ml_cur",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["numpy", "nlopt", "scikit-learn", "autograd", "scipy"],
)
