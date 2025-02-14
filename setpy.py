from setuptools import setup, find_packages

setup(
    name="wasp",
    version="0.1.0",
    description="A package for weighted averaging of MO coefficients and MCPDFT energy calculations",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "h5py",
        "pyscf",
        "ase",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)

