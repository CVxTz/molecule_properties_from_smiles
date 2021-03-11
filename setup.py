import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="smiles_mol",
    version="0.0.1",
    author="Youness MANSAR",
    author_email="mansaryounessecp@gmail.com",
    description="Predict Molecular properties",
    license="MIT",
    keywords="tabular",
    url="https://github.com/CVxTz/molecule_properties_from_smiles",
    packages=["smiles_mol"],
    entry_points={"console_scripts": ["smiles_mol=smiles_mol.main:cli"]},
    classifiers=[
        "License :: MIT",
    ],
)