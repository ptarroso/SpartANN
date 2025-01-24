#!/usr/bin/env python

from setuptools import setup, find_packages
import os

def get_version():
    version = {}
    version_file = os.path.join(os.path.dirname(__file__), "spartann", "version.py")
    with open(version_file, "r") as stream:
        exec(stream.read(), version)
    return(version["__version__"])

setup(name='SpartANN',
      version=get_version(),
      description='Spectral Pattern Analysis Remote-sensing Tool with Artificial Neural Networks',
      author='Pedro Tarroso',
      author_email='ptarroso@cibio.up.pt',
      url='https://github.com/ptarroso/SpartANN',
      classifiers=[
          "Programming Language :: Python :: 3.12",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering :: GIS",
      ],
      packages=find_packages(),
      )
