#!/usr/bin/env python

from distutils.core import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='spectroscopy',
      version='0.10',
      description='Spectroscopy of systems with explicit vibrational degrees of Freedom',
      author='Joseph Goodknight',
      author_email='joey.goodknight@gmail.com',
      url='https://github.com/jgoodknight/spectroscopy/',
      long_description=long_description,
      license="MIT",
      packages=['spectroscopy', 'spectroscopy.experiments'],
      package_dir = {'spectroscopy': 'src'},
      install_requires=['numpy', 'scipy', 'matplotlib']
     )
