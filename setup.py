"""
  Pythia Mill
"""

from setuptools import setup, find_packages

from codecs import open
import os.path as osp


here = osp.abspath(osp.dirname(__file__))

with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


setup(
  name='abo',

  version='1.0.0',

  description="""Adversarial Bayesian Optimization.""",

  long_description=long_description,

  url='https://github.com/maxim-borisyak/abo',

  author='Maxim Borisyak',
  author_email='mborisyak at hse dot ru',

  maintainer = 'Maxim Borisyak',
  maintainer_email = 'mborisyak at hse dot ru',

  license='MIT',

  classifiers=[
    'Development Status :: 4 - Beta',

    'Intended Audience :: Science/Research',

    'Topic :: Scientific/Engineering :: Physics',

    'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 3',
  ],

  keywords='Bayesian Optimization',

  packages=find_packages('src'),
  package_dir={'': 'src'},

  extras_require={
    'dev': ['check-manifest'],
    'test': ['nose>=1.3.0'],
  },

  install_requires=[
    'numpy',
    'scikit-optimize',
    'scipy',
    'pythia-mill'
  ],
)
