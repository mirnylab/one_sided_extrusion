#!/usr/bin/env python
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='looplib',
    version='0.1',
    description='A library to simulate loop extrusion on a 1D lattice',
    author='Anton Goloborodko',
    author_email='goloborodko.anton@gmail.com',
    url='https://github.com/golobor/looplib/',
    packages=['looplib'],
    install_requires=['numpy', 'matplotlib'],
    ext_modules = cythonize([
                             'looplib/looptools_c.pyx',
                             'looplib/simlef_mix.pyx',
                             'looplib/simlef_onesided.pyx',
                             'looplib/simlef_paired.pyx',
                             'looplib/simlef_pushers.pyx',
                             'looplib/simlef_mix_slidedistrib.pyx'
                             ]),
    include_dirs=[np.get_include()]
)
