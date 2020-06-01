# -*- coding: utf-8 -*-
# Copyright 2015-2020 Tom Furnival
#
# This file is part of robustpca.
#
# robustpca is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# robustpca is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with robustpca.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

extensions = [
    Extension(
        "robustpca._rosl",
        sources=["robustpca/_rosl.pyx"],
        include_dirs=["robustpca/", np.get_include()],
        libraries=["openblas", "lapack", "armadillo"],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-fPIC",
            "-Wall",
            "-std=c++11",
            "-march=native",
            "-D NPY_NO_DEPRECATED_API",
        ],
    ),
]

exec(open("robustpca/release_info.py").read())

setup(
    name="robustpca",
    version=version,
    description="Robust PCA implementation in Python/C++",
    author=author,
    author_email=email,
    license=license,
    url=url,
    long_description=open("README.md").read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(),
    install_requires=["numpy", "scipy", "scikit-learn"],
    setup_requires=["wheel", "auditwheel", "cython"],
    package_data={"": ["LICENSE", "README.md"], "robustpca": ["*.py"]},
    ext_modules=cythonize(extensions),
)
