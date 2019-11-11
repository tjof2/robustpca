# -*- coding: utf-8 -*-
# Copyright 2019 Tom Furnival
#
# This file is part of RobustPCA.
#
# RobustPCA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RobustPCA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RobustPCA.  If not, see <http://www.gnu.org/licenses/>.


from setuptools import setup, find_packages

try:
    with open("robustpca/__init__.py", "r") as f:
        exec(f.read())
except Exception as e:
    print(f"Encountered {type(e).__name__}: {e.args}")

setup(
    name="robustpca",
    version=__version__,
    description="Robust Orthonormal Subspace Learning in Python.",
    author=__author__,
    author_email=__email__,
    license="GPLv3",
    url="https://github.com/tjof2/robustpca",
    long_description=open("README.md").read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(),
    install_requires=[],
    package_data={
        "": ["LICENSE", "readme.rst", "requirements.txt"],
        "robustpca": ["*.py"],
    },
)
