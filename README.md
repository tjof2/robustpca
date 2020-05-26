
[![Build Status](https://travis-ci.org/tjof2/robustpca.svg?branch=master)](https://travis-ci.org/tjof2/robustpca)
[![DOI](https://zenodo.org/badge/46107795.svg)](https://zenodo.org/badge/latestdoi/46107795)

# Robust PCA
Python and C++ implementation of Robust Orthonormal Subspace Learning using the Armadillo linear algebra library.

- [Robust PCA](#robust-pca)
  - [Description](#description)
  - [Installation](#installation)
  - [Usage](#usage)

## Description

This is a C++ implementation of the Robust Orthonormal Subspace Learning (ROSL) algorithm [1]. ROSL solves the robust PCA problem, recovering a low-rank matrix **X** and a sparse error matrix **E** from the corrupted observations **Y** according to **Y=X+E**. ROSL also incorporates a memory-efficient method for recovering **A** from a sub-sample of the matrix **X**.

> [1] X. Shu, F. Porikli and N. Ahuja, "Robust Orthonormal Subspace Learning: Efficient Recovery of Corrupted Low-Rank Matrices," 2014 IEEE Conference on Computer Vision and Pattern Recognition, Columbus, OH, 2014, pp. 3874-3881, DOI: [10.1109/CVPR.2014.495](http://dx.doi.org/10.1109/CVPR.2014.495).

## Installation

**Dependencies**

This library makes use of the **[Armadillo](http://arma.sourceforge.net)** C++ linear algebra library,
which needs to be installed first. It is recommended that you use a high-speed replacement for
LAPACK and BLAS such as OpenBLAS, MKL or ACML; more information can be found in the [Armadillo
FAQs](http://arma.sourceforge.net/faq.html#dependencies).

One way to install the latest version of Armadillo is to run:

```bash
$ tar -xzf robustpca.tar.gz
$ cd robustpca/src
$ ./install-dependencies.sh
```

**Building from source**

To build the library, unpack the source and `cd` into the unpacked directory, then type `make`:

```bash
$ tar -xzf robustpca.tar.gz
$ cd robustpca/src
$ make
```

This will generate a C++ library called `librosl.so`, which is called by the Python module `robustpca`.

## Usage
_To be completed_

Copyright (C) 2015-2020 Tom Furnival. robustpca is released free of charge under the GNU General Public License (GPLv3).

