# Robust PCA
C++ implementation of Robust Orthonormal Subspace Learning using the Armadillo
linear algebra library.

## Contents

+ [Description](#description)
+ [Installation](#installation)
+ [Usage](#usage)

## Description

This is a C++ implementation of the Robust Orthonormal Subspace Learning (ROSL) algorithm [1].
ROSL solves the robust PCA problem, recovering a low-rank matrix **A**
from the corrupted observation **X** according to:

<img src="http://i.imgur.com/76Wse2e.png" width="360">

where **E** is the sparse error term. ROSL incorporates a memory-efficient method for recovering **A** from a sub-sample
of the matrix **X**.

[1] X Shu, F Porikli, N Ahuja. (2014) "Robust Orthonormal Subspace Learning: Efficient Recovery of Corrupted Low-rank Matrices". ([paper](http://dx.doi.org/10.1109/CVPR.2014.495))<br/>

## Installation

**Dependencies**

This library makes use of the **[Armadillo](http://arma.sourceforge.net)** C++ linear algebra library,
which needs to be installed first. It is recommended that you use a high-speed replacement for
LAPACK and BLAS such as OpenBLAS, MKL or ACML; more information can be found in the [Armadillo
FAQs](http://arma.sourceforge.net/faq.html#dependencies).

**Building from source**

To build the library, unpack the source and `cd` into the unpacked directory, then type `make`:

```bash
$ tar -xzf robustpca.tar.gz
$ cd rosl
$ make
```

This will generate a C++ library called `librosl.so`, which is called by the Python module `pyrosl`.

## Usage

For a corrupted observation matrix **X**, one can run the ROSL algorithm with the following required
parameters:

```python
import pyrosl

example_rosl = pyrosl.ROSL(
    method='full',
    rank = 5,
    reg = 0.1
)
example_rosl.fit_transform(X)

A = example_rosl.model_
E = example_rosl.residuals_

```

A simple Python script is included with examples of both ROSL and the fast ROSL+ algorithms, as well
as further optional parameters. It can be run on the command line with:

```bash
$ python example.py
```

Further documentation can be found in the file `pyrosl.py`.

## Todo

+ Write tests
