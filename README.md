# ROSL

Last updated: 13/11/2015

C++ implementation of Robust Orthonormal Subspace Learning (ROSL)[1] using the Armadillo linear algebra library.

> [1] Xianbiao Shu, Fatih Porikli, Narendra Ahuja. "Robust Orthonormal Subspace Learning: Efficient Recovery of Corrupted Low-rank Matrices". 
> *Proc. of International Conference on Computer Vision and Pattern Recognition (CVPR)*, **2014**

## Contents

+ [Installation](#installation)
+ [Using ROSL](#using-pgure-svt)

## Installation

**Dependencies**

PGURE-SVT makes use of the **[Armadillo](http://arma.sourceforge.net)** C++ linear algebra library, 
which needs to be installed first.

**Building from source**

To build ROSL, unpack the source and `cd` into the unpacked directory:

```
$ tar -xzf rosl.tar.gz
$ cd rosl
```

The next step is to configure the build, and then compile it. This will generate a C++
library called `librosl.so`:

```
make
```

## Using ROSL

*Python description to go here*
