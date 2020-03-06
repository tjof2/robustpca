#!/bin/sh
#   This script builds from source:
#       - Armadillo 9.800.2

set -ex

wget http://sourceforge.net/projects/arma/files/armadillo-9.800.2.tar.xz
tar -xvf armadillo-9.800.2.tar.xz > log-file 2>&1
cd armadillo-9.800.2
cmake .
make
sudo make install
