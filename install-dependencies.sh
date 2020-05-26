#!/bin/sh
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

set -ex
mkdir build_deps
cd build_deps

# Versions to install
ARMA_VER="9.880.1"

# Armadillo
wget http://sourceforge.net/projects/arma/files/armadillo-${ARMA_VER}.tar.xz
tar -xvf armadillo-${ARMA_VER}.tar.xz > arma.log 2>&1
cd armadillo-${ARMA_VER}
cmake .
make
sudo make install
cd ../

# Tidy-up
cd ../
sudo rm -rf build_deps/
sudo ldconfig