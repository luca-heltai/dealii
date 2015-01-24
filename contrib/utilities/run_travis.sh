#!/bin/sh

PRG=$PWD/programs

mkdir build
cd build
export PATH=$PRG/cmake/bin:$PATH
export PATH=$PRG/ninja:$PATH
cmake \
    -G Ninja \
    -D CMAKE_INSTALL_PREFIX:PATH=$PRG/dealii \
    -D CMAKE_CXX_FLAGS:STRING=-w \
    -D DEAL_II_WITH_MPI:BOOL=OFF \
    -D DEAL_II_WITH_THREADS:BOOL=OFF \
    .. 
ninja -j4 install
tar cfz $PRG/dealii-trilinos-serial-CI-build.tgz $PRG/dealii
ctest -j4 -V
