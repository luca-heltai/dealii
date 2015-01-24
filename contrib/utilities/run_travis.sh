#!/bin/sh
echo "Building and running mini tests."
export CC=clang
export CXX=clang++
mkdir build && \
cd build && \
cmake \
-GNinja \
-DCMAKE_INSTALL_PREFIX=/home/travis/dealii-inst \
../ && \
ninja -j4 install && \
tar cfz /home/travis/dealii-travis-CI-build.tgz /home/travis/dealii-inst
