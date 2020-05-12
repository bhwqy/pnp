#!/bin/bash
if test -d ./build; then
    echo "build exists!"
    exit 0
fi
mkdir build
cd ./build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/home/qy101/Desktop/vcpkg/scripts/buildsystems/vcpkg.cmake
make
