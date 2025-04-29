echo "This script should build your project now..."
#!/bin/bash

mkdir -p build

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo "Running on Windows with Ninja..."
    cmake -G "Ninja" -S src/ -B build/
    ninja -C build
else
    echo "Running on Linux/Unix..."
    cmake -S src/ -B build/
	cd build
    make
fi
