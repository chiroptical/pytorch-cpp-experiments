Build
-----

1. Get pybind11 `git clone https://github.com/pybind/pybind11`
2. Get libtorch `wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip` and `unzip` it
3. Build, I built using `python` from inside a virtual env

```console
mkdir build
cd build
cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_PREFIX_PATH=$(pwd)/../libtorch ..
cmake --build . --config Release
```

Next Steps
----------

- Get a basic class built for running a model
- Wrap class in pybind11
- Call pybind11 wrapped model with actual Torch inputs
- Benchmark for any differences
