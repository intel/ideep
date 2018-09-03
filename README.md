# Chainer Backend for Intel Architecture

A Chainer module providing numpy like API and DNN acceleration using MKL-DNN.


## Requirements

This preview version is tested on Ubuntu 16.04 and OS X.

Minimum requirements:
- cmake 3.0.0+
- C++ compiler with C++11 standard support (GCC 5.3+ if you want to build tests)
- Python 2.7.6+, 3.5.2+, 3.6.0+
- Numpy 1.13
- Swig 3.0.12
- Doxygen 1.8.5
- (optional) MPICH devel 3.2


Other requirements:
- Testing utilities
  - Gtest
  - pytest

## Installation

If you use old ``setuptools``, upgrade it:

```
pip install -U setuptools
```

Install python package from the source code:

CentOS:
```
git submodule update --init && mkdir build && cd build && cmake3 ..
cd ../python
python setup.py install

```
Other:
```
git submodule update --init && mkdir build && cd build && cmake ..
cd ../python
python setup.py install
```

**Multinode support:**

IDeep provide non-blocking multinode data parallelism support.  The system is requried to meet MPICH dependency and user needs to replace the cmake command in build process:

Make sure your MPI executable is in PATH:

```
PATH=$PATH:<path-to-mpiexec>
# use the following line when you execute cmake or cmake3
# CentOS:
cmake3 -Dmultinode=ON ..
# Other:
cmake -Dmultinode=ON ..
```

Execute the test:
```
cd total_reduce/test
mpirun -N 4 python3 test_1payload_inplace.py
```
The commands above will start 4 MPI processes on your machine and conduct a blocking allreduce operation among all 4 processes.  To test it in a real multinode environment, compile your <hostlist> file and use the following commands:
```
cd total_reduce/test
mpirun -f <hostlist> -N 4 python3 test_1payload_inplace.py
```

## More information
- MKL-DNN github: https://gitlab.devtools.intel.com/ipl/mkl-dnn
- Chainer github: https://github.com/chainer/chainer

## License
MIT License (see `LICENSE` file).
