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

Multinode support:

Ideep provide non-blocking multinode data parallelism support.  The system is requried to meet MPICH dependency and user needs to replace the cmake command in build process:

Make sure your MPI executable is in PATH

```
PATH=$PATH:<path-to-mpiexec>
cmake -Dmultinode=ON ..
```

## More information
- MKL-DNN github: https://github.com/01org/mkl-dnn
- Chainer github: https://github.com/chainer/chainer

## License
MIT License (see `LICENSE` file).
