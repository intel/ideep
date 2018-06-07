# MKL-DNN bridge for Chainer

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

```
git submodule update --init && mkdir build && cd build && cmake ..
cd ../python
python setup.py install
```

## More information
- MKL-DNN github: https://github.com/01org/mkl-dnn
- Chainer github: https://github.com/chainer/chainer

## License
MIT License (see `LICENSE` file).
