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


Other requirements:
- Testing utilities
  - Gtest
  - pytest

## Installation

### Install setuptools:
If you use old ``setuptools``, upgrade it:

```
pip install -U setuptools
```

### Install python package from the source code:

```
git submodule update --init && mkdir build && cd build && cmake ..
cd ../python
python setup.py install
```
### Install python package via PYPI:
```
pip install ideep4py
```
Suggest installing Python package using [virtualenv](https://packaging.python.org/key_projects/#virtualenv) to avoid installing Python packages globally which could break system tools or other projects.
### Install python package via Conda:

```
conda install -c intel ideep4py
```
### Install python package via Docker: 
We are providing the official Docker images based on different platforms on [Docker Hub](https://hub.docker.com/r/chainer/chainer/tags). 
```
docker pull chainer/chainer:latest-intel-python2
docker run -it chainer/chainer:latest-intel-python2 /bin/bash
```

## More information
- MKL-DNN github: https://github.com/01org/mkl-dnn
- Chainer github: https://github.com/chainer/chainer

## License
MIT License (see `LICENSE` file).
