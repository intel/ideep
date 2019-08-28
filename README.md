# Chainer Backend for Intel Architecture

A Chainer module providing numpy like API and DNN acceleration using MKL-DNN.


## Requirements

This preview version is tested on Ubuntu 16.04, Centos 7.4 and OS X.

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

### Install setuptools:
If you use old ``setuptools``, upgrade it:

```
pip install -U setuptools
```

### Install python package from the source code:

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
### Install python package via PYPI:
```
pip install ideep4py
```
Since Python3.7 doesn't work with numpy==1.13, we built iDeep4py Python3.7 wheel based on numpy==1.16.0, remember to upgrade numpy version to 1.16.0 before install iDeep4py Python3.7 wheel.
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

## Multinode support:

Non-blocking multinode data parallelism is supported.  The system is requried to meet MPICH dependency and user needs to replace the cmake command in build process:

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
- MKL-DNN github: https://github.com/01org/mkl-dnn
- Chainer github: https://github.com/chainer/chainer

## License
MIT License (see `LICENSE` file).
