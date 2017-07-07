# ideep: Intel deep learning extension for python

Intel deep learning extension for python is a python module for collection of accelerated deep learning operations like convolution, deconvolution, relu etc. It uses intel MKL and MKL-DNN as acceleration engine. The operator object called Compute Complex (CC), each operator are implemented as one Compute Complex, and its tensor oprand is called 'MD-Array'. 'MD-Array' supports python new buffer protocol and operates compatibily with NumPY ND-Array.

Refer example and tests directories for more information

## Requirements

This preview version of ideep is tested on Ubuntu 14.04 and OS X, and examples are implemented as a suggestion for its integration of chainer v3.0.0.

Minimum requirements:
- Python 2.7.6+, 3.6.0+
- Chainer v3.0.0
- Numpy 1.9+
- Six 1.9+
- MKL 2018 Initial Release 
- MKL-DNN 0.1+
- Swig 3.0.12
- Glog 0.3.5
- Cmake 2.8.0
- Doxygen 1.8.5
- C++ compiler with C++11 standard support

Requirements for some features:
- Testing utilities
  - Mock
  - Nose

## Installation
### Source intel mkl library environment

```
. /opt/intel/mkl/bin/mklvar.sh intel64
```

### Install MKL

Download and install intel MKL at https://software.intel.com/en-us/mkl

### Install MKL-DNN

Refer https://github.com/01org/mkl-dnn for install instruction

### Install ideep

If you use old ``setuptools``, upgrade it:

```
pip install -U setuptools
```

Then, install ideep via PyPI:
```
pip install ideep
```

You can also install ideep from the source code:
```
python setup.py install
```

## More information
- MKL site: https://software.intel.com/en-us/mkl
- MKL-DNN github: https://github.com/01org/mkl-dnn
- ideep github: https://github.com/intel/ideep.git
- Chainer github: https://github.com/pfnet/chainer

## License
MIT License (see `LICENSE` file).
