# ideep: Intel deep learning extension for python

'ideep' is a python module for accelerating deep learning workload. Currently it uses intel MKL and MKL-DNN as acceleration engine. The interface object called Compute Complex (CC), support convolution, deconvolution, relu and linear. Refer example directory for more information.

## Requirements

'ideep' preview is tested on Ubuntu 14.04 and OS X, and examples are implemented as a suggestion for its integration of chainer v3.0.0.

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
-
-Download and install intel MKL at https://software.intel.com/en-us/mkl
-
### Install MKL-DNN
-
-Refer https://github.com/01org/mkl-dnn for install instruction
-
### ideep Installation

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
Apache License Version 2.0 (see `LICENSE` file).
