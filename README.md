# ideep: Intel deep learning extension for python

'ideep' is a python module for accelerating deep learning workload. Currently it uses intel MKL and MKL-DNN as acceleration engine. The interface object called Compute Complex (CC), support convolution, deconvolution, relu and linear. Refer example directory for more information.

## Requirements

'ideep' preview is tested on Ubuntu and OS X, and examples are implemented as a suggestion for its integration of chainer v3.0.0.

- Python 2.7.6+, 3.6.0+
- Chainer v3.0.0
- Numpy 1.9+
- Six 1.9+
- MKL 
- MKL-DNN
- Testing utilities
  - Mock
  - Nose

## Installation

### Install MKL

Download and install intel MKL at https://software.intel.com/en-us/mkl

### Install MKL-DNN

https://github.com/01org/mkl-dnn

#### Source intel mkl library environment

```
. /opt/intel/mkl/bin/mklvar.sh intel64
```

#### Install ideep from source

```
pip install --user .
```

## More information
- github:
