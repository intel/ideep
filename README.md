# iDeep: Intel Deep Learning Package

Intel Deep Learning Package (iDeep) is an open source performance library of primitives for accelerating deep learning frameworks on Intel Architecture. iDeep provides user-friendly API and highly tuned implementations for DNN standard routines.

The package provides C and Python API.

## iDeep Python Package (ideep4py) Requirements

We recommend these Linux distributions.
- Ubuntu 14.04/16.04 LTS 64bit
- CentOS 7 64bit

The following versions of Python can be used:
- 2.7.5+, 3.5.2+, and 3.6.0+

Above recommended environments are tested. We cannot guarantee that ideep4py works on other environments including Windows and macOS, even if ideep4py looks running correctly.


Minimum requirements:
- Numpy 1.9+
- Six 1.9+
- Swig 3.0.12
- Glog 0.3.5
- Cmake 2.8.0
- Doxygen 1.8.5
- C++ compiler with C++11 standard support

## Installation of ideep4py

If you use old ``setuptools``, upgrade it:

```
pip install -U setuptools
```

Then, install ideep from the source code:
```
python setup.py install
```

Use pip to uninstall ideep4py:

```sh
$ pip uninstall ideep4py
```

## More information
- ideep github: https://github.com/intel/ideep.git

## License
MIT License (see `LICENSE` file).
